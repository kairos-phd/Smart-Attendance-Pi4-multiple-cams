"""Real loopback API/SQLite smoke tests for the standalone collector."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sqlite3
import subprocess
import sys
import signal
import time
import pytest
import threading

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/stress_recognition.py'


def test_collector_is_available():
    assert SCRIPT.is_file(), 'Standalone collector has not been implemented'


@pytest.mark.parametrize('mode', ['normal', 'enable_failure', 'interrupt'])
def test_short_soak_excludes_history_and_restores_control(tmp_path, mode):
    db = tmp_path / 'smart_attendance.sqlite3'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE recognition_events (observed_at TEXT, decided_at TEXT, status TEXT, name TEXT)')
        conn.execute("INSERT INTO recognition_events VALUES ('2026-01-01T00:00:00Z','2026-01-01T00:00:01Z','accepted','HISTORICAL')")
    state = {'enabled': False, 'patches': [], 'health_reads': 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == '/api/control':
                result = {'recognition_enabled': state['enabled']}
            else:
                state['health_reads'] += 1
                if state['health_reads'] == 1:
                    with sqlite3.connect(db) as conn:
                        conn.execute("INSERT INTO recognition_events VALUES ('2026-01-01T00:00:02Z','2026-01-01T00:00:02.125Z','accepted','PRIVATE')")
                result = {'system': {'cpu_percent': 20, 'process_cpu_percent': 5, 'ram_percent': 40}, 'queue_depth': 2, 'dropped_frames': state['health_reads'], 'model_version': 'legacy', 'inference_alive': True}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def do_PATCH(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['enabled'] = payload['recognition_enabled']
            state['patches'].append(state['enabled'])
            self.send_response(503 if mode == 'enable_failure' and len(state['patches']) == 1 else 200)
            self.end_headers()
            self.wfile.write(json.dumps({'recognition_enabled': state['enabled']}).encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        command = [sys.executable, str(SCRIPT), '--duration', '10' if mode == 'interrupt' else '.3', '--interval', '.1', '--url', f'http://127.0.0.1:{server.server_port}', '--data-root', str(tmp_path), '--output-dir', str(tmp_path/'out'), '--enable-recognition']
        if mode == 'interrupt':
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 5
            while state['health_reads'] < 2 and time.monotonic() < deadline:
                time.sleep(.01)
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=5)
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        else:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == {'normal': 0, 'enable_failure': 1, 'interrupt': 130}[mode], result.stderr
    assert state['patches'] == [True, False]
    output = tmp_path / 'out'
    if mode == 'enable_failure':
        summary = json.loads((output/'summary.json').read_text())
        assert summary['recognition_restored'] is True
        assert any('startup failed' in warning for warning in summary['warnings'])
        return
    events = [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    assert len(events) == 1
    assert events[0]['decision_delay_ms'] == 125
    summary = json.loads((output/'summary.json').read_text())
    assert summary['event_count'] == 1
    assert summary['metrics']['system_cpu_percent_p95'] == 20
    assert summary['metrics']['service_process_cpu_percent_p95'] == 5
    assert summary['recognition_enabled_samples'] >= 1
    assert 'PRIVATE' not in (output/'summary.json').read_text()
    assert 'PRIVATE' not in (output/'events.jsonl').read_text()
    assert summary['recognition_restored'] is True
    if mode == 'normal':
        assert summary['duration_actual_seconds'] >= .3
    else:
        assert summary['interrupted'] is True


def test_remote_origin_rejected_before_network(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), '--url', 'http://192.168.0.208:8080', '--output-dir', str(tmp_path/'out')], capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert 'loopback' in result.stderr


def test_quantile_interpolates():
    spec = importlib.util.spec_from_file_location('stress_recognition', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.percentile([0, 10, 20], .95) == 19


def test_insert_between_maximum_and_read_does_not_replay(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('stress_recognition', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    database = tmp_path / 'events.sqlite'
    with sqlite3.connect(database) as writer:
        writer.execute('CREATE TABLE recognition_events (observed_at TEXT, decided_at TEXT)')
        writer.execute("INSERT INTO recognition_events VALUES ('2026-01-01T00:00:00Z','2026-01-01T00:00:01Z')")
    connect = module.event_connection
    class InsertAfterMaximum:
        def __enter__(self):
            self.connection = connect(database)
            return self
        def __exit__(self, *args):
            self.connection.close()
        @property
        def row_factory(self):
            return self.connection.row_factory
        @row_factory.setter
        def row_factory(self, value):
            self.connection.row_factory = value
        def execute(self, query, *parameters):
            result = self.connection.execute(query, *parameters)
            if 'MAX(rowid)' in query:
                value = result.fetchone()
                result.close()
                with sqlite3.connect(database) as writer:
                    writer.execute("INSERT INTO recognition_events VALUES ('2026-01-01T00:00:02Z','2026-01-01T00:00:03Z')")
                return type('Result', (), {'fetchone': lambda self: value})()
            return result
    monkeypatch.setattr(module, 'event_connection', lambda _: InsertAfterMaximum())
    events, cursor = module.read_events(database, 0)
    monkeypatch.setattr(module, 'event_connection', connect)
    following, _ = module.read_events(database, cursor)
    assert len(events) + len(following) == 2
    assert len({event['cursor'] for event in events + following}) == 2
