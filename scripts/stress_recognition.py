#!/usr/bin/env python3
"""Standalone, standard-library recognition soak collector for existing Pi runtimes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import time
from urllib import parse, request


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = request.build_opener(request.ProxyHandler({}), NoRedirect())
EVENT_FIELDS = {'observed_at', 'decided_at', 'observation_id', 'status', 'camera_id',
                'reason', 'similarity_scaled', 'model_version'}


def api(origin, endpoint, payload=None):
    req = request.Request(origin + endpoint, method='GET' if payload is None else 'PATCH',
                          data=None if payload is None else json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=2) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError('Expected API JSON object')
    return result


def now():
    return datetime.now(timezone.utc).isoformat()


def percentile(values, quantile=.95):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=3)
        return {'returncode': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def pi_state():
    result = {}
    try:
        result['temperature_c'] = float(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
    except (OSError, ValueError):
        pass
    throttle = command(['vcgencmd', 'get_throttled'])
    if throttle.get('returncode') == 0:
        try:
            result['throttled_bits'] = int(throttle['stdout'].split('=', 1)[1], 16)
        except (ValueError, IndexError):
            pass
    return result


def event_connection(database):
    return sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=2)


def initial_cursor(database):
    with event_connection(database) as conn:
        return conn.execute('SELECT COALESCE(MAX(rowid),0) FROM recognition_events').fetchone()[0]


def read_events(database, cursor):
    with event_connection(database) as conn:
        conn.row_factory = sqlite3.Row
        maximum = conn.execute('SELECT COALESCE(MAX(rowid),0) FROM recognition_events').fetchone()[0]
        if maximum < cursor:
            raise ValueError('Event cursor decreased: database was reset; refusing historical replay')
        rows = conn.execute('SELECT rowid AS cursor,* FROM recognition_events WHERE rowid>? ORDER BY rowid', (cursor,)).fetchall()
    events = []
    for row in rows:
        event = {key: row[key] for key in row.keys() if key in EVENT_FIELDS or key == 'cursor'}
        try:
            observed = datetime.fromisoformat(event['observed_at'].replace('Z', '+00:00'))
            decided = datetime.fromisoformat(event['decided_at'].replace('Z', '+00:00'))
            if observed.tzinfo is None or decided.tzinfo is None:
                raise ValueError('Timestamps have no timezone')
            delay = (decided - observed).total_seconds() * 1000
            if delay < 0:
                raise ValueError('Negative decision delay')
            event['decision_delay_ms'] = delay
        except (KeyError, TypeError, ValueError):
            event['decision_delay_ms'] = None
        events.append(event)
    return events, max(maximum, max((row['cursor'] for row in rows), default=cursor))


def write_jsonl(file, record):
    file.write(json.dumps(record, ensure_ascii=False) + '\n')
    file.flush()


def summarize(samples, events, elapsed, requested, warnings, prior, restored, interrupted):
    metrics = {}
    sources = [('system_cpu_percent', ('system', 'cpu_percent')),
               ('service_process_cpu_percent', ('system', 'process_cpu_percent')),
               ('ram_percent', ('system', 'ram_percent')),
               ('service_process_rss_mb', ('system', 'process_rss_mb')),
               ('queue_depth', ('queue_depth',))]
    for name, keys in sources:
        values = []
        for sample in samples:
            value = sample.get('health', {})
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            if numeric(value):
                values.append(value)
        metrics[name + '_p95'] = percentile(values)
    for key in ('system_cpu_percent_p95', 'service_process_cpu_percent_p95', 'queue_depth_p95'):
        if metrics[key] is None:
            warnings.append(key + ' unavailable from this runtime')
    temperatures = [s['pi']['temperature_c'] for s in samples if numeric(s.get('pi', {}).get('temperature_c'))]
    metrics['temperature_peak_c'] = max(temperatures, default=None)
    metrics['decision_delay_ms_p95'] = percentile([e['decision_delay_ms'] for e in events if numeric(e.get('decision_delay_ms'))])
    for key in ('queue_age_ms', 'inference_ms', 'full_processing_ms'):
        values = [s.get('health', {}).get('performance_metrics', {}).get('distributions', {}).get(key, {}).get('p95') for s in samples]
        metrics[key + '_rolling_p95_peak'] = max((v for v in values if numeric(v)), default=None)
    counters = {}
    for label, keys in [('dropped_frames', ('dropped_frames',)), ('telemetry_samples_dropped', ('telemetry', 'samples_dropped'))]:
        values = []
        for s in samples:
            value = s.get('health', {})
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            if numeric(value):
                values.append(value)
        if any(b < a for a, b in zip(values, values[1:])):
            warnings.append(label + ' counter reset; delta unavailable')
            counters[label + '_delta'] = None
        else:
            counters[label + '_delta'] = values[-1] - values[0] if len(values) >= 2 else None
    enabled = sum(s.get('control', {}).get('recognition_enabled') is True for s in samples)
    if enabled != len(samples):
        warnings.append('Recognition was disabled or unknown in one or more samples')
    if not events:
        warnings.append('No new recognition events; old runtimes may not persist recognition_events')
    if any(e.get('decision_delay_ms') is None for e in events):
        warnings.append('Some event decision delays unavailable: invalid/missing timestamps')
    if any(s.get('health', {}).get('inference_alive') is False for s in samples):
        warnings.append('Inference was not alive in one or more samples')
    if any(s.get('pi', {}).get('throttled_bits', 0) for s in samples):
        warnings.append('Pi throttling flags observed; raw bitmask includes current and historical flags')
    return {'duration_requested_seconds': requested, 'duration_actual_seconds': round(elapsed, 3),
            'interrupted': interrupted, 'sample_count': len(samples), 'event_count': len(events),
            'recognition_enabled_samples': enabled, 'recognition_prior_enabled': prior,
            'recognition_restored': restored, 'metrics': metrics, 'counters': counters,
            'event_status_counts': dict(Counter(e.get('status', 'unknown') for e in events)),
            'model_versions': sorted({str(s['health']['model_version']) for s in samples if s.get('health', {}).get('model_version') is not None}),
            'warnings': sorted(set(warnings)),
            'metric_notes': ['p95 uses linear interpolation over collected samples.',
                             'CPU percentages from health: system aggregate 0-100; service process normalized by logical CPU count (0-100).',
                             'Rolling latency p95 peak is the largest reported rolling-window p95, not whole-run p95.',
                             'Decision delay is decided_at minus observed_at; includes decision/voting and is not pure inference latency.',
                             'No labeled ground truth: FAR/FRR and recognition accuracy are not measured.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=1800)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp') / ('sic-recog-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    parser.add_argument('--data-root', type=Path, default=Path('/var/lib/sic-attendance'))
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--enable-recognition', action='store_true', help='Explicitly enable recognition and restore the prior flag on exit')
    parser.add_argument('--journal', action='store_true', help='Collect readable service journal without sudo')
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < .1:
        parser.error('--duration must be finite and at least 0.1 seconds')
    if not math.isfinite(args.interval) or args.interval < .1:
        parser.error('--interval must be finite and at least 0.1 seconds')
    origin = args.url.rstrip('/')
    parsed = parse.urlparse(origin)
    if (parsed.scheme not in {'http', 'https'} or parsed.hostname not in {'localhost', '127.0.0.1', '::1'}
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path):
        parser.error('--url must be a loopback API origin without credentials')
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    start_stamp = now()
    service_start = command(['systemctl', 'show', 'sic-pi4-attendance', '--property=ActiveState,SubState,MainPID,NRestarts', '--no-pager'])
    started = time.monotonic()
    samples, events, warnings = [], [], []
    prior = None
    restore_needed = False
    restored = None
    interrupted = False
    exit_code = 0
    database = args.data_root / 'smart_attendance.sqlite3'
    cursor = None
    try:
        cursor = initial_cursor(database)
    except (OSError, sqlite3.Error) as exc:
        warnings.append('SQLite events unavailable: ' + str(exc))
    with (args.output_dir / 'health.jsonl').open('w') as health_file, (args.output_dir / 'events.jsonl').open('w') as event_file:
        try:
            control = api(origin, '/api/control')
            prior = control.get('recognition_enabled')
            if args.enable_recognition:
                if not isinstance(prior, bool):
                    raise ValueError('Prior recognition flag unavailable; refusing to change it')
                restore_needed = not prior
                if restore_needed:
                    api(origin, '/api/control', {'recognition_enabled': True})
                if api(origin, '/api/control').get('recognition_enabled') is not True:
                    raise ValueError('Recognition enable could not be verified')
            while True:
                sample = {'timestamp': now(), 'elapsed_seconds': round(time.monotonic() - started, 3), 'pi': pi_state()}
                for endpoint, key in [('/health', 'health'), ('/api/control', 'control')]:
                    try:
                        sample[key] = api(origin, endpoint)
                    except (OSError, ValueError) as exc:
                        sample[key + '_error'] = str(exc)
                        warnings.append(key + ' API samples failed')
                        exit_code = 1
                samples.append(sample)
                write_jsonl(health_file, sample)
                if cursor is not None:
                    try:
                        new_events, cursor = read_events(database, cursor)
                        for event in new_events:
                            write_jsonl(event_file, event)
                        events.extend(new_events)
                    except (OSError, sqlite3.Error, ValueError) as exc:
                        warnings.append('SQLite event sampling failed: ' + str(exc))
                        cursor = None
                remaining = args.duration - (time.monotonic() - started)
                if remaining <= 0:
                    break
                time.sleep(min(args.interval, remaining))
        except KeyboardInterrupt:
            interrupted = True
            exit_code = 130
        except (OSError, ValueError) as exc:
            warnings.append('Collector startup failed: ' + str(exc))
            exit_code = 1
        finally:
            if restore_needed:
                try:
                    api(origin, '/api/control', {'recognition_enabled': prior})
                    restored = api(origin, '/api/control').get('recognition_enabled') is prior
                    if not restored:
                        raise ValueError('Restored flag did not match prior flag')
                except (OSError, ValueError, KeyboardInterrupt) as exc:
                    restored = False
                    warnings.append('RECOGNITION RESTORE FAILED: ' + str(exc))
                    exit_code = 1
    elapsed = time.monotonic() - started
    service = command(['systemctl', 'show', 'sic-pi4-attendance', '--property=ActiveState,SubState,MainPID,NRestarts', '--no-pager'])
    (args.output_dir / 'service.json').write_text(json.dumps({'start': service_start, 'end': service}, indent=2) + '\n')
    if service.get('returncode') != 0:
        warnings.append('Service process state unavailable (see service.json)')
    if args.journal:
        journal = command(['journalctl', '-u', 'sic-pi4-attendance', '--since', start_stamp, '--until', now(), '--no-pager', '-o', 'short-iso', '-n', '10000'])
        (args.output_dir / 'service-journal.txt').write_text(journal.get('stdout', '') + '\n' + journal.get('stderr', journal.get('error', '')))
        if journal.get('returncode') != 0:
            warnings.append('Journal unavailable without elevated permission')
    summary = summarize(samples, events, elapsed, args.duration, warnings, prior, restored, interrupted)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output_dir / 'report.txt').write_text('Recognition soak report\n' + json.dumps(summary, indent=2) + '\n')
    print(str(args.output_dir / 'report.txt'))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
