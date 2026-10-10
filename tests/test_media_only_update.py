import importlib.util
from pathlib import Path

import pytest

from pi4_attendance import webrtc


def test_media_credentials_do_not_change_attendance_configuration(monkeypatch):
    for key, value in {
        'FIREBASE_SERVICE_ACCOUNT': 'attendance.json',
        'FIREBASE_PROJECT_ID': 'attendance-project',
        'FIREBASE_DATABASE_URL': 'https://attendance.firebaseio.com',
        'WEBRTC_FIREBASE_SERVICE_ACCOUNT': 'media.json',
        'WEBRTC_FIREBASE_PROJECT_ID': 'sicccc-30fed',
        'WEBRTC_FIREBASE_DATABASE_URL': 'https://sicccc-30fed.firebaseio.com',
    }.items():
        monkeypatch.setenv(key, value)
    seen = {}
    def get_app(name):
        seen['lookup'] = name
        raise ValueError('not initialized')
    monkeypatch.setattr(webrtc.firebase_admin, 'get_app', get_app)
    monkeypatch.setattr(webrtc.credentials, 'Certificate', lambda path: path)
    def initialize(certificate, options, name):
        seen.update(certificate=certificate, options=options, name=name)
        return seen
    monkeypatch.setattr(webrtc.firebase_admin, 'initialize_app', initialize)
    app = webrtc.WebRTCManager._get_firebase_app()
    assert app['lookup'] == app['name'] == 'sic-webrtc'
    assert app['certificate'] == 'media.json'
    assert app['options']['projectId'] == 'sicccc-30fed'
    assert app['options']['databaseURL'] == 'https://sicccc-30fed.firebaseio.com'
    assert webrtc.os.environ['FIREBASE_PROJECT_ID'] == 'attendance-project'


def load_updater():
    path = Path(__file__).parents[1] / 'scripts' / 'update_media_only.py'
    assert path.exists(), 'media-only updater is required'
    spec = importlib.util.spec_from_file_location('media_updater', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_media_update_backs_up_only_webrtc_and_preserves_inference(tmp_path):
    updater = load_updater()
    repo = tmp_path / 'repo'
    source_root = repo / 'src' / 'pi4_attendance'
    source_root.mkdir(parents=True)
    old = source_root / 'webrtc.py'
    old.write_text('class PiVideoTrack: pass\nclass WebRTCManager: pass\n')
    engine = source_root / 'face_engine.py'
    engine.write_bytes(b'legacy model loader')
    candidate = Path(webrtc.__file__)
    result = updater.update(repo, candidate, apply=False)
    assert result['changed']
    assert old.read_text().startswith('class PiVideoTrack: pass')
    result = updater.update(repo, candidate, apply=True)
    assert old.read_bytes() == candidate.read_bytes()
    assert Path(result['backup']).read_text().startswith('class PiVideoTrack: pass')
    assert engine.read_bytes() == b'legacy model loader'
    assert updater.update(repo, candidate, apply=True)['changed'] is False


def test_invalid_media_candidate_leaves_current_source_unchanged(tmp_path):
    updater = load_updater()
    root = tmp_path / 'src' / 'pi4_attendance'
    root.mkdir(parents=True)
    current = root / 'webrtc.py'
    current.write_text('class PiVideoTrack: pass\nclass WebRTCManager: pass\n')
    candidate = tmp_path / 'bad.py'
    candidate.write_text('raise RuntimeError("wrong module")')
    with pytest.raises(ValueError):
        updater.update(tmp_path, candidate, apply=True)
    assert current.read_text().startswith('class PiVideoTrack: pass')


def test_legacy_feed_patch_uses_existing_admin_auth_and_reads_unknown(tmp_path):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from pi4_attendance.storage import PiStore
    from test_recognition_event_feed import decision
    updater = load_updater()
    code = '''from fastapi import FastAPI, Depends
from pi4_attendance.auth import build_admin_dependency
def create_app(runtime):
    app = FastAPI()
    require_admin = build_admin_dependency(runtime.settings)
    return app
'''
    assert hasattr(updater, 'patch_event_feed'), 'Legacy event-feed adaptation is required'
    patched = updater.patch_event_feed(code)
    assert updater.patch_event_feed(patched) == patched
    scope = {}
    exec(patched, scope)
    store = PiStore(tmp_path)
    try:
        store.record_decision(decision(), 300)
        runtime = SimpleNamespace(store=store, settings=SimpleNamespace(
            api_auth_mode='shared_token', admin_shared_token='fixture'))
        with TestClient(scope['create_app'](runtime)) as client:
            assert client.get('/api/recognition-events').status_code == 401
            rows = client.get('/api/recognition-events', headers={
                'Authorization': 'Bearer fixture'}).json()
        assert rows[0]['display_status'] == 'unknown'
        assert rows[0]['model_version'] == 'runtime-model'
        assert rows[0]['name'] is None
        assert store.recent_attendance() == []
    finally:
        store.close()


def test_interrupted_two_file_update_restores_all_applied_files(tmp_path, monkeypatch):
    updater = load_updater()
    root = tmp_path / 'src' / 'pi4_attendance'
    root.mkdir(parents=True)
    media = root / 'webrtc.py'
    media.write_text('class PiVideoTrack: pass\nclass WebRTCManager: pass\n')
    api = root / 'api.py'
    api.write_text('from fastapi import FastAPI, Depends\ndef create_app(runtime):\n    app = FastAPI()\n    require_admin = lambda: True\n    return app\n')
    before = {p: p.read_bytes() for p in (media, api)}
    replace = updater.os.replace
    def interrupt_second(source, target):
        if Path(target) == api:
            raise KeyboardInterrupt()
        replace(source, target)
    monkeypatch.setattr(updater.os, 'replace', interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        updater.update(tmp_path, Path(webrtc.__file__), apply=True)
    assert {p: p.read_bytes() for p in before} == before
