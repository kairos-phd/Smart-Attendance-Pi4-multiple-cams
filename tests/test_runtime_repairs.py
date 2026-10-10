import asyncio
import builtins
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from starlette.requests import Request

from pi4_attendance import api
from pi4_attendance.cloud import CloudSummary
from pi4_attendance.model_identity import model_identity
from pi4_attendance.service import AttendanceRuntime
from pi4_attendance.webrtc import WebRTCManager


def preview_runtime():
    sequence = [0]
    def latest_sample(camera_id):
        sequence[0] += 1
        return np.zeros((32, 32, 3), np.uint8), sequence[0]
    return SimpleNamespace(
        settings=SimpleNamespace(api_cors_origins=(), api_auth_mode='shared_token',
                                 admin_shared_token='fixture'),
        stop_event=threading.Event(),
        preview=SimpleNamespace(camera_ids=('camera-1',), latest_sample=latest_sample),
    )


@pytest.mark.parametrize('expire', [False, True])
def test_mjpeg_stops_when_session_revoked_or_expired(monkeypatch, expire):
    for key in ('PREVIEW_USERNAME', 'PREVIEW_PASSWORD_HASH', 'PREVIEW_PASSWORD_SALT'):
        monkeypatch.setenv(key, 'fixture')
    runtime = preview_runtime()
    app = api.create_app(runtime)
    token = api._create_preview_session()
    request = Request({'type':'http', 'headers':[(b'cookie',f'{api.PREVIEW_SESSION_COOKIE}={token}'.encode())]})
    endpoint = next(r.endpoint for r in app.routes if r.path == '/preview/{camera_id}')
    response = endpoint('camera-1', request)
    generator = response.body_iterator.ag_frame.f_locals['iterator']
    assert next(generator).startswith(b'--frame')
    if expire:
        api._PREVIEW_SESSIONS[token] = time.time() - 1
    else:
        api._delete_preview_session(token)
    try:
        with pytest.raises(StopIteration):
            next(generator)
    finally:
        runtime.stop_event.set()
        generator.close()
        api._delete_preview_session(token)


def test_preview_cookie_is_secure_by_default(monkeypatch):
    monkeypatch.delenv('PREVIEW_COOKIE_SECURE', raising=False)
    monkeypatch.setenv('PREVIEW_USERNAME','fixture')
    monkeypatch.setenv('PREVIEW_PASSWORD_HASH','fixture')
    monkeypatch.setenv('PREVIEW_PASSWORD_SALT','fixture')
    monkeypatch.setattr(api, '_preview_password_valid', lambda _:True)
    app = api.create_app(preview_runtime())
    endpoint = next(r.endpoint for r in app.routes if r.path == '/preview/login' and 'POST' in r.methods)
    response = endpoint('fixture','fixture')
    assert '; secure' in response.headers['set-cookie'].lower()
    api._PREVIEW_SESSIONS.clear()


def test_missing_media_dependency_does_not_stop_api(monkeypatch):
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.__dict__.update(preview_runtime().__dict__)
    runtime.settings.api_host = '127.0.0.1'
    runtime.settings.api_port = 8080
    original_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == 'webrtc':
            raise ModuleNotFoundError('simulated missing media dependency')
        return original_import(name, *args, **kwargs)
    started = []
    class Server:
        def __init__(self, config): self.config=config
        def run(self): started.append(True)
    monkeypatch.setattr(builtins,'__import__',missing)
    monkeypatch.setattr('uvicorn.Server',Server)
    runtime._api_loop()
    assert started == [True]


def test_firestore_summary_does_not_require_realtime_database(monkeypatch):
    monkeypatch.setattr('firebase_admin._apps', {'fixture':object()})
    monkeypatch.setattr('pi4_attendance.cloud.firestore.client',lambda:object())
    def forbidden(*args, **kwargs):
        raise AssertionError('summary must not construct an unused RTDB client')
    monkeypatch.setattr('pi4_attendance.cloud.db.reference',forbidden)
    cloud = CloudSummary(SimpleNamespace(cloud_summary_enabled=True, pi_id='pi',
                                        firebase_service_account=Path('fixture.json')))
    assert cloud.db is not None


class Ref:
    def __init__(self, root, path=()): self.root,self.path=root,path
    def child(self,key): return Ref(self.root,self.path+(key,))
    def get(self): return self.root
    def delete(self):
        if len(self.path)==2:
            self.root.get(self.path[0],{}).pop(self.path[1],None)
    def update(self,payload): self.root[self.path[0]][self.path[1]].update(payload)
    def transaction(self, updater):
        result = updater(self.root.get(self.path[0], {}).get(self.path[1]))
        if result is None: self.delete()
        else: self.root.setdefault(self.path[0], {})[self.path[1]] = result
        return result


class Peer:
    connectionState = 'connected'
    def __init__(self): self.closed=False
    async def close(self): self.closed=True


@pytest.mark.parametrize('cause',['expired','deleted','revoked'])
def test_webrtc_closes_existing_peer_on_expiry_or_revocation(cause):
    peer=Peer()
    session={'answer':{'sdp':'fixture'}, 'state':'answered', 'created_at':time.time()*1000}
    if cause=='expired': session['created_at']=(time.time()-1000)*1000
    if cause=='revoked': session['state']='revoked'
    snapshot={'viewer':{'session':session}} if cause!='deleted' else {}
    manager=WebRTCManager.__new__(WebRTCManager)
    manager.root_ref=Ref(snapshot)
    manager.peer_connections={'viewer/session':peer}
    manager._peer_deadlines={'viewer/session':time.monotonic()+300}
    manager.runtime=SimpleNamespace(preview=SimpleNamespace(camera_ids=('camera-1',)), settings=SimpleNamespace(pi_id='pi'))
    asyncio.run(manager._poll_sessions())
    assert peer.closed
    assert not manager.peer_connections


def test_identity_hashes_the_actual_external_recognizer(tmp_path):
    pack=tmp_path/'models'/'pack'
    pack.mkdir(parents=True)
    detector=pack/'det.onnx'
    detector.write_bytes(b'detector')
    recognizer=tmp_path/'models'/'external.onnx'
    recognizer.write_bytes(b'first')
    first=model_identity(tmp_path,'pack',model_files=[detector,recognizer],
                         preprocessing={'mean':127.5}).compatibility_key
    recognizer.write_bytes(b'second')
    second=model_identity(tmp_path,'pack',model_files=[detector,recognizer],
                          preprocessing={'mean':127.5}).compatibility_key
    third=model_identity(tmp_path,'pack',model_files=[detector,recognizer],
                         preprocessing={'mean':0}).compatibility_key
    assert len({first,second,third})==3


def test_benchmark_uses_the_runtime_face_engine(monkeypatch,tmp_path):
    from pi4_attendance.tools import benchmark
    constructed=[]
    class Engine:
        model_version='fixture'
        model_info={'recognizer_source':'fixture'}
        def __init__(self,*args,**kwargs): constructed.append(kwargs)
        def start(self): pass
        def analyze(self,image):
            return [(SimpleNamespace(bbox=np.array([0,0,20,20]),det_score=0.9),np.array([1.,0.]))]
    monkeypatch.setattr('pi4_attendance.face_engine.FaceEngine',Engine)
    result=benchmark.run_pack(tmp_path,'pack',320,
        [benchmark.BenchmarkSample(np.zeros((32,32,3),np.uint8),'person')],0.45,
        recognizer_model='external.onnx')
    assert constructed[0]['recognizer_model']=='external.onnx'
    assert result['model_version']=='fixture'
    assert result['detections']==1


def test_preview_annotations_do_not_replace_raw_enrollment_frame():
    from pi4_attendance.preview import PreviewWindow
    preview=PreviewWindow(('camera-1',),30)
    raw=np.zeros((32,32,3),np.uint8)
    drawn=np.full_like(raw,255)
    preview.submit('camera-1',raw)
    _,original_sequence=preview.latest_sample('camera-1')
    preview.submit_overlay('camera-1',drawn)
    frame,sequence=preview.latest_sample('camera-1')
    np.testing.assert_array_equal(frame,raw)
    assert sequence==original_sequence
    np.testing.assert_array_equal(preview.latest_display_sample('camera-1')[0],drawn)
    preview.submit('camera-1',raw)
    np.testing.assert_array_equal(preview.latest_display_sample('camera-1')[0],raw)
