import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import tkinter as tk
from PIL import ImageTk
import pytest

from pi4_attendance.face_engine import FaceEngine
from pi4_attendance.model_identity import model_identity
from pi4_attendance.preview import PreviewWindow
from pi4_attendance.webrtc import WebRTCManager


def test_pack_mode_cannot_reuse_ambiguous_legacy_identity(tmp_path, monkeypatch):
    pack = tmp_path / 'models' / 'pack'
    pack.mkdir(parents=True)
    detector = pack / 'det.onnx'; detector.write_bytes(b'detector')
    recognizer = pack / 'rec.onnx'; recognizer.write_bytes(b'recognizer')
    class App:
        def __init__(self, **kwargs):
            self.models = {'detection': SimpleNamespace(model_file=str(detector)),
                           'recognition': SimpleNamespace(model_file=str(recognizer), input_mean=127.5, input_std=127.5)}
        def prepare(self, **kwargs): pass
    monkeypatch.setitem(sys.modules, 'insightface.app', SimpleNamespace(FaceAnalysis=App))
    old_key = model_identity(tmp_path, 'pack').compatibility_key
    engine = FaceEngine(tmp_path, 'pack', 320, .45, .06, recognizer_model='pack')
    engine.start()
    assert engine.model_version != old_key
    assert set(engine.model_info['files']) == {str(detector), str(recognizer)}
    assert engine.model_info['recognizer_source'] == 'pack'


def test_hdmi_renders_overlay_while_enrollment_reads_raw(monkeypatch):
    preview = PreviewWindow(('camera-1',), 30)
    preview.submit('camera-1', np.zeros((4, 4, 3), np.uint8))
    preview.submit_overlay('camera-1', np.full((4, 4, 3), 255, np.uint8))
    stop = threading.Event()
    rendered = []
    class Root:
        def title(self, *a): pass
        def bind(self, *a): pass
        def protocol(self, *a): pass
        def update_idletasks(self): pass
        def update(self): stop.set()
        def destroy(self): pass
    class Label:
        def __init__(self, *a, **kw): pass
        def pack(self, **kw): pass
        def configure(self, **kw): pass
    monkeypatch.setattr(tk, 'Tk', Root)
    monkeypatch.setattr(tk, 'Label', Label)
    monkeypatch.setattr(ImageTk, 'PhotoImage', lambda image: rendered.append(np.array(image)))
    preview.run(stop)
    assert np.all(rendered[0] == 255)
    assert np.all(preview.latest_sample()[0] == 0)


class Ref:
    def __init__(self, root, path=()): self.root, self.path = root, path
    def child(self, key): return Ref(self.root, self.path + (key,))
    def get(self):
        if not self.path: return self.root
        return self.root.get(self.path[0], {}).get(self.path[1])
    def transaction(self, updater):
        result = updater(self.get())
        if result is None: self.root.get(self.path[0], {}).pop(self.path[1], None)
        else: self.root.setdefault(self.path[0], {})[self.path[1]] = result
        return result
    def update(self, payload): self.root.setdefault(self.path[0], {}).setdefault(self.path[1], {}).update(payload)
    def delete(self): self.root.get(self.path[0], {}).pop(self.path[1], None)


def manager(snapshot):
    obj = WebRTCManager.__new__(WebRTCManager)
    obj.root_ref = Ref(snapshot)
    obj.peer_connections = {}; obj._peer_deadlines = {}
    obj.runtime = SimpleNamespace(settings=SimpleNamespace(pi_id='pi'),
        preview=SimpleNamespace(camera_ids=('camera-1',)))
    obj.stop_event = threading.Event()
    return obj


@pytest.mark.parametrize('revoke', ['revoked', 'deleted'])
def test_revocation_during_answer_is_never_overwritten(monkeypatch, revoke):
    offer = {'type':'offer', 'sdp':'fixture'}
    snapshot = {'viewer':{'session':{'state':'offer', 'offer':offer,
                'camera_id':'camera-1', 'created_at':time.time()*1000}}}
    obj = manager(snapshot)
    class Peer:
        iceGatheringState = 'complete'
        localDescription = SimpleNamespace(type='answer', sdp='answer')
        connectionState = 'connected'
        def __init__(self, *a): self.closed = False
        def on(self, *a): return lambda callback: callback
        async def setRemoteDescription(self, *a): pass
        def addTrack(self, *a): pass
        async def createAnswer(self): return self.localDescription
        async def setLocalDescription(self, *a):
            if revoke == 'deleted': snapshot['viewer'].pop('session')
            else: snapshot['viewer']['session']['state'] = 'revoked'
        async def close(self): self.closed = True
    peers = []
    def create(*a):
        peer = Peer(); peers.append(peer); return peer
    monkeypatch.setattr('pi4_attendance.webrtc.RTCPeerConnection', create)
    asyncio.run(obj._handle_offer('viewer', 'session', 'camera-1', offer))
    assert peers[0].closed
    assert not obj.peer_connections
    session = snapshot['viewer'].get('session')
    assert session is None or session['state'] == 'revoked'


def test_expiry_closes_locally_when_firebase_is_unavailable():
    obj = manager({})
    class Peer:
        closed = False
        async def close(self): self.closed = True
    peer = Peer()
    obj.peer_connections['viewer/session'] = peer
    obj._peer_deadlines['viewer/session'] = time.monotonic() - 1
    obj.root_ref.get = lambda: (_ for _ in ()).throw(OSError('offline'))
    async def check():
        with pytest.raises(OSError): await obj._poll_sessions()
    asyncio.run(check())
    assert peer.closed


def test_expiry_watchdog_runs_while_signaling_is_blocked():
    obj = manager({})
    class Peer:
        closed = False
        async def close(self): self.closed = True
    peer = Peer()
    obj.peer_connections['viewer/session'] = peer
    obj._peer_deadlines['viewer/session'] = time.monotonic() - 1
    async def stalled(): await asyncio.sleep(30)
    obj._publish_status = stalled
    async def check():
        task = asyncio.create_task(obj._run())
        try:
            await asyncio.sleep(.05)
            assert peer.closed
        finally:
            task.cancel()
            try: await task
            except asyncio.CancelledError: pass
    asyncio.run(check())


def test_local_expiry_uses_remaining_lifetime_of_claimed_offer(monkeypatch):
    offer = {'type':'offer','sdp':'fixture'}
    snapshot = {'viewer':{'session':{'state':'offer','offer':offer,'camera_id':'camera-1',
                'created_at':(time.time()-290)*1000}}}
    obj = manager(snapshot)
    class Peer:
        iceGatheringState = 'complete'
        localDescription = SimpleNamespace(type='answer',sdp='answer')
        def __init__(self,*a): pass
        def on(self,*a): return lambda f:f
        async def setRemoteDescription(self,*a): pass
        def addTrack(self,*a): pass
        async def createAnswer(self): return self.localDescription
        async def setLocalDescription(self,*a): pass
        async def close(self): pass
    monkeypatch.setattr('pi4_attendance.webrtc.RTCPeerConnection',Peer)
    asyncio.run(obj._handle_offer('viewer','session','camera-1',offer))
    remaining = obj._peer_deadlines['viewer/session'] - time.monotonic()
    assert 8 < remaining <= 10


@pytest.mark.parametrize('created', [float('nan'), float('inf'), 'invalid'])
def test_malformed_session_timestamp_is_expired(created):
    assert WebRTCManager._session_expired({'created_at':created})
