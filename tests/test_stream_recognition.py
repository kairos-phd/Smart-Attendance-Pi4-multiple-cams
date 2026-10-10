import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pi4_attendance.preview import PreviewWindow
from pi4_attendance.webrtc import PiVideoTrack


@pytest.mark.parametrize('camera_id',['icspring','jieli','sonix'])
def test_each_stream_continues_with_overlay_when_recognition_is_enabled(camera_id):
    preview=PreviewWindow(('icspring','jieli','sonix'),30)
    for index,cam in enumerate(preview.camera_ids):
        preview.submit(cam,np.full((32,32,3),index+1,np.uint8))
    # Streaming must not acquire the inference lock or run face inference.
    class BusyInference:
        def __enter__(self): raise AssertionError('video waited on inference')
        def __exit__(self,*args): pass
    runtime=SimpleNamespace(preview=preview,inference_lock=BusyInference())
    async def exercise():
        track=PiVideoTrack(runtime,camera_id)
        first=await track.recv()
        preview.set_recognition(True)
        raw,sequence=preview.latest_sample(camera_id)
        overlay=np.full_like(raw,255)
        preview.submit_overlay(camera_id,overlay)
        track._next_frame_at=time.monotonic()
        second=await track.recv()
        assert second.pts>first.pts
        assert np.all(second.to_ndarray(format='bgr24')==255)
        np.testing.assert_array_equal(preview.latest_sample(camera_id)[0],raw)
        assert preview.latest_sample(camera_id)[1]==sequence
        preview.set_recognition(False)
        preview.submit(camera_id,raw)
        track._next_frame_at=time.monotonic()
        third=await track.recv()
        assert third.pts>second.pts
        np.testing.assert_array_equal(third.to_ndarray(format='bgr24'),raw)
        track.stop()
    asyncio.run(exercise())
