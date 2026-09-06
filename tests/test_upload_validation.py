from io import BytesIO
import struct
from types import SimpleNamespace
import zlib

import cv2
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers
import numpy as np
import pytest

from pi4_attendance import api


def endpoint():
    runtime = SimpleNamespace(
        settings=SimpleNamespace(api_cors_origins=[], api_auth_mode="shared_token", admin_shared_token="test"),
        enroll=lambda name, student_id, images, actor: {"shapes": [image.shape[:2] for image in images]},
    )
    return next(route.endpoint for route in api.create_app(runtime).routes if route.path == "/api/users" and "POST" in route.methods)


def upload(raw, mime="image/png", stream=None):
    return UploadFile(stream if stream is not None else BytesIO(raw), filename="image", headers=Headers({"content-type": mime}))


def png_header(width, height):
    chunk = b"IHDR" + struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + chunk + struct.pack(">I", zlib.crc32(chunk))


@pytest.mark.parametrize("extension,mime", [(".png", "image/png"), (".jpg", "image/jpeg"), (".webp", "image/webp")])
def test_valid_images_keep_working(extension, mime):
    ok, encoded = cv2.imencode(extension, np.full((24, 32, 3), 127, dtype=np.uint8))
    assert ok
    assert endpoint()("User", "S1", [upload(encoded.tobytes(), mime) for _ in range(3)], "admin") == {"shapes": [(24, 32)] * 3}


@pytest.mark.parametrize("raw,mime", [
    (png_header(100000, 100000), "image/png"),
    (b"\xff\xd8\xff\xc0\x00\x11\x08" + struct.pack(">HH", 50000, 50000) + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00", "image/jpeg"),
    (b"RIFF" + struct.pack("<I", 22) + b"WEBPVP8X" + struct.pack("<I", 10) + b"\x00" * 4 + (99999).to_bytes(3, "little") * 2, "image/webp"),
    (b"\x89PNG\r\n\x1a\ntruncated", "image/png"),
    (b"\xff\xd8\xff\xe0\x00\x01", "image/jpeg"),
    (b"RIFF\x00\x00\x00\x00WEBP", "image/webp"),
    (png_header(16, 16), "image/jpeg"),
    (png_header(0, 16), "image/png"),
])
def test_invalid_header_rejected_before_decode(raw, mime, monkeypatch):
    calls = []
    monkeypatch.setattr(api.cv2, "imdecode", lambda *args: calls.append(1))
    with pytest.raises(HTTPException) as error:
        endpoint()("User", "S1", [upload(raw, mime) for _ in range(3)], "admin")
    assert error.value.status_code == 400
    assert calls == []


def test_all_compressed_sizes_checked_before_read_or_decode(monkeypatch):
    class NoRead(BytesIO):
        def read(self, *args):
            raise AssertionError("Oversized batch was read")

    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 16)
    with pytest.raises(HTTPException):
        endpoint()("User", "S1", [upload(b"", stream=NoRead(b"x" * size)) for size in [8, 8, 17]], "admin")


def test_total_compressed_size_checked_before_read(monkeypatch):
    class NoRead(BytesIO):
        def read(self, *args):
            raise AssertionError("Oversized batch was read")

    monkeypatch.setattr(api, "MAX_ENROLLMENT_BYTES", 20, raising=False)
    with pytest.raises(HTTPException):
        endpoint()("User", "S1", [upload(b"", stream=NoRead(b"x" * 8)) for _ in range(3)], "admin")


def test_total_pixels_checked_before_any_decode(monkeypatch):
    calls = []
    monkeypatch.setattr(api.cv2, "imdecode", lambda *args: calls.append(1))
    with pytest.raises(HTTPException):
        endpoint()("User", "S1", [upload(png_header(2048, 2048)) for _ in range(5)], "admin")
    assert calls == []


@pytest.mark.parametrize("extension,options,mime", [
    (".jpg", [cv2.IMWRITE_JPEG_PROGRESSIVE, 1], "image/jpeg"),
    (".webp", [cv2.IMWRITE_WEBP_QUALITY, 101], "image/webp"),
])
def test_progressive_jpeg_and_lossless_webp(extension, options, mime):
    _, encoded = cv2.imencode(extension, np.full((24, 32, 3), 127, dtype=np.uint8), options)
    assert endpoint()("User", "S1", [upload(encoded.tobytes(), mime) for _ in range(3)], "admin")["shapes"] == [(24, 32)] * 3


@pytest.mark.parametrize("animated,mismatch", [(True, False), (False, True)])
def test_webp_canvas_must_match_static_bitstream(animated, mismatch, monkeypatch):
    _, encoded = cv2.imencode(".webp", np.zeros((24, 32, 3), dtype=np.uint8))
    canvas = bytes([2 if animated else 0]) + b"\x00" * 3
    canvas += (99999 if mismatch else 31).to_bytes(3, "little") + (23).to_bytes(3, "little")
    body = b"WEBPVP8X" + struct.pack("<I", 10) + canvas + encoded.tobytes()[12:]
    raw = b"RIFF" + struct.pack("<I", len(body)) + body
    calls = []
    monkeypatch.setattr(api.cv2, "imdecode", lambda *args: calls.append(1))
    with pytest.raises(HTTPException):
        endpoint()("User", "S1", [upload(raw, "image/webp") for _ in range(3)], "admin")
    assert calls == []


@pytest.mark.parametrize("raw,mime", [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF\x04\x00\x00\x00WEBP", "image/webp"),
])
def test_magic_alone_is_rejected_before_decode(raw, mime, monkeypatch):
    calls = []
    monkeypatch.setattr(api.cv2, "imdecode", lambda *args: calls.append(1))
    with pytest.raises(HTTPException):
        endpoint()("User", "S1", [upload(raw, mime) for _ in range(3)], "admin")
    assert calls == []


def test_decoded_dimensions_must_match_header(monkeypatch):
    monkeypatch.setattr(api.cv2, "imdecode", lambda *args: np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(HTTPException, match="declared dimensions"):
        endpoint()("User", "S1", [upload(png_header(24, 32)) for _ in range(3)], "admin")
