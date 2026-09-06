from pi4_attendance.api import _detected_image_type


def test_image_type_detection_uses_file_signature() -> None:
    assert _detected_image_type(b"\xff\xd8\xff\xe0payload") == "jpeg"
    assert _detected_image_type(b"\x89PNG\r\n\x1a\npayload") == "png"
    assert _detected_image_type(b"RIFF\x00\x00\x00\x00WEBPpayload") == "webp"
    assert _detected_image_type(b"<script>alert(1)</script>") is None
