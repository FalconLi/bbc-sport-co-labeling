"""Regression checks for the byte-range requests used by browser seeking."""

from io import BytesIO

import pytest

import run_demo


@pytest.mark.parametrize("requested,status,expected,content_range", [
    (None, 200, b"0123456789", None),
    ("bytes=0-1", 206, b"01", "bytes 0-1/10"),
    ("bytes=5-", 206, b"56789", "bytes 5-9/10"),
    ("bytes=-3", 206, b"789", "bytes 7-9/10"),
    ("bytes=8-100", 206, b"89", "bytes 8-9/10"),
    ("bytes=12-", 416, b"", "bytes */10"),
    ("bytes=8-3", 416, b"", "bytes */10"),
    ("bytes=-0", 416, b"", "bytes */10"),
])
def test_video_range_response(monkeypatch, tmp_path, requested, status, expected, content_range):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo/training-animation.mp4").write_bytes(b"0123456789")
    monkeypatch.setattr(run_demo, "ROOT", tmp_path)
    handler = object.__new__(run_demo.DemoHandler)
    handler.headers = {} if requested is None else {"Range": requested}
    handler.wfile = BytesIO()
    response = {"headers": {}}
    handler.send_response = lambda code: response.update(status=code)
    handler.send_header = lambda key, value: response["headers"].update({key: value})
    handler.end_headers = lambda: None
    handler._serve_video()
    assert response["status"] == status
    assert handler.wfile.getvalue() == expected
    assert int(response["headers"]["Content-Length"]) == len(expected)
    assert response["headers"].get("Content-Range") == content_range
    if status != 416:
        assert response["headers"]["Accept-Ranges"] == "bytes"
