"""Serve the BBCSport demo using only Python's standard library."""

from __future__ import annotations

import argparse
import errno
import json
import re
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
REPLAY_ASSETS = ["demo-data.js", "story-data.js", "training-animation.mp4", "training-results.png"]


class DemoHandler(SimpleHTTPRequestHandler):
    def _serve_video(self, head_only=False):
        """HTML video players need byte ranges for reliable scene seeking."""
        path = ROOT / "demo/training-animation.mp4"
        if not path.is_file():
            self.send_error(404, "Animation not found")
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            valid = match is not None and bool(match[1] or match[2])
            if valid:
                if match[1]:
                    start = int(match[1])
                    end = min(int(match[2]), end) if match[2] else end
                else:
                    count = int(match[2])
                    start = max(0, size - count)
                    valid = count > 0
                valid = valid and start <= end and start < size
            if not valid:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # A scene change can cancel the previous video range request.

    def do_HEAD(self):
        if urlparse(self.path).path == "/training-animation.mp4":
            self._serve_video(head_only=True)
        else:
            super().do_HEAD()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/training-animation.mp4":
            self._serve_video()
            return
        if parsed.path in ["/api/article", "/health"]:
            if parsed.path == "/health":
                payload = {"ready": all((ROOT / "demo" / name).is_file() for name in REPLAY_ASSETS), "offline": True}
            else:
                try:
                    index = int(parse_qs(parsed.query).get("id", [""])[0])
                    documents = (ROOT / "data/features/bbcsport.docs").read_text().splitlines()
                    if not 0 <= index < len(documents):
                        raise ValueError("Invalid article index")
                    category, number = documents[index].split(".")
                    path = ROOT / "data/raw/bbcsport" / category / f"{number}.txt"
                    text = path.read_text(encoding="utf-8", errors="replace")
                    lines = text.splitlines()
                    payload = {"available": True, "title": lines[0], "excerpt": " ".join(lines[1:])[:400]}
                except (FileNotFoundError, ValueError, IndexError):
                    payload = {"available": False}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, format, *args):
        # Keep the demo terminal quiet; errors remain visible.
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Open the BBCSport demo.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not all((ROOT / "demo" / name).is_file() for name in REPLAY_ASSETS):
        raise SystemExit("Replay assets are missing. Follow the Demo section in README.md and run: python render_presentation.py")
    handler = partial(DemoHandler, directory=str(ROOT / "demo"))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            raise SystemExit(f"Port {args.port} is unavailable. Try: python run_demo.py --port {args.port + 1}") from error
        raise SystemExit(f"Could not start the local demo: {error}") from error
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nBBCSport demo: {url}", flush=True)
    print("Runs offline using saved, real predictions. Press Ctrl+C to stop.\n", flush=True)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
