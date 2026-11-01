"""Serve the offline classroom demo using only Python's standard library."""

from __future__ import annotations

import argparse
import errno
import json
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent


class DemoHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ["/api/article", "/health"]:
            if parsed.path == "/health":
                payload = {"ready": (ROOT / "demo/demo-data.js").is_file(), "offline": True}
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
        # Keep the presentation terminal quiet; errors remain visible.
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Open the offline BBCSport presentation demo.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not (ROOT / "demo/demo-data.js").is_file():
        raise SystemExit("No recording found. Run the experiment with --export-demo first.")
    handler = partial(DemoHandler, directory=str(ROOT / "demo"))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            raise SystemExit(f"Port {args.port} is unavailable. Try: python run_demo.py --port {args.port + 1}") from error
        raise SystemExit(f"Could not start the local demo: {error}") from error
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nBBCSport presentation demo: {url}", flush=True)
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
