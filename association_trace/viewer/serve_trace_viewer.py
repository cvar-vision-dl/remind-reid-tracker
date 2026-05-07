from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
DEFAULT_RUNS_DIR = HERE.parents[2] / "outputs" / "association_trace"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_run_dir(runs_dir: Path, run_id: str) -> Path | None:
    candidate = (runs_dir / run_id).resolve()
    try:
        candidate.relative_to(runs_dir.resolve())
    except ValueError:
        return None
    if not candidate.is_dir():
        return None
    return candidate


def list_runs(runs_dir: Path) -> list[dict]:
    if not runs_dir.exists():
        return []
    rows = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True):
        manifest_path = run_dir / "manifest.json"
        schema_path = run_dir / "pipeline_schema.json"
        if not manifest_path.exists() or not schema_path.exists():
            continue
        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue
        class_entries = list(manifest.get("class_entries", []) or [])
        rows.append(
            {
                "run_id": str(manifest.get("run_id", run_dir.name)),
                "created_at": str(manifest.get("created_at", "")),
                "schema_version": str(manifest.get("schema_version", "")),
                "trace_version": str(manifest.get("trace_version", "")),
                "frame_count": int(len(manifest.get("frame_ids", []) or [])),
                "class_entry_count": int(len(class_entries)),
                "path": str(run_dir.relative_to(runs_dir.parent)),
            }
        )
    return rows


class TraceViewerHandler(SimpleHTTPRequestHandler):
    runs_dir: Path = DEFAULT_RUNS_DIR

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": str(message)}, status=status)


    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, f"File not found: {path.name}")
            return
        body = path.read_bytes()
        mime = content_type or (mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api(parsed)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def handle_api(self, parsed) -> None:
        path = parsed.path
        query = parse_qs(parsed.query or "")

        if path == "/api/runs":
            self._send_json({"runs": list_runs(self.runs_dir)})
            return

        if path.startswith("/api/run/"):
            parts = [part for part in path.split("/") if part]
            if len(parts) < 4:
                self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown API route")
                return
            run_id = unquote(parts[2])
            action = parts[3]
            run_dir = safe_run_dir(self.runs_dir, run_id)
            if run_dir is None:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"Run not found: {run_id}")
                return

            if action == "manifest":
                self._send_json(load_json(run_dir / "manifest.json"))
                return
            if action == "schema":
                self._send_json(load_json(run_dir / "pipeline_schema.json"))
                return
            if action == "trace":
                frame_values = query.get("frame_id", [])
                class_values = query.get("class_id", [])
                if not frame_values or not class_values:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "frame_id and class_id are required")
                    return
                try:
                    frame_id = int(frame_values[0])
                    class_id = int(class_values[0])
                except ValueError:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "frame_id and class_id must be integers")
                    return
                trace_path = run_dir / "frames" / f"frame_{frame_id:06d}_class_{class_id:03d}.json"
                if not trace_path.exists():
                    self._send_error_json(HTTPStatus.NOT_FOUND, "Trace file not found")
                    return
                self._send_json(load_json(trace_path))
                return
            if action == "preview":
                frame_values = query.get("frame_id", [])
                if not frame_values:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "frame_id is required")
                    return
                try:
                    frame_id = int(frame_values[0])
                except ValueError:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "frame_id must be an integer")
                    return
                preview_path = run_dir / "frame_previews" / f"frame_{frame_id:06d}.png"
                if not preview_path.exists():
                    self._send_error_json(HTTPStatus.NOT_FOUND, "Preview file not found")
                    return
                self._send_file(preview_path, content_type="image/png")
                return
            if action == "memory":
                frame_values = query.get("frame_id", [])
                if not frame_values:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "frame_id is required")
                    return
                try:
                    frame_id = int(frame_values[0])
                except ValueError:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "frame_id must be an integer")
                    return
                memory_path = run_dir / "memory_snapshots" / f"frame_{frame_id:06d}.json"
                if not memory_path.exists():
                    self._send_error_json(HTTPStatus.NOT_FOUND, "Memory snapshot not found")
                    return
                self._send_json(load_json(memory_path))
                return

        self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown API route")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the association trace viewer")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Directory containing association_trace runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    TraceViewerHandler.runs_dir = runs_dir
    server = ThreadingHTTPServer((args.host, int(args.port)), TraceViewerHandler)
    print(f"Association trace viewer listening on http://{args.host}:{args.port}/")
    print(f"Runs dir: {runs_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
