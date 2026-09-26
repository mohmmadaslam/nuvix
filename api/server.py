"""Local demo server for NUVIX: a search UI over the same `search.query.search`
the CLI and the eval use. Standard library only (no web framework), so it adds
no dependencies.

    python -m api.server            # then open http://127.0.0.1:8000

Routes:
  GET /                    the single-page UI (api/static/index.html)
  GET /api/search          ?q=...&k=10&rerank=1&legs=lexical,fuzzy,semantic
  GET /api/recordings      the corpus: title, language, length, utterances, speakers
  GET /api/eval            the saved ablation report (eval/report.json)
  GET /audio/<recording>   the source audio, with HTTP Range support so the
                           browser can seek to a result's timestamp
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import psycopg

from ingest.common import DATABASE_URL, ROOT, get_logger
from search.query import ALL_LEGS, get_model, get_reranker, search

log = get_logger("api")

STATIC = ROOT / "api" / "static"
EVAL_REPORT = ROOT / "eval" / "report.json"
RECORDING_ID = re.compile(r"^[a-z0-9-]+$")

# The embedding model and reranker share one GPU (MPS) and are not thread-safe
# in practice; serialize searches. Fine for a single-user demo.
_SEARCH_LOCK = threading.Lock()


def _recordings() -> list[dict]:
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.id, r.title, r.language, r.duration_ms, r.audio_path,
                   (SELECT count(*) FROM utterances u WHERE u.recording_id = r.id),
                   (SELECT array_agg(s.display_name ORDER BY s.display_name)
                      FROM speakers s WHERE s.recording_id = r.id)
            FROM recordings r ORDER BY r.title
            """
        )
        return [
            {"id": i, "title": t, "language": lang, "duration_ms": d,
             "audio_path": ap, "utterances": n, "speakers": sp or []}
            for i, t, lang, d, ap, n, sp in cur.fetchall()
        ]


class Handler(BaseHTTPRequestHandler):
    server_version = "NUVIX"

    def log_message(self, fmt, *args):  # quieter than the default stderr access log
        log.info(fmt % args)

    # -- helpers ---------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False, default=float).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", {"Cache-Control": "no-store"})

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/search":
                self._search(parse_qs(url.query))
            elif url.path == "/api/recordings":
                self._json(_recordings())
            elif url.path == "/api/eval":
                if EVAL_REPORT.exists():
                    self._send(200, EVAL_REPORT.read_bytes(), "application/json; charset=utf-8")
                else:
                    self._json({"error": "no eval/report.json - run: python -m eval.run_eval --out eval/report.json"}, 404)
            elif url.path.startswith("/audio/"):
                self._audio(url.path.removeprefix("/audio/"))
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser aborted (e.g. seeking in the audio element)
        except Exception as e:  # keep the demo alive; surface the error to the UI
            log.exception("request failed")
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _search(self, qs: dict):
        q = (qs.get("q", [""])[0]).strip()
        if not q:
            return self._json({"query": q, "took_ms": 0, "results": []})
        k = max(1, min(int(qs.get("k", ["10"])[0]), 30))
        rerank = qs.get("rerank", ["1"])[0] != "0"
        legs = frozenset(qs.get("legs", [",".join(sorted(ALL_LEGS))])[0].split(",")) & ALL_LEGS
        if not legs:
            return self._json({"error": "select at least one retrieval method"}, 400)
        t0 = time.perf_counter()
        with _SEARCH_LOCK:
            results = search(q, top_k=k, rerank=rerank, legs=legs)
        self._json({
            "query": q, "took_ms": round((time.perf_counter() - t0) * 1000),
            "rerank": rerank, "legs": sorted(legs), "results": results,
        })

    def _audio(self, recording_id: str):
        if not RECORDING_ID.match(recording_id):
            return self._json({"error": "bad recording id"}, 400)
        rec = next((r for r in _recordings() if r["id"] == recording_id), None)
        path = ROOT / rec["audio_path"] if rec else None
        if not path or not path.is_file():
            return self._json({"error": "audio not found"}, 404)
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:  # suffix range: last N bytes
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                return self._send(416, b"", "audio/wav", {"Content-Range": f"bytes */{size}"})
            status = 206
        else:
            status = 200
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    log.info("Loading models (first start takes a few seconds)...")
    get_model()
    get_reranker()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info(f"NUVIX UI ready: http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")


if __name__ == "__main__":
    main()
