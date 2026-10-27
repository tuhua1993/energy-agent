"""零依赖启动器：Python 标准库 HTTP 服务，不需要 pip install 任何包。

    python run.py

打开 http://127.0.0.1:8848 即可使用。
配置环境变量后自动切换到真模型：
    set LLM_API_KEY=sk-xxxx
    set LLM_BASE_URL=https://api.deepseek.com/v1
    set LLM_MODEL=deepseek-chat
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from agent.agent import EnergyAgent  # noqa: E402
from agent import config as cfg  # noqa: E402

AGENT = EnergyAgent(str(BASE / "data" / "docs"))
LOCK = threading.Lock()

MIME = {".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body=b"", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            file = BASE / "web" / "index.html"
        elif path == "/api/status":
            self._send(200, "application/json",
                       json.dumps({"llm": bool(cfg.LLM_API_KEY),
                                   "model": cfg.LLM_MODEL if cfg.LLM_API_KEY else "mock"},
                                  ensure_ascii=False).encode("utf-8"))
            return
        else:
            file = BASE / path.lstrip("/")
        if not file.exists() or file.is_dir():
            self._send(404, "text/plain; charset=utf-8", b"404")
            return
        self._send(200, MIME.get(file.suffix, "application/octet-stream"),
                   file.read_bytes())

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = {}

        if path == "/api/reset":
            AGENT.reset()
            self._send(200, "application/json", b'{"ok":true}')
            return

        if path != "/api/chat":
            self._send(404, "text/plain; charset=utf-8", b"404")
            return

        msg = (body.get("message") or "").strip()
        if not msg:
            self._send(400, "application/json", b'{"error":"empty message"}')
            return

        # NDJSON 事件流：一行一个 JSON，前端逐行解析
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def write_event(obj):
            data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
            chunk = b"%X\r\n" % len(data) + data + b"\r\n"
            self.wfile.write(chunk)
            self.wfile.flush()

        try:
            with LOCK:
                for ev in AGENT.chat(msg):
                    write_event(ev)
        except Exception as e:
            write_event({"type": "error", "message": str(e)})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main():
    print("=" * 56)
    print("  能源行业运维 Agent  ·  RAG + Function Calling")
    print("=" * 56)
    print(f"  地址： http://{cfg.HOST}:{cfg.PORT}")
    print(f"  模型： {cfg.LLM_MODEL if cfg.LLM_API_KEY else 'Mock（未配置 API Key，全链路可演示）'}")
    print(f"  语料： {len(AGENT.retriever.chunks)} 个片段 / BM25"
          + ("+ 向量混合" if cfg.EMBED_MODEL else ""))
    print("  提示： 配置 LLM_API_KEY 环境变量后自动切换到真模型")
    print("=" * 56)
    ThreadingHTTPServer((cfg.HOST, cfg.PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
