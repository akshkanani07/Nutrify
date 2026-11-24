"""Nutrify AI — Production FastAPI Server Runner."""

from __future__ import annotations

import os
import sys
import socket
from pathlib import Path

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir.parent))

def find_available_port(host: str, preferred_port: int, fallbacks: list[int] | None = None) -> int:
    if fallbacks is None:
        fallbacks = [8001, 8000]
    candidates = [preferred_port] + [p for p in fallbacks if p != preferred_port]
    for p in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, p))
                return p
        except OSError:
            continue
    return preferred_port

if __name__ == "__main__":
    os.environ.setdefault("ENABLE_ML_INTEGRATION", "true")
    os.environ.setdefault("AI_PROVIDER", "ml_model")

    from app import config
    config.AI_PROVIDER = os.getenv("AI_PROVIDER", "ml_model")

    import uvicorn
    from app.main import create_app

    server_app = create_app()

    preferred = int(os.getenv("PORT", "8001"))
    host = os.getenv("HOST", "127.0.0.1")
    port = find_available_port(host, preferred)
    if port != preferred:
        print(f"Notice: Port {preferred} is in use/lingering. Auto-switching to port {port}.")
    print(f"Starting Nutrify AI backend on http://{host}:{port} ...")
    uvicorn.run(
        server_app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
