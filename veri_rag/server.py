"""Server entrypoint alias (`veri_rag/server.py`) forwarding to `veri_rag/web_api.py`."""

from __future__ import annotations

import os

from veri_rag.web_api import (
    STUDIO,
    LocalVeriRAGServerHandler,
    VeriRAGStudioEngine,
    lambda_handler,
    route_api_request,
    run_local_server,
)

__all__ = [
    "STUDIO",
    "LocalVeriRAGServerHandler",
    "VeriRAGStudioEngine",
    "lambda_handler",
    "route_api_request",
    "run_local_server",
]

if __name__ == "__main__":
    run_local_server(port=int(os.environ.get("PORT", "8087")))
