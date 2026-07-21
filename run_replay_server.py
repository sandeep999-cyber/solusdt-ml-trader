#!/usr/bin/env python3
"""
Entry point for the replay UI server.
Precomputes inference on startup, then serves the FastAPI app.

Usage: python run_replay_server.py
"""
import sys
from pathlib import Path

# Ensure project root is importable from anywhere
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ui.backend.state import get_state
import uvicorn
from ui.backend.main import app

if __name__ == "__main__":
    get_state()  # warm up cache
    uvicorn.run(app, host="127.0.0.1", port=8000)
