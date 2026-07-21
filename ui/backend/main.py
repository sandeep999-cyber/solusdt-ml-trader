import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on the path so all subpackages are importable
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ui.backend.routes import router  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Teacher Model Replay Server", version="0.1.0")

# Register API routes FIRST so they aren't shadowed
app.include_router(router)

# Serve built frontend assets and SPA fallback
_dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_dist_dir / "assets")), name="assets")
    logger.info("Serving assets from %s", _dist_dir / "assets")

    # Read index.html per request so a frontend rebuild (npm run build) is
    # picked up without restarting the server.
    @app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
    async def _serve_spa(full_path: str):  # noqa: ARG001
        return HTMLResponse((_dist_dir / "index.html").read_text(encoding="utf-8"))
else:
    logger.warning("UI dist not found at %s — run 'npm run build' in ui/frontend/", _dist_dir)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def main():
    uvicorn.run("ui.backend.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
