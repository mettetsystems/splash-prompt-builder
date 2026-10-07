"""Splash's local HTTP and native WebSocket server."""

import json
import logging
import asyncio
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from backend.engine import MockEngine, utf16_length
from backend.storage import Store, Conflict
from backend.models import ModelService
from backend.mcp_host import ToolHost
from backend.agents import Agents
from backend.api import router, project_socket

MAX_PROMPT_LENGTH = 8000
MAX_MESSAGE_BYTES = 100_000
DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(application):
    state = application.state
    state.store = Store()
    os.environ.setdefault("HF_MODULES_CACHE", str(state.store.directory / "model-code-cache"))
    state.models = ModelService()
    state.tools = ToolHost(state.store.directory)
    state.agents = Agents(state.store, state.models, state.tools)
    state.jobs, state.job_results, state.downloads, state.sockets = {}, {}, {}, {}
    state.background = set()
    state.tools.start()
    configured = state.store.config("model")
    if configured:
        task = asyncio.create_task(state.models.load(configured))
        state.background.add(task)
        task.add_done_callback(state.background.discard)
    yield
    for task in list(state.jobs.values()) + list(state.background):
        task.cancel()
    await asyncio.gather(*state.jobs.values(), *state.background, return_exceptions=True)
    await state.tools.close()
    await state.models.close()


app = FastAPI(title="Splash", version="2.0.0", lifespan=lifespan)
app.include_router(router)
app.add_api_websocket_route("/ws/projects/{pid}", project_socket)


@app.middleware("http")
async def local_access(request: Request, call_next):
    host = request.url.hostname
    if host not in ("localhost", "127.0.0.1", "::1", "testserver"):
        return JSONResponse(status_code=403, content={"detail": "Splash serves local hosts only."})
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).hostname not in ("localhost", "127.0.0.1", "::1"):
        return JSONResponse(status_code=403, content={"detail": "Cross-site requests are not permitted."})
    try:
        if int(request.headers.get("content-length", "0")) > 13_000_000:
            return JSONResponse(status_code=413, content={"detail": "Request exceeds the upload limit."})
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "Invalid content length."})
    return await call_next(request)


@app.exception_handler(Conflict)
async def conflict_handler(request, exc):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def missing_handler(request, exc):
    return JSONResponse(status_code=404, content={"detail": str(exc).strip("'")})


@app.exception_handler(ValueError)
async def value_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    # Do not echo invalid Unicode or uploaded/prompt input into an error response.
    detail = "; ".join(error["msg"] for error in exc.errors())
    return JSONResponse(status_code=422, content={"detail": detail.encode("utf-8", errors="replace").decode()})


class PromptUpdate(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["prompt_update"]
    request_id: int = Field(ge=0, le=9_007_199_254_740_991)
    text: str = Field(max_length=MAX_PROMPT_LENGTH)
    # Browsers report selectionStart in UTF-16 code units, not Python characters.
    cursor: int = Field(ge=0)
    freeze_radius: int = Field(default=3, ge=0, le=12)
    commit: bool = False

    @model_validator(mode="after")
    def valid_browser_text(self):
        try:
            self.text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("Text must contain valid Unicode.") from exc
        if utf16_length(self.text) > MAX_PROMPT_LENGTH:
            raise ValueError("Prompt exceeds 8000 characters.")
        if self.cursor > utf16_length(self.text):
            raise ValueError("Cursor is outside the prompt.")
        return self


@app.get("/healthz")
async def health_check():
    runtime = getattr(app.state, "models", None)
    return {"status": "healthy", "engine": "diffusion", "version": "2.0.0",
            "model": runtime.status if runtime else {"state": "unloaded"}}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Retained solely for the explicitly labeled v1 mock protocol and regression harness.
    origin = websocket.headers.get("origin")
    if origin and urlsplit(origin).hostname not in ("localhost", "127.0.0.1", "::1"):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    # State belongs to this connection. Prompts are never broadcast or persisted.
    engine = MockEngine()
    try:
        while True:
            frame = await websocket.receive()
            if frame["type"] == "websocket.disconnect":
                break
            raw = frame.get("text")
            if raw is None:
                await websocket.send_json({"type": "error", "message": "Send a JSON text message."})
                continue
            if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
                await websocket.close(code=1009, reason="Message too large")
                break
            request_id = None
            try:
                data = json.loads(raw)
                if isinstance(data, dict) and type(data.get("request_id")) is int:
                    candidate = data["request_id"]
                    if 0 <= candidate <= 9_007_199_254_740_991:
                        request_id = candidate
                message = PromptUpdate.model_validate(data)
            except (json.JSONDecodeError, ValidationError, RecursionError):
                await websocket.send_json({
                    "type": "error", "request_id": request_id,
                    "message": "Invalid prompt update. Use text up to 8000 characters, a valid cursor, and a freeze radius from 0 to 12.",
                })
                continue
            started = perf_counter()
            try:
                result = engine.update(message.text, message.cursor, message.freeze_radius, message.commit)
            except Exception:
                logger.exception("Mock engine failed")
                await websocket.send_json({"type": "error", "request_id": message.request_id,
                                           "message": "Could not update the preview. Please edit your prompt to retry."})
                continue
            await websocket.send_json({
                "type": "diffusion_update", "request_id": message.request_id,
                "source": message.text, "engine": "mock",
                "latency_ms": round((perf_counter() - started) * 1000, 2), **result,
            })
    except WebSocketDisconnect:
        pass


@app.get("/")
async def index():
    if (DIST / "index.html").is_file():
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})
    return JSONResponse(status_code=503, content={
        "message": "Frontend not built. Run npm ci and npm run build in frontend, or use the Vite dev server.",
    })


# check_dir=False lets the API start before the first frontend build.
app.mount("/assets", StaticFiles(directory=DIST / "assets", check_dir=False), name="assets")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, ws_max_size=MAX_MESSAGE_BYTES)
