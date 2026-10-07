"""Version 2 project API. Every main-prompt mutation passes through merge review."""
import asyncio
import json
import copy
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.storage import Conflict, diff, now, uid
from backend.models import CATALOG, hardware
from backend.agents import apply_proposals
from backend.research import parse_document

router = APIRouter(prefix="/api")
MAX_TEXT = 80_000


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="after")
    def valid_unicode(self):
        for value in self.model_dump().values():
            if isinstance(value, str):
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError("Use valid Unicode text.") from exc
        return self


class Create(Input):
    agent: Literal["splash", "splash-code", "splash-search"]
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    title: str = Field(default="", max_length=180)
    source: str | None = None
    revision_id: str | None = None


class Draft(Input):
    text: str = Field(max_length=MAX_TEXT)
    expected_version: int = Field(ge=1)


class Version(Input):
    expected_version: int = Field(ge=1)


class Restore(Version):
    revision_id: str


class Selection(Version):
    ids: list[str] = Field(max_length=500)


class Refine(Version):
    focus: int | None = Field(default=None, ge=0)


class Settings(Input):
    title: str | None = Field(default=None, max_length=180)
    research: bool | None = None
    scholar: bool | None = None
    branch_count: int | None = Field(default=None, ge=1, le=5)
    output: Literal["prose", "structured"] | None = None


class LoadModel(Input):
    model: Literal["compact", "dream", "llada"]
    path: str = ""
    device: str = Field(default="cpu", pattern=r"^(cpu|mps|cuda:[0-9]+)$")
    steps: int = Field(default=64, ge=16, le=512)
    context: int | None = Field(default=None, ge=512, le=32768)
    quantize: bool = False
    offload: bool = False
    gpu_memory: str = Field(default="6GiB", pattern=r"^[1-9][0-9]?GiB$")
    cpu_memory: str = Field(default="24GiB", pattern=r"^[1-9][0-9]{0,2}GiB$")

    @model_validator(mode="after")
    def compatible_options(self):
        if self.quantize and self.offload:
            raise ValueError("Choose NF4 or CPU offload, not both.")
        if (self.quantize or self.offload) and not self.device.startswith("cuda:"):
            raise ValueError("NF4 and GPU/CPU offload require a CUDA device in this release.")
        return self


class Download(Input):
    model: Literal["compact", "dream", "llada"]


def services(request):
    return request.app.state


async def publish(state, pid, event, **data):
    for socket in list(state.sockets.get(pid, set())):
        try:
            await socket.send_json({"protocol": 2, "project_id": pid, "type": event, **data})
        except Exception:
            state.sockets[pid].discard(socket)


def public_project(p):
    # The review tokens are needed only by the initiating review dialog.
    return {k:v for k,v in p.items() if k != "reviews"}


async def changed(state, p):
    await publish(state, p["id"], "project_changed", draft_version=p["draft"]["version"], main_id=p["main_id"], agent=p["agent"])
    return public_project(p)


@router.get("/projects")
async def projects(request: Request, q: str = ""):
    return services(request).store.list(q)


@router.post("/projects", status_code=201)
async def create_project(data: Create, request: Request):
    return public_project(services(request).store.create(data.agent, data.text, data.title, data.source, data.revision_id))


@router.get("/projects/{pid}")
async def get_project(pid: str, request: Request):
    return public_project(services(request).store.get(pid))


@router.patch("/projects/{pid}")
async def configure_project(pid: str, data: Settings, request: Request):
    state = services(request)
    with state.store.edit(pid) as p:
        values = data.model_dump(exclude_none=True)
        if "title" in values:
            title = values.pop("title").strip()
            if not title:
                raise ValueError("Project name cannot be empty.")
            p["title"] = title
        p["settings"].update(values)
    return await changed(state, p)


@router.delete("/projects/{pid}")
async def delete_project(pid: str, request: Request):
    state = services(request)
    for (project_id, _), task in list(state.jobs.items()):
        if project_id == pid:
            task.cancel()
    state.store.delete(pid)
    await publish(state, pid, "project_deleted")
    return {"deleted": pid}


@router.put("/projects/{pid}/draft")
async def save_draft(pid: str, data: Draft, request: Request):
    state = services(request)
    for kind in ("refine", "research", "branches", "synthesize"):
        task = state.jobs.get((pid, kind))
        if task:
            task.cancel()
    return await changed(state, state.store.save_draft(pid, data.text, data.expected_version))


@router.post("/projects/{pid}/restore")
async def restore_draft(pid: str, data: Restore, request: Request):
    state = services(request)
    return await changed(state, state.store.restore(pid, data.revision_id, data.expected_version))


@router.post("/projects/{pid}/merge-review")
async def merge_review(pid: str, data: Version, request: Request):
    return services(request).store.review(pid, data.expected_version)


@router.post("/projects/{pid}/merge/{review_id}")
async def confirm_merge(pid: str, review_id: str, request: Request):
    state = services(request)
    return await changed(state, state.store.merge(pid, review_id))


@router.delete("/projects/{pid}/merge/{review_id}")
async def cancel_merge(pid: str, review_id: str, request: Request):
    with services(request).store.edit(pid) as p:
        for review in p["reviews"]:
            if review["id"] == review_id:
                review["canceled"] = True
    return {"canceled": True}


@router.get("/projects/{pid}/compare/{revision_id}")
async def compare_revision(pid: str, revision_id: str, request: Request):
    store = services(request).store
    p = store.get(pid)
    return {"diff": diff(store.find_revision(p, revision_id)["text"], store.find_revision(p, p["main_id"])["text"])}


@router.get("/projects/{pid}/export")
async def export_project(pid: str, request: Request, format: str = "json"):
    store = services(request).store
    p = store.get(pid)
    if format == "json":
        body, media, suffix = json.dumps({"format": "splash-project", "version": 2, "project": public_project(p)}, ensure_ascii=False, indent=2), "application/json", "json"
    elif format in ("txt", "md"):
        body, media, suffix = store.find_revision(p, p["main_id"])["text"], "text/plain; charset=utf-8", format
    else:
        raise ValueError("Choose json, txt, or md.")
    return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="splash-{pid[:8]}.{suffix}"'})


@router.post("/projects/import-file", status_code=201)
async def import_file(request: Request, agent: str, file: UploadFile = File(...)):
    body = await file.read(12_000_001)
    if len(body) > 12_000_000:
        raise ValueError("Project export exceeds 12 MB.")
    try:
        data = json.loads(body)
        if data.get("format") != "splash-project" or data.get("version") != 2:
            raise ValueError("Invalid format")
        source = data["project"]
        main = next(r for r in source["revisions"] if r["id"] == source["main_id"] and r["kind"] == "main")
        validated = Create(agent=agent, text=main["text"], title=source["title"][:169] + " (imported)")
        if not isinstance(source["id"], str) or not isinstance(main.get("sources", []), list) or not all(isinstance(s, str) for s in main.get("sources", [])):
            raise ValueError("Invalid source identity")
    except (ValueError, KeyError, TypeError, AttributeError, StopIteration, RecursionError):
        raise ValueError("Choose a valid Splash version 2 export containing a main prompt and valid Unicode.") from None
    store = services(request).store
    p = store.create(validated.agent, validated.text, validated.title)
    try:
        with store.edit(p["id"]) as p:
            if len(source.get("documents", [])) > 30:
                raise ValueError("Too many context documents.")
            p["documents"] = []
            document_ids = {}
            for d in source.get("documents", []):
                if any(type(page["page"]) is not int or page["page"] < 1 or not isinstance(page["text"], str) for page in d["pages"]):
                    raise ValueError("Invalid page")
                pages = [{"page": int(page["page"]), "text": str(page["text"])} for page in d["pages"]]
                for page in pages:
                    page["text"].encode("utf-8")
                if len(pages) > 200 or sum(len(page["text"]) for page in pages) > 500_000:
                    raise ValueError("Imported context exceeds the project limits.")
                document_ids[d["id"]] = uid()
                p["documents"].append({"id": document_ids[d["id"]], "name": str(d["name"])[:180], "pages": pages,
                                        "created": now(), "origin": {"document_id": d["id"], "project_id": source["id"]}})
            p["sources"] = []
            for s in source.get("sources", []):
                item = {k:s[k] for k in ("id", "title", "url", "text", "provider", "level", "retrieved")}
                for value in item.values():
                    if not isinstance(value, str):
                        raise ValueError("Invalid source")
                    value.encode("utf-8")
                item.update({k:s[k] for k in ("page", "doi", "year") if k in s})
                if s.get("document_id") in document_ids:
                    item["document_id"] = document_ids[s["document_id"]]
                item["origin"] = {"project_id":source["id"], "source_id":s["id"]}
                p["sources"].append(item)
            if not set(main.get("sources", [])).issubset({s["id"] for s in p["sources"]}):
                raise ValueError("Missing source references")
            p["origin"] = {"project_id": source["id"], "revision_id": main["id"], "title": source["title"], "agent":source.get("agent", "unknown"), "imported": now()}
            p["draft"]["sources"] = main.get("sources", [])
            p["revisions"][0]["sources"] = main.get("sources", [])
            p["revisions"][0]["metadata"]["origin"] = copy.deepcopy(p["origin"])
    except Exception:
        store.delete(p["id"])
        raise ValueError("The export contains invalid context or source records.")
    return public_project(p)


@router.post("/projects/{pid}/documents")
async def upload_document(pid: str, request: Request, file: UploadFile = File(...)):
    data = await file.read(8_000_001)
    if len(data) > 8_000_000:
        raise ValueError("Uploads are limited to 8 MB each.")
    document = await asyncio.to_thread(parse_document, file.filename or "context.txt", data)
    state = services(request)
    with state.store.edit(pid) as p:
        if len(p["documents"]) >= 30:
            raise ValueError("A project can contain up to 30 context documents.")
        p["documents"].append(document)
    return await changed(state, p)


@router.delete("/projects/{pid}/documents/{document_id}")
async def remove_document(pid: str, document_id: str, request: Request):
    state = services(request)
    with state.store.edit(pid) as p:
        p["documents"] = [d for d in p["documents"] if d["id"] != document_id]
    return await changed(state, p)


@router.post("/projects/{pid}/proposals/accept")
async def accept_proposals(pid: str, data: Selection, request: Request):
    state = services(request)
    with state.store.edit(pid) as p:
        state.store.check_version(p, data.expected_version)
        old_proposals = copy.deepcopy(p["proposals"])
        text, sources = apply_proposals(p, data.ids)
        if len(text) > MAX_TEXT:
            raise ValueError("Accepted draft would exceed 80,000 characters.")
        accepted = [c for c in old_proposals if c["id"] in data.ids]
        state.store.replace_draft(p, text, sources, {"base_main":p["main_id"], "prior_draft":p["draft"]["id"], "accepted_edits":accepted})
        for change in old_proposals:
            if change["id"] in data.ids:
                continue
            shift = sum(len(c["after"]) - (c["end"]-c["start"]) for c in accepted if c["end"] <= change["start"])
            change["start"] += shift
            change["end"] += shift
            change["version"] = p["draft"]["version"]
            if text[change["start"]:change["end"]] == change["before"]:
                p["proposals"].append(change)
    return await changed(state, p)


@router.post("/projects/{pid}/proposals/reject")
async def reject_proposals(pid: str, data: Selection, request: Request):
    state = services(request)
    with state.store.edit(pid) as p:
        state.store.check_version(p, data.expected_version)
        p["proposals"] = [s for s in p["proposals"] if s["id"] not in data.ids]
    return await changed(state, p)


@router.post("/projects/{pid}/branches/select")
async def select_branches(pid: str, data: Selection, request: Request):
    state = services(request)
    with state.store.edit(pid) as p:
        state.store.check_version(p, data.expected_version)
        if len(set(data.ids)) > 5 or not set(data.ids).issubset({c["id"] for c in p["branches"]}):
            raise ValueError("Select up to five current branches.")
        for card in p["branches"]:
            card["selected"] = card["id"] in data.ids
    return await changed(state, p)


async def launch(state, pid, kind, operation):
    current = state.jobs.get((pid, kind))
    if current and not current.done():
        operation.close()
        return {"status": "already_running"}
    async def run():
        await publish(state, pid, "progress", job=kind, status="running")
        try:
            async with asyncio.timeout(120 if kind == "research" else 600):
                result = await operation
            state.job_results[(pid, kind)] = {"status": "complete", "result": result, "finished": now()}
            await publish(state, pid, "job_complete", job=kind, result=result)
        except asyncio.CancelledError:
            state.job_results[(pid, kind)] = {"status": "canceled"}
            await publish(state, pid, "job_canceled", job=kind)
            raise
        except TimeoutError:
            message = "This task reached its time limit. The main prompt is unchanged; try a smaller scope or a faster profile."
            state.job_results[(pid, kind)] = {"status": "error", "error": message}
            await publish(state, pid, "job_error", job=kind, message=message)
        except Exception as exc:
            state.job_results[(pid, kind)] = {"status": "error", "error": str(exc)}
            await publish(state, pid, "job_error", job=kind, message=str(exc))
        finally:
            state.jobs.pop((pid, kind), None)
    state.jobs[(pid, kind)] = asyncio.create_task(run())
    return {"status": "started"}


@router.post("/projects/{pid}/refine")
async def refine(pid: str, data: Refine, request: Request):
    state = services(request)
    state.store.check_version(state.store.get(pid), data.expected_version)
    return await launch(state, pid, "refine", state.agents.enrich(pid, data.expected_version, data.focus))


@router.post("/projects/{pid}/branches/generate")
async def generate_branches(pid: str, data: Version, request: Request):
    state = services(request)
    state.store.check_version(state.store.get(pid), data.expected_version)
    return await launch(state, pid, "branches", state.agents.branches(pid, data.expected_version))


@router.post("/projects/{pid}/branches/synthesize")
async def synthesize(pid: str, data: Selection, request: Request):
    state = services(request)
    state.store.check_version(state.store.get(pid), data.expected_version)
    return await launch(state, pid, "synthesize", state.agents.synthesize(pid, data.expected_version, data.ids))


@router.post("/projects/{pid}/research")
async def research(pid: str, data: Version, request: Request, force: bool = False):
    state = services(request)
    state.store.check_version(state.store.get(pid), data.expected_version)
    return await launch(state, pid, "research", state.agents.research(pid, data.expected_version, force))


@router.get("/projects/{pid}/jobs")
async def jobs(pid: str, request: Request):
    state = services(request)
    state.store.get(pid)
    result = {kind:value for (project, kind), value in state.job_results.items() if project == pid}
    result.update({kind: {"status": "running"} for project, kind in state.jobs if project == pid})
    return result


@router.get("/hardware")
async def inspect_hardware(request: Request):
    return await asyncio.to_thread(hardware)


@router.get("/models")
async def models(request: Request):
    state = services(request)
    catalog = [{"id": key, **info, "downloaded": (state.store.directory / "models" / key / "config.json").is_file()} for key, info in CATALOG.items()]
    return {"catalog": catalog, "runtime": state.models.status, "downloads": state.downloads,
            "mcp": {"state": state.tools.state, "error": state.tools.error}, "data_directory": str(state.store.directory)}


@router.post("/models/download")
async def download_model(data: Download, request: Request):
    state = services(request)
    if state.downloads.get(data.model, {}).get("state") == "downloading":
        return state.downloads[data.model]
    info = CATALOG[data.model]
    path = state.store.directory / "models" / data.model
    async def download():
        state.downloads[data.model] = {"state": "downloading", "path": str(path), "revision": info["revision"]}
        try:
            from huggingface_hub import snapshot_download
            await asyncio.to_thread(snapshot_download, info["repo"], revision=info["revision"], local_dir=path)
            state.downloads[data.model]["state"] = "complete"
        except Exception as exc:
            state.downloads[data.model].update(state="error", error=str(exc))
    task = asyncio.create_task(download())
    state.background.add(task)
    task.add_done_callback(state.background.discard)
    return {"state": "downloading"}


@router.post("/models/load")
async def load_model(data: LoadModel, request: Request):
    state = services(request)
    if state.models.status["state"] == "loading":
        raise Conflict("A model is already loading.")
    config = data.model_dump(exclude_none=True)
    config["path"] = str(Path(data.path).expanduser().resolve()) if data.path else str(state.store.directory / "models" / data.model)
    if not (Path(config["path"]) / "config.json").is_file():
        raise ValueError("Download the model or supply an existing checkpoint directory first.")
    config["context"] = min(config.get("context", CATALOG[data.model]["context"]), CATALOG[data.model]["context"])
    state.store.set_config("model", config)
    task = asyncio.create_task(state.models.load(config))
    state.background.add(task)
    task.add_done_callback(state.background.discard)
    return {"state": "loading"}


async def project_socket(socket: WebSocket, pid: str):
    from urllib.parse import urlsplit
    origin = socket.headers.get("origin")
    if origin and urlsplit(origin).hostname not in ("localhost", "127.0.0.1", "::1"):
        await socket.close(code=1008)
        return
    state = socket.app.state
    try:
        project = state.store.get(pid)
    except KeyError:
        await socket.close(code=1008)
        return
    await socket.accept()
    state.sockets.setdefault(pid, set()).add(socket)
    await socket.send_json({"protocol": 2, "type": "connected", "project_id": pid,
                            "agent": project["agent"], "draft_version": project["draft"]["version"]})
    try:
        while True:
            message = await socket.receive_text()
            if message == "ping":
                await socket.send_json({"protocol": 2, "type": "pong", "project_id": pid})
    except WebSocketDisconnect:
        pass
    finally:
        state.sockets[pid].discard(socket)
