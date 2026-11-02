"""Version 2 project API. Every main-prompt mutation passes through merge review."""
import asyncio
import json
import copy
import re
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.storage import Conflict, diff, now, uid
from backend.models import CATALOG, hardware
from backend.ollama import OLLAMA_CATALOG, EMBEDDING_MODEL, OllamaClient, select_ollama_model, gpu_supports_large_models
from backend.endpoints import discover_endpoints, probe_endpoint, endpoint_config, normalize_url
from backend.agents import apply_proposals, library_defaults, operator_note
from backend.research import parse_document
from backend.checkpoints import dependency_status, source_spec, inspect_local, inspect_remote, fetch_checkpoint, downloaded_source
from backend.comments import anchor, thread, get_thread, apply_suggestion, review_document, answer_comment, add_threads, consistency_comments
from backend.state import input_key, claim, finish, interrupted, reject_edits, accept_edits, comment_edit, overlaps, mark_accepted_threads

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


class MergeReview(Version):
    proposal_ids: list[str] = Field(default_factory=list, max_length=10000)


class CommentCreate(Version):
    text: str = Field(min_length=1, max_length=4000)
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class CommentReply(Input):
    text: str = Field(min_length=1, max_length=4000)


class CommentStatus(Input):
    status: Literal["open", "resolved", "dismissed"]


class Refine(Version):
    focus: int | None = Field(default=None, ge=0)


class Settings(Input):
    title: str | None = Field(default=None, max_length=180)
    research: bool | None = None
    scholar: bool | None = None
    branch_count: int | None = Field(default=None, ge=1, le=5)
    output: Literal["prose", "structured"] | None = None


class LocalIndex(Input):
    directory: str = Field(default="", max_length=4096)
    recursive: bool = False
    confirm_nested: bool = False


class OllamaDownload(Input):
    model: str = Field(max_length=100)


class OllamaLoad(OllamaDownload):
    automatic: bool = False
    context: int = Field(default=4096, ge=1024, le=32768)
    confirm_separate_model: bool = False


class LoadModel(Input):
    model: Literal["compact", "dream", "llada"]
    path: str = Field(default="", max_length=4096)
    source: Literal["catalog", "huggingface", "local"] = "catalog"
    repo: str = Field(default="", max_length=300)
    revision: str = Field(default="", max_length=200)
    device: str = Field(default="cpu", pattern=r"^(cpu|mps|cuda:[0-9]+)$")
    steps: int = Field(default=64, ge=16, le=512)
    quality_mode: Literal['quality', 'speed'] = 'quality'
    context: int | None = Field(default=None, ge=512, le=32768)
    quantize: bool = False
    offload: bool = False
    gpu_memory: str = Field(default="6GiB", pattern=r"^[1-9][0-9]?GiB$")
    cpu_memory: str = Field(default="24GiB", pattern=r"^[1-9][0-9]{0,2}GiB$")
    confirm_separate_model: bool = False

    @model_validator(mode="after")
    def compatible_options(self):
        if self.quantize and self.offload:
            raise ValueError("Choose NF4 or CPU offload, not both.")
        if (self.quantize or self.offload) and not self.device.startswith("cuda:"):
            raise ValueError("NF4 and GPU/CPU offload require a CUDA device in this release.")
        return self


class Download(Input):
    model: Literal["compact", "dream", "llada"]
    repo: str = Field(default="", max_length=300)
    revision: str = Field(default="", max_length=200)


class EndpointScan(Input):
    base_url: str = Field(default='', max_length=2048)


class EndpointSelection(Input):
    base_url: str = Field(max_length=2048)
    model: str = Field(min_length=1, max_length=500)
    context: int = Field(default=4096, ge=1024, le=32768)


class InspectCheckpoint(Download):
    path: str = Field(default="", max_length=4096)


class SystemPrompt(Input):
    id: str = Field(default="", max_length=32)
    name: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=2000)
    enabled: bool = True

    @model_validator(mode="after")
    def ready(self):
        self.name, self.text = self.name.strip(), self.text.strip()
        if not self.name or not self.text:
            raise ValueError("Give each system prompt a name and instruction.")
        if len(self.name) > 80 or len(self.text) > 2000:
            raise ValueError("A system prompt is limited to 80 characters in the name and 2,000 in the instruction.")
        if self.id and not re.fullmatch(r"[0-9a-f]{32}", self.id):
            raise ValueError("Use the saved prompt id.")
        return self


class SystemPromptList(Input):
    prompts: list[SystemPrompt] = Field(max_length=12)


def services(request):
    return request.app.state


class LibraryPrompt(Input):
    name: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=2000)
    agent: Literal["splash", "splash-code", "splash-search", "any"] = "any"
    note: str = Field(default="", max_length=240)

    @model_validator(mode="after")
    def ready(self):
        self.name, self.text, self.note = self.name.strip(), self.text.strip(), self.note.strip()
        if not self.name or not self.text:
            raise ValueError("Give the library prompt a name and instruction.")
        if len(self.name) > 80 or len(self.text) > 2000 or len(self.note) > 240:
            raise ValueError("A library prompt is limited to 80 characters in the name, 2,000 in the instruction, and 240 in the note.")
        return self


class LibraryFile(Input):
    format: Literal["splash-system-prompt"]
    version: Literal[1]
    prompt: LibraryPrompt


LIBRARY_LIMIT = 40


def saved_library(store):
    saved = store.config("prompt_repo", [])
    return [item for item in saved if isinstance(item, dict)] if isinstance(saved, list) else []


def remember_library_prompt(store, item):
    saved = saved_library(store)
    if len(saved) >= LIBRARY_LIMIT:
        raise ValueError("The prompt library holds 40 saved prompts. Remove one before adding another.")
    record = {"id": uid(), "name": item.name, "text": item.text, "agent": item.agent, "note": item.note, "builtin": False}
    store.set_config("prompt_repo", saved + [record])
    return record


def stored_system_prompts(items):
    prompts, seen = [], set()
    for item in items:
        prompt_id = item.id or uid()
        if prompt_id in seen:
            raise ValueError("Each system prompt needs its own id.")
        seen.add(prompt_id)
        prompts.append({"id": prompt_id, "name": item.name, "text": item.text, "enabled": item.enabled})
    return prompts


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
    from backend.local_index import remove_index
    await asyncio.to_thread(remove_index, state.store.directory, pid)
    state.store.delete(pid)
    await publish(state, pid, "project_deleted")
    return {"deleted": pid}


@router.put("/projects/{pid}/draft")
async def save_draft(pid: str, data: Draft, request: Request):
    state = services(request)
    for (project_id, kind), task in list(state.jobs.items()):
        if project_id == pid:
            task.cancel()
    return await changed(state, state.store.save_draft(pid, data.text, data.expected_version))


@router.post("/projects/{pid}/restore")
async def restore_draft(pid: str, data: Restore, request: Request):
    state = services(request)
    return await changed(state, state.store.restore(pid, data.revision_id, data.expected_version))


@router.post("/projects/{pid}/merge-review")
async def merge_review(pid: str, data: MergeReview, request: Request):
    return services(request).store.review(pid, data.expected_version, data.proposal_ids)


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
            p["system_prompts"] = stored_system_prompts(SystemPromptList(prompts=source.get("system_prompts", [])).prompts)
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
    from backend.local_index import remove_index
    await asyncio.to_thread(remove_index, state.store.directory, pid)
    return await changed(state, p)


@router.get("/projects/{pid}/local-index")
async def local_index_status(pid: str, request: Request):
    state = services(request)
    state.store.get(pid)
    from backend.local_index import index_status
    return await asyncio.to_thread(index_status, state.store.directory, pid)


@router.post("/projects/{pid}/local-index")
async def create_local_index(pid: str, data: LocalIndex, request: Request):
    state = services(request)
    project = state.store.get(pid)
    from backend.local_index import directory_documents, index_documents
    if data.recursive and not data.confirm_nested:
        raise ValueError("Confirm indexing supported files from nested folders.")
    documents = list(project["documents"])
    if data.directory:
        imported = await asyncio.to_thread(directory_documents, data.directory, data.recursive)
        existing_names = {d["name"] for d in documents}
        documents.extend(d for d in imported if d["name"] not in existing_names)
    if len(documents) > 30:
        raise ValueError("A project can contain up to 30 context documents.")
    if not documents:
        raise ValueError("Import context files or choose a directory before creating an index.")
    status = await state.models.inspect()
    if not status.get("available"):
        raise ValueError("Start Ollama before creating a local vector index.")
    installed = status.get("installed_models", [])
    if EMBEDDING_MODEL not in installed and EMBEDDING_MODEL+":latest" not in installed:
        raise ValueError("Download EmbeddingGemma before creating a local vector index.")
    try:
        result = await index_documents(state.store.directory, pid, documents, state.models)
    except ImportError as exc:
        raise ValueError("Install backend/requirements.txt and restart Splash to enable local vector indexing.") from exc
    with state.store.edit(pid) as current:
        if current["documents"] != project["documents"]:
            raise Conflict("Context documents changed during indexing. Rebuild the index.")
        current["documents"] = documents
    await changed(state, current)
    return {"ready": True, "chunks": result.get("indexed", 0), "documents": len(documents)}


@router.delete("/projects/{pid}/local-index")
async def delete_local_index(pid: str, request: Request):
    state = services(request)
    state.store.get(pid)
    from backend.local_index import remove_index
    removed = await asyncio.to_thread(remove_index, state.store.directory, pid)
    return {"ready": False, "removed": removed}


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
        mark_accepted_threads(p, accepted)
        state.store.replace_draft(p, text, sources, {"base_main":p["main_id"], "prior_draft":p["draft"]["id"], "accepted_edits":accepted})
        accept_edits(p, accepted)
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
        rejected = [s for s in p['proposals'] if s['id'] in data.ids]
        if len(rejected) != len(set(data.ids)):
            raise Conflict('Select current proposals to dismiss.')
        reject_edits(p, rejected)
    return await changed(state, p)


@router.post("/projects/{pid}/comments")
async def create_comment(pid: str, data: CommentCreate, request: Request):
    state = services(request)
    if not data.text.strip() or data.start == data.end:
        raise ValueError("Select a passage and write a comment.")
    with state.store.edit(pid) as p:
        state.store.check_version(p, data.expected_version)
        p["comments"].append(thread("note", data.text.strip(), [anchor(p['draft']['text'], data.start, data.end, data.expected_version)], author="you", origin="user"))
    return await changed(state, p)


@router.post("/projects/{pid}/comments/{comment_id}/reply")
async def reply_comment(pid: str, comment_id: str, data: CommentReply, request: Request):
    if not data.text.strip():
        raise ValueError("Write a reply first.")
    state = services(request)
    with state.store.edit(pid) as p:
        comment = get_thread(p, comment_id)
        comment['messages'].append({"id":uid(), "author":"you", "text":data.text.strip(), "created":now()})
        comment['updated'] = now()
    return await changed(state, p)


@router.patch("/projects/{pid}/comments/{comment_id}")
async def resolve_comment(pid: str, comment_id: str, data: CommentStatus, request: Request):
    state = services(request)
    with state.store.edit(pid) as p:
        comment = get_thread(p, comment_id)
        if comment['status'] == 'accepted':
            raise ValueError("Applied edits stay in discussion history. Start a new comment to revisit the wording.")
        if comment['kind']=='suggestion' and not comment['outdated']:
            edit = comment_edit(comment)
            if data.status in ('resolved','dismissed'):
                reject_edits(p, [edit], data.status, comment_id)
            elif data.status=='open':
                for d in p['edit_decisions']:
                    if d['status']!='accepted' and (d.get('comment_id')==comment_id or overlaps(d, edit)):
                        d['active']=False
        comment['status'], comment['updated'] = data.status, now()
    return await changed(state, p)


@router.post("/projects/{pid}/comments/{comment_id}/apply")
async def accept_comment(pid: str, comment_id: str, data: Version, request: Request):
    state = services(request)
    with state.store.edit(pid) as p:
        state.store.check_version(p, data.expected_version)
        comment = get_thread(p, comment_id)
        text = apply_suggestion(p, comment)
        edit = comment_edit(comment)
        comment['status'], comment['updated'] = 'accepted', now()
        state.store.replace_draft(p, text, list(dict.fromkeys(p['draft']['sources'] + comment['sources'])),
            {'comment_id':comment_id, 'prior_draft':p['draft']['id'], 'accepted_edits':[copy.deepcopy(comment)]})
        accept_edits(p, [edit])
    return await changed(state, p)


@router.post("/projects/{pid}/comments/review")
async def review_comments(pid: str, data: Version, request: Request, force: bool = False):
    state = services(request)
    p = state.store.get(pid)
    state.store.check_version(p, data.expected_version)
    return await launch(state, pid, 'comments', review_document(state.agents, pid, data.expected_version), expected=data.expected_version, force=force)


@router.post("/projects/{pid}/comments/{comment_id}/ask")
async def ask_comment(pid: str, comment_id: str, data: Version, request: Request):
    state = services(request)
    p = state.store.get(pid)
    state.store.check_version(p, data.expected_version)
    comment = get_thread(p, comment_id)
    if comment['status'] != 'open' or comment['outdated']:
        raise ValueError("Reopen a current discussion before asking the reviewer.")
    if state.models.status['state'] != 'ready':
        raise ValueError("Load a model to ask the reviewer. Your comment is saved locally.")
    return await launch(state, pid, 'comment_reply', answer_comment(state.agents, pid, comment_id, comment['messages'][-1]['id']))


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


async def launch(state, pid, kind, operation, *, expected=None, force=False, focus=None):
    current = state.jobs.get((pid, kind))
    if current and not current.done():
        operation.close()
        return {"status": "already_running"}
    run_id = None
    if expected is not None:
        try:
            with state.store.edit(pid) as p:
                state.store.check_version(p, expected)
                run_id, response = claim(p, kind, input_key(p, kind, state.models.status, focus), force)
            if response:
                operation.close()
                return response
        except Exception:
            operation.close()
            raise
    def checkpoint(status, result=None, error=None):
        if run_id:
            try:
                with state.store.edit(pid) as p:
                    finish(p, kind, run_id, status, result, error)
            except KeyError:
                pass  # The user deleted the project while this job was running.
    async def run():
        await publish(state, pid, "progress", job=kind, status="running", draft_version=expected)
        try:
            async with asyncio.timeout(120 if kind == "research" else 600):
                result = await operation
            checkpoint('complete', result)
            state.job_results[(pid, kind)] = {"status": "complete", "result": result, "finished": now(), 'version':expected}
            await publish(state, pid, "job_complete", job=kind, result=result, draft_version=expected)
        except asyncio.CancelledError:
            checkpoint('canceled')
            state.job_results[(pid, kind)] = {"status": "canceled", 'version':expected}
            await publish(state, pid, "job_canceled", job=kind, draft_version=expected)
            raise
        except (InterruptedError, Conflict):
            checkpoint('canceled')
            state.job_results[(pid, kind)] = {"status": "canceled", 'version':expected}
            await publish(state, pid, "job_canceled", job=kind, draft_version=expected)
        except TimeoutError:
            message = "This task reached its time limit. The main prompt is unchanged; try a smaller scope or a faster profile."
            checkpoint('error', error=message)
            state.job_results[(pid, kind)] = {"status": "error", "error": message, 'version':expected}
            await publish(state, pid, "job_error", job=kind, message=message, draft_version=expected)
        except Exception as exc:
            checkpoint('error', error=str(exc))
            state.job_results[(pid, kind)] = {"status": "error", "error": str(exc), 'version':expected}
            await publish(state, pid, "job_error", job=kind, message=str(exc), draft_version=expected)
        finally:
            operation.close()
            if state.jobs.get((pid, kind)) is asyncio.current_task():
                state.jobs.pop((pid, kind), None)
    state.jobs[(pid, kind)] = asyncio.create_task(run())
    def canceled_before_start(task):
        if task.cancelled():
            checkpoint('canceled')
            operation.close()
        if state.jobs.get((pid, kind)) is task and task.done():
            state.jobs.pop((pid, kind), None)
    state.jobs[(pid, kind)].add_done_callback(canceled_before_start)
    return {"status": "started"}


@router.post("/projects/{pid}/refine")
async def refine(pid: str, data: Refine, request: Request, force: bool = False):
    state = services(request)
    state.store.check_version(state.store.get(pid), data.expected_version)
    return await launch(state, pid, "refine", state.agents.enrich(pid, data.expected_version, data.focus), expected=data.expected_version, force=force, focus=data.focus)


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
    return await launch(state, pid, "research", state.agents.research(pid, data.expected_version, force), expected=data.expected_version, force=force)


@router.get("/projects/{pid}/jobs")
async def jobs(pid: str, request: Request):
    state = services(request)
    p = state.store.get(pid)
    result = {}
    for kind, runs in p['processing']['runs'].items():
        if runs:
            r = runs[-1]
            result[kind] = {'status':'canceled' if interrupted(r) else r['status'], 'version':r['version'], 'result':r.get('result',{}), 'error':r.get('error'), 'finished':r.get('finished')}
    result.update({kind:value for (project, kind), value in state.job_results.items() if project == pid})
    result.update({kind: {"status": "running"} for project, kind in state.jobs if project == pid})
    return result


@router.get("/projects/{pid}/system-prompts")
async def system_prompts(pid: str, request: Request):
    project = services(request).store.get(pid)
    return {"prompts": project.get("system_prompts", [])}


@router.put("/projects/{pid}/system-prompts")
async def save_system_prompts(pid: str, data: SystemPromptList, request: Request):
    state = services(request)
    with state.store.edit(pid) as project:
        prompts = stored_system_prompts(data.prompts)
        instructions_changed = operator_note(project.get('system_prompts')) != operator_note(prompts)
        project["system_prompts"] = prompts
        if instructions_changed:
            project['system_prompt_revision'] = project.get('system_prompt_revision', 0) + 1
            project['processing']['settled'] = None
            project['proposals'] = []
            project['enrichment_review'] = None
            project['comment_review'] = None
            for comment in project['comments']:
                if comment['origin'] in ('enrichment', 'review') and comment['status'] == 'open':
                    comment.update(outdated=True, outdated_reason='Project system prompts changed. Review again with the current instructions.')
    if instructions_changed:
        state.agents.last_research.pop(pid, None)
        for (project_id, _), task in list(state.jobs.items()):
            if project_id == pid:
                task.cancel()
    return await changed(state, project)


@router.get("/prompt-repo")
async def prompt_repo(request: Request):
    return {"prompts": library_defaults() + saved_library(services(request).store)}


@router.post("/prompt-repo")
async def save_prompt_repo(data: LibraryPrompt, request: Request):
    return {"prompt": remember_library_prompt(services(request).store, data)}


@router.post("/prompt-repo/import")
async def import_prompt_repo(data: LibraryFile, request: Request):
    return {"prompt": remember_library_prompt(services(request).store, data.prompt)}


@router.delete("/prompt-repo/{prompt_id}")
async def delete_prompt_repo(prompt_id: str, request: Request):
    if any(prompt["id"] == prompt_id for prompt in library_defaults()):
        raise ValueError("Built-in prompts stay in the library.")
    store = services(request).store
    saved = saved_library(store)
    kept = [item for item in saved if item.get("id") != prompt_id]
    if len(kept) == len(saved):
        raise KeyError("Library prompt not found")
    store.set_config("prompt_repo", kept)
    return {"prompts": library_defaults() + kept}


@router.get("/hardware")
async def inspect_hardware(request: Request):
    return await asyncio.to_thread(hardware)


@router.get("/models/ollama")
async def ollama_models(request: Request):
    state = services(request)
    status = await state.models.inspect()
    device = await asyncio.to_thread(hardware)
    try:
        recommended = select_ollama_model(device["gpus"], device["available_ram_gb"])
        recommendation_error = None
    except ValueError as exc:
        recommended, recommendation_error = None, str(exc)
    installed = set(status.get("installed_models", []))
    catalog = [{"id": key, **info, "downloaded": key in installed or key+":latest" in installed,
                "compatible": (device["available_ram_gb"] >= info["ram_gb"] if key == "qwen3.5:4b" else
                    any(gpu_supports_large_models(g.get("name", "")) and g.get("free_mb", 0) >= info["ram_gb"] * 1.2 * 1024
                        for g in device["gpus"]) and device["available_ram_gb"] >= info["ram_gb"]),
                "recommended": key == recommended} for key, info in OLLAMA_CATALOG.items()]
    return {"catalog": catalog, "runtime": state.models.status, "downloads": state.downloads,
            "recommended_model": recommended, "recommendation_error": recommendation_error,
            "ollama": {**status, "embedding_model": EMBEDDING_MODEL},
            "mcp": {"state": state.tools.state, "error": state.tools.error}, "data_directory": str(state.store.directory)}


@router.post("/models/ollama/download")
async def download_ollama(data: OllamaDownload, request: Request):
    state = services(request)
    if data.model not in (*OLLAMA_CATALOG.keys(), EMBEDDING_MODEL):
        raise ValueError("Choose a model from the supported local catalog.")
    if state.downloads.get(data.model, {}).get("state") == "downloading":
        return state.downloads[data.model]
    state.downloads[data.model] = {"state": "downloading", "model": data.model}
    async def download():
        try:
            await state.models.request("POST", "/api/pull", json={"name": data.model, "stream": False})
            state.downloads[data.model]["state"] = "complete"
        except Exception as exc:
            state.downloads[data.model].update(state="error", error=str(exc))
    task = asyncio.create_task(download())
    state.background.add(task)
    task.add_done_callback(state.background.discard)
    return {"state": "downloading"}


@router.post("/models/ollama/load")
async def load_ollama(data: OllamaLoad, request: Request):
    state = services(request)
    if state.models.status['state'] == 'loading':
        raise Conflict('A model is already loading.')
    if not data.confirm_separate_model:
        raise Conflict('Confirm starting an Ollama model, or reuse an already loaded endpoint instead.')
    detected = await asyncio.to_thread(hardware)
    model = select_ollama_model(detected['gpus'], detected['available_ram_gb'], None if data.automatic else data.model)
    info = OLLAMA_CATALOG[model]
    gpu_fits = any(gpu_supports_large_models(g.get('name', '')) and g.get('free_mb', 0) >= info['ram_gb'] * 1.2 * 1024 for g in detected['gpus'])
    if detected['available_ram_gb'] < info['ram_gb'] or (model != 'qwen3.5:4b' and not gpu_fits):
        raise ValueError('This model does not fit the currently available memory with reserve. Reuse an existing endpoint or select a smaller model.')
    status = await state.models.inspect()
    if model not in status.get('installed_models', []):
        raise ValueError('Download this Ollama model before loading it.')
    config = {'source': 'ollama', 'base_url': OllamaClient().base_url + '/v1',
              'model': model, 'context': data.context, 'cpu_only': not gpu_fits}
    # Persist the choice before switching, so a failed load cannot restore old weights on restart.
    state.store.set_config('model', config)
    state.models.status = {'state': 'loading', 'model': model, 'engine': 'ollama', 'config': config}
    async def load():
        await state.models.load(config, explicit_ollama_load=True)
    task = asyncio.create_task(load())
    state.background.add(task)
    task.add_done_callback(state.background.discard)
    return {'state': 'loading'}


@router.get("/models")
async def models(request: Request):
    state = services(request)
    catalog = [{"id": key, **info, "downloaded": (state.store.directory / "models" / key / "config.json").is_file()} for key, info in CATALOG.items()]
    return {"catalog": catalog, "runtime": state.models.status, "downloads": state.downloads,
            "endpoints": getattr(state, 'endpoints', []),
            "mcp": {"state": state.tools.state, "error": state.tools.error}, "data_directory": str(state.store.directory),
            "dependencies": dependency_status()}


@router.post('/models/endpoints/scan')
async def scan_endpoints(data: EndpointScan, request: Request):
    urls = [normalize_url(data.base_url)] if data.base_url.strip() else None
    found = await asyncio.to_thread(discover_endpoints, urls)
    services(request).endpoints = found
    return {'endpoints': found}


@router.post('/models/endpoint')
async def select_endpoint(data: EndpointSelection, request: Request):
    state = services(request)
    if state.models.status['state'] == 'loading':
        raise Conflict('A model connection is already being prepared.')
    base = normalize_url(data.base_url)
    found = await asyncio.to_thread(probe_endpoint, base)
    if not found:
        raise ValueError('No compatible model server responded at that address.')
    config = endpoint_config(found, data.model, data.context)
    # The explicit choice survives a connection failure during the switch.
    # Otherwise a restart could silently restore the old local checkpoint.
    state.store.set_config('model', config)
    state.models.status = {'state': 'loading', 'model': data.model, 'engine': 'endpoint', 'config': config}
    # Finish switching under ModelService's lock and release only Splash's own
    # weights. Never ask the external server to load/unload or stop a model.
    result = await state.models.load(config)
    if result['state'] != 'ready':
        raise ValueError(result.get('error') or 'Could not connect to model endpoint.')
    state.store.set_config('model', state.models.status['config'])
    return result


@router.post("/models/inspect")
async def inspect_model(data: InspectCheckpoint, request: Request):
    if data.path.strip():
        return {"source": "local", **await asyncio.to_thread(inspect_local, data.path, data.model)}
    spec = source_spec(data.model, data.repo, data.revision)
    try:
        return await asyncio.to_thread(inspect_remote, services(request).store.directory, spec)
    except Exception as exc:
        raise ValueError(f"Could not inspect the Hugging Face checkpoint: {exc}") from exc


@router.post("/models/download")
async def download_model(data: Download, request: Request):
    state = services(request)
    spec = source_spec(data.model, data.repo, data.revision)
    key = spec["id"]
    if state.downloads.get(key, {}).get("state") == "downloading":
        return state.downloads[key]
    # Reserve before starting the task so duplicate HTTP requests share one job.
    state.downloads[key] = {**spec, "state": "downloading", "stage": "Checking repository"}
    async def download():
        try:
            record = await asyncio.to_thread(inspect_remote, state.store.directory, spec)
            state.downloads[key].update(record, stage="Downloading checkpoint files")
            checked = await asyncio.to_thread(fetch_checkpoint, record)
            state.downloads[key].update(checked, state="complete", stage="Ready to load")
            state.store.set_config("model_downloads", {k:v for k,v in state.downloads.items() if v.get("state") == "complete"})
        except Exception as exc:
            state.downloads[key].update(state="error", error=str(exc))
    task = asyncio.create_task(download())
    state.background.add(task)
    task.add_done_callback(state.background.discard)
    return state.downloads[key]


@router.post("/models/load")
async def load_model(data: LoadModel, request: Request):
    state = services(request)
    if state.models.status["state"] == "loading":
        raise Conflict("A model is already loading.")
    state.endpoints = await asyncio.to_thread(discover_endpoints)
    if (state.endpoints or state.models.status.get('config', {}).get('source') in ('endpoint', 'ollama')) and not data.confirm_separate_model:
        raise Conflict('A local model server is running. Use its endpoint, or confirm loading a separate model in Model & hardware.')
    dependencies = dependency_status()
    missing = dependencies["missing"] + (["bitsandbytes"] if data.quantize and not dependencies["packages"]["bitsandbytes"] else [])
    if missing:
        raise ValueError(f"Missing model dependencies in {dependencies['python']}: {', '.join(missing)}. Run python scripts/setup.py --install-models{' --quantize' if data.quantize else ''} in the Splash project folder, then restart Splash.")
    config = data.model_dump(exclude_none=True, exclude={'confirm_separate_model'})
    if data.source == "huggingface":
        if not data.repo.strip():
            raise ValueError("Enter the Hugging Face repository to load.")
        record = downloaded_source(state.downloads, source_spec(data.model, data.repo, data.revision))
        config.update(path=record["path"], repo=record["repo"], revision=record["revision"])
    elif data.path.strip():
        config.update(path=str(Path(data.path).expanduser().resolve()), source="local", repo="", revision="")
    elif data.source == "local":
        raise ValueError("Enter the local checkpoint directory.")
    else:
        config.update(path=str(state.store.directory / "models" / data.model),
                      repo=CATALOG[data.model]["repo"], revision=CATALOG[data.model]["revision"])
        if state.downloads.get(data.model, {}).get("state") == "downloading":
            raise Conflict("Wait for this checkpoint download to finish before loading.")
    await asyncio.to_thread(inspect_local, config["path"], data.model)
    config["context"] = min(config.get("context", CATALOG[data.model]["context"]), CATALOG[data.model]["context"])
    state.models.status = {"state": "loading", "model": data.model, "engine": "diffusion", "config": config}
    async def load():
        result = await state.models.load(config)
        if result["state"] == "ready":
            state.store.set_config("model", config)
    task = asyncio.create_task(load())
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
