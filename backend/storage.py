"""Transactional local projects. Main revisions are append-only until project deletion."""
from contextlib import contextmanager
from datetime import datetime, timezone
from difflib import SequenceMatcher
import copy
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4


AGENTS = ("splash", "splash-code", "splash-search")


def uid():
    return uuid4().hex


def now():
    return datetime.now(timezone.utc).isoformat()


def data_directory():
    from platformdirs import user_data_dir
    return Path(os.environ.get("SPLASH_DATA_DIR", user_data_dir("Splash", appauthor=False)))


class Conflict(ValueError):
    pass


def diff(before, after):
    # Tokenize without dropping whitespace: the review reconstructs both inputs exactly.
    import re
    a, b = (re.findall(r"\s+|\w+|[^\w\s]", value) for value in (before, after))
    return [{"kind": tag, "before": "".join(a[i:j]), "after": "".join(b[k:l])}
            for tag, i, j, k, l in SequenceMatcher(None, a, b, autojunk=max(len(a), len(b)) > 12000).get_opcodes()]


def revision(text, kind, sources=None, metadata=None):
    return {"id": uid(), "text": text, "kind": kind, "sources": sources or [],
            "metadata": metadata or {}, "created": now()}


class Store:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else data_directory()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "splash.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, body TEXT NOT NULL)")
            db.execute("PRAGMA user_version=1")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.execute("PRAGMA busy_timeout=15000")
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def edit(self, project_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM projects WHERE id=?", (project_id,)).fetchone()
            if not row:
                raise KeyError("Project not found")
            project = json.loads(row[0])
            yield project
            project["updated"] = now()
            db.execute("UPDATE projects SET body=? WHERE id=?", (json.dumps(project), project_id))

    def get(self, project_id):
        with self.connect() as db:
            row = db.execute("SELECT body FROM projects WHERE id=?", (project_id,)).fetchone()
        if not row:
            raise KeyError("Project not found")
        return json.loads(row[0])

    def list(self, query=""):
        with self.connect() as db:
            projects = [json.loads(row[0]) for row in db.execute("SELECT body FROM projects")]
        return sorted([{"id": p["id"], "title": p["title"], "agent": p["agent"],
                        "updated": p["updated"], "created": p["created"]}
                       for p in projects if query.casefold() in (p["title"] + " " + p["draft"]["text"]).casefold()],
                      key=lambda p: p["updated"], reverse=True)

    def create(self, agent, text, title="", source=None, revision_id=None):
        if agent not in AGENTS or not text.strip():
            raise ValueError("Choose an agent and enter an initial prompt.")
        imported = self.get(source) if source else None
        origin = None
        sources = []
        if imported:
            original = self.find_revision(imported, revision_id or imported["main_id"])
            text, sources = original["text"], copy.deepcopy(original["sources"])
            origin = {"project_id": source, "title": imported["title"], "revision_id": original["id"],
                      "agent": imported["agent"], "imported": now()}
        first = revision(text, "main", sources, {"origin": origin})
        draft = revision(text, "draft", sources, {"base_main": first["id"]})
        draft["version"] = 1
        project = {"id": uid(), "title": title.strip() or text.strip().splitlines()[0][:70],
                   "agent": agent, "created": now(), "updated": now(), "main_id": first["id"],
                   "draft": draft, "revisions": [first], "origin": origin,
                   "documents": copy.deepcopy(imported["documents"]) if imported else [],
                   "sources": copy.deepcopy(imported["sources"]) if imported else [],
                   "branches": [], "proposals": [], "reviews": [], "queries": [],
                   "settings": {"research": False, "scholar": False, "branch_count": 3,
                                "output": "prose" if agent == "splash" else "structured"}}
        with self.connect() as db:
            db.execute("INSERT INTO projects VALUES (?,?)", (project["id"], json.dumps(project)))
        return project

    @staticmethod
    def find_revision(p, revision_id):
        if p["draft"]["id"] == revision_id:
            return p["draft"]
        result = next((r for r in p["revisions"] if r["id"] == revision_id), None)
        if result is None:
            raise KeyError("Revision not found in this project")
        return result

    @staticmethod
    def check_version(p, expected):
        if p["draft"]["version"] != expected:
            raise Conflict("The draft changed. Reload the saved draft before retrying.")

    @staticmethod
    def replace_draft(p, text, sources=None, metadata=None):
        old = p["draft"]
        p["revisions"].append(copy.deepcopy(old))
        p["draft"] = revision(text, "draft", old["sources"] if sources is None else sources,
                              metadata or {"base_main": p["main_id"], "prior_draft":old["id"]})
        p["draft"]["version"] = old["version"] + 1
        # Proposals always belong to exactly one input revision.
        p["proposals"] = []

    def save_draft(self, pid, text, expected):
        with self.edit(pid) as p:
            self.check_version(p, expected)
            if text != p["draft"]["text"]:
                self.replace_draft(p, text)
        return p

    def restore(self, pid, revision_id, expected):
        with self.edit(pid) as p:
            self.check_version(p, expected)
            target = self.find_revision(p, revision_id)
            self.replace_draft(p, target["text"], copy.deepcopy(target["sources"]), {"restored_from": revision_id})
        return p

    def review(self, pid, expected):
        with self.edit(pid) as p:
            self.check_version(p, expected)
            main = self.find_revision(p, p["main_id"])
            if main["text"] == p["draft"]["text"]:
                raise ValueError("The draft has no changes to merge.")
            review = {"id": uid(), "main_id": p["main_id"], "draft_version": expected,
                      "text": p["draft"]["text"], "sources": copy.deepcopy(p["draft"]["sources"]),
                      "diff": diff(main["text"], p["draft"]["text"]), "created": now(), "result_id": None}
            p["reviews"].append(review)
        return review

    def merge(self, pid, review_id):
        with self.edit(pid) as p:
            review = next((r for r in p["reviews"] if r["id"] == review_id), None)
            if not review:
                raise KeyError("Merge review not found")
            if review["result_id"]:
                return p  # A retried confirmation has no additional side effects.
            if review.get("canceled") or review["main_id"] != p["main_id"] or review["draft_version"] != p["draft"]["version"]:
                raise Conflict("This merge review is stale. Review the current draft again.")
            new = revision(review["text"], "main", review["sources"],
                           {"prior_main": p["main_id"], "draft_id": p["draft"]["id"], "review_id": review_id})
            p["revisions"].append(new)
            p["main_id"] = new["id"]
            review["result_id"] = new["id"]
            self.replace_draft(p, new["text"], copy.deepcopy(new["sources"]))
        return p

    def delete(self, pid):
        with self.connect() as db:
            if not db.execute("DELETE FROM projects WHERE id=?", (pid,)).rowcount:
                raise KeyError("Project not found")

    def config(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT body FROM config WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_config(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO config VALUES (?,?)", (key, json.dumps(value)))
