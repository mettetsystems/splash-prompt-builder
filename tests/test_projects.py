import asyncio
import copy
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch

from backend.storage import Store, Conflict, diff
from backend.agents import Agents, proposals, apply_proposals, missing_constraints, rank_sources, retain_explicit_requirements
from backend.mcp_host import ToolHost
from backend.research import Providers, parse_document, retrieve_context, public_get


class Projects(unittest.TestCase):
    def test_synthesis_keeps_explicit_consent_and_negative_constraints(self):
        original = "Study bilingual reading. Preserve informed consent. Do not infer causality."
        output = retain_explicit_requirements(original, "Compare bilingual reading outcomes across languages.")
        self.assertIn("Preserve informed consent.", output)
        self.assertIn("Do not infer causality.", output)
        self.assertEqual(retain_explicit_requirements(original, output), output)

    def test_source_cards_prioritize_supplied_docs_and_deduplicate_reposts(self):
        sources = [
            {"title":"Python CLI", "url":"https://example.org/one", "level":"abstract"},
            {"title":"Python CLI (repost)", "url":"https://example.org/two", "level":"abstract"},
            {"title":"argparse documentation", "url":"https://docs.python.org/3/library/argparse.html", "level":"full_text_excerpt"},
        ]
        ranked = rank_sources(sources, [sources[-1]["url"]])
        self.assertEqual(ranked[0]["title"], "argparse documentation")
        self.assertEqual(len(ranked), 2)

    def test_protected_constraints_survive_multilingual_rewrites(self):
        self.assertEqual(missing_constraints('Inclure des tests. Keep `user_id`, "UTF-8", and 2015.', 'Include tests. Keep `user_id`, "UTF-8", and 2015.'), [])
        missing = missing_constraints('Inclure des tests. Keep `user_id` and 2015.', 'Build the tool.')
        self.assertIn('`user_id`', missing)
        self.assertIn('2015', missing)
        self.assertIn('the explicit requirement to include tests', missing)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.p = self.store.create("splash-search", "Research bilingual education. Preserve consent.")

    def save(self, text):
        p = self.store.get(self.p["id"])
        return self.store.save_draft(p["id"], text, p["draft"]["version"])

    def test_original_immutable_and_reload(self):
        original = copy.deepcopy(self.p["revisions"][0])
        p = self.save("Explore a different angle")
        reopened = Store(self.temp.name).get(p["id"])
        self.assertEqual(reopened["revisions"][0], original)
        self.assertEqual(reopened["draft"]["text"], "Explore a different angle")
        self.assertEqual(reopened["main_id"], original["id"])
        self.assertTrue(any(r["kind"] == "draft" for r in reopened["revisions"]))

    def test_two_stage_merge_idempotent_restore(self):
        p = self.save("New direction")
        review = self.store.review(p["id"], p["draft"]["version"])
        self.assertEqual(self.store.get(p["id"])["main_id"], self.p["main_id"])
        p = self.store.merge(p["id"], review["id"])
        count = len(p["revisions"])
        self.assertEqual(self.store.merge(p["id"], review["id"])["main_id"], p["main_id"])
        self.assertEqual(len(self.store.get(p["id"])["revisions"]), count)
        restored = self.store.restore(p["id"], self.p["main_id"], p["draft"]["version"])
        self.assertEqual(restored["main_id"], p["main_id"])
        self.assertEqual(restored["draft"]["text"], self.p["draft"]["text"])
        final = self.store.merge(p["id"], self.store.review(p["id"], restored["draft"]["version"])["id"])
        self.assertEqual(sum(r["kind"] == "main" for r in final["revisions"]), 3)

    def test_stale_or_canceled_review_rejected(self):
        p = self.save("First proposal")
        review = self.store.review(p["id"], p["draft"]["version"])
        self.save("Changed after review")
        with self.assertRaises(Conflict): self.store.merge(p["id"], review["id"])
        p = self.store.get(p["id"])
        review = self.store.review(p["id"], p["draft"]["version"])
        with self.store.edit(p["id"]) as saved: saved["reviews"][-1]["canceled"] = True
        with self.assertRaises(Conflict): self.store.merge(p["id"], review["id"])

    def test_concurrent_saves_only_one_wins(self):
        def write(text):
            try: self.store.save_draft(self.p["id"], text, 1); return True
            except Conflict: return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(write, ["a", "b"])), 1)

    def test_transaction_rolls_back(self):
        with self.assertRaises(RuntimeError):
            with self.store.edit(self.p["id"]) as p:
                p["main_id"] = "invalid"
                raise RuntimeError("storage failure")
        self.assertEqual(self.store.get(self.p["id"])["main_id"], self.p["main_id"])

    def test_import_copies_context_survives_source_deletion(self):
        with self.store.edit(self.p["id"]) as p:
            p["documents"].append(parse_document("brief.txt", b"Important evidence"))
            p["sources"].append({"id": "evidence", "title": "Example", "url": "https://example.org"})
            p["revisions"][0]["sources"] = ["evidence"]
        imported = self.store.create("splash-code", "import", source=self.p["id"])
        self.store.delete(self.p["id"])
        imported = self.store.get(imported["id"])
        self.assertEqual(imported["agent"], "splash-code")
        self.assertEqual(imported["documents"][0]["pages"][0]["text"], "Important evidence")
        self.assertEqual(imported["revisions"][0]["sources"], ["evidence"])

    def test_cross_project_revision_rejected(self):
        other = self.store.create("splash", "Other task")
        with self.assertRaises(KeyError): self.store.restore(other["id"], self.p["main_id"], 1)

    def test_exact_unicode_diff(self):
        before, after = "研究 🌙\n Do not delete `x`.  ", "Research 🌙\n Never delete `x`.  "
        parts = diff(before, after)
        self.assertEqual("".join(p["before"] for p in parts), before)
        self.assertEqual("".join(p["after"] for p in parts), after)

    def test_proposal_acceptance_uses_exact_revision(self):
        p = self.p
        p["proposals"] = proposals(p["draft"]["text"], "Changed prompt", 1)
        text, _ = apply_proposals(p, [c["id"] for c in p["proposals"]])
        self.assertEqual(text, "Changed prompt")
        p["draft"]["version"] = 2
        with self.assertRaises(Conflict): apply_proposals(p, [c["id"] for c in p["proposals"]])


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.p = self.store.create("splash-search", "Study renewable energy without removing the cost constraint.")
        self.models = type("Models", (), {"status": {"state": "ready", "config": {"context": 2048}}})()
        self.models.generate = AsyncMock(return_value={"text": "Compare mechanisms and cost constraints. Open questions: reconcile incompatible scopes.", "latency_ms": 10})
        self.models.sections = AsyncMock(side_effect=lambda text, *args: [{"text":text, "start":0, "end":len(text)}])
        self.tools = type("Tools", (), {"call": AsyncMock(return_value={"sources": []})})()
        self.agents = Agents(self.store, self.models, self.tools)

    async def test_five_branches_synthesis_preserves_draft_and_main(self):
        with self.store.edit(self.p["id"]) as p: p["settings"]["branch_count"] = 5
        p = await self.agents.branches(self.p["id"], 1)
        self.assertEqual(len(p["branches"]), 5)
        old = p["draft"]["id"]
        p = await self.agents.synthesize(p["id"], 1, [c["id"] for c in p["branches"]])
        self.assertEqual(p["main_id"], self.p["main_id"])
        self.assertTrue(any(r["id"] == old for r in p["revisions"]))
        variant = next(r for r in p["revisions"] if r["kind"] == "variant")
        self.assertEqual(len(variant["metadata"]["branches"]), 5)

    async def test_selected_branches_not_replaced(self):
        p = await self.agents.branches(self.p["id"], 1)
        with self.store.edit(p["id"]) as p: p["branches"][0]["selected"] = True
        with self.assertRaises(Conflict): await self.agents.branches(p["id"], 1)
        self.assertTrue(self.store.get(p["id"])["branches"][0]["selected"])

    async def test_lossy_enrichment_never_creates_proposals(self):
        p = self.store.create("splash-code", "Read `user_id`. Include tests.")
        self.models.generate.return_value = {"text":"Read the user identifier."}
        with self.assertRaisesRegex(ValueError, "omitted protected content"):
            await self.agents.enrich(p["id"], 1)
        saved = self.store.get(p["id"])
        self.assertEqual(saved["proposals"], [])
        self.assertEqual(saved["draft"]["text"], p["draft"]["text"])

    async def test_synthesis_rejects_unknown_or_six_cards(self):
        for ids in (["unknown"], list("abcdef"), []):
            with self.assertRaises(ValueError): await self.agents.synthesize(self.p["id"], 1, ids)

    async def test_empty_model_branches_do_not_replace_existing_cards(self):
        p = await self.agents.branches(self.p["id"], 1)
        self.models.generate.return_value = {"text":""}
        with self.assertRaisesRegex(ValueError, "empty research direction"):
            await self.agents.branches(p["id"], 1)
        self.assertEqual(self.store.get(p["id"])["branches"], p["branches"])

    async def test_context_proposals_keep_exact_page_provenance(self):
        with self.store.edit(self.p["id"]) as p:
            p["documents"].append({"id":"notes", "name":"brief.pdf", "pages":[{"page":7,"text":"Use observational evidence."}]})
        self.tools.call.return_value = retrieve_context(p, "observational evidence")
        result = await self.agents.enrich(p["id"], 1)
        saved = self.store.get(p["id"])
        self.assertTrue(result["proposals"])
        self.assertEqual(saved["sources"][0]["page"], 7)
        self.assertEqual(result["proposals"][0]["sources"], [saved["sources"][0]["id"]])

    async def test_malformed_planner_has_one_repair_then_fallback(self):
        self.models.generate.return_value = {"text": "not json"}
        plan = await self.agents.tool_plan(self.p, "renewable energy")
        self.assertEqual(self.models.generate.await_count, 2)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["tool"], "search_scholarly")

    async def test_private_scope_and_agent_permissions(self):
        host = ToolHost(self.temp.name)
        self.addAsyncCleanup(host.close)
        project = self.store.create("splash", "Only local context")
        with self.assertRaises(ValueError): await host.call(project, "fetch_source", {"url":"https://example.com"})

    async def test_context_isolation(self):
        other = self.store.create("splash", "Other project")
        with self.store.edit(other["id"]) as p: p["documents"].append(parse_document("private.txt", b"Private detail"))
        self.assertEqual(retrieve_context(self.p, "Private")["passages"], [])

    async def test_private_network_and_redirect_blocked(self):
        resolved = [(2, 1, 6, "", ("127.0.0.1", 443))]
        with patch("backend.research.socket.getaddrinfo", return_value=resolved):
            with self.assertRaises(ValueError): await public_get("https://example.org")
        with self.assertRaises(ValueError): await public_get("file:///etc/passwd")

    async def test_scholar_captcha_stops_and_cache_reuses(self):
        providers = Providers()
        with patch("backend.research.public_get", AsyncMock(return_value=(b"unusual traffic captcha", "text/html", "https://scholar.google.com"))) as fetch:
            with self.assertRaises(ValueError): await providers.scholarly("test", "google_scholar")
            with self.assertRaises(ValueError): await providers.scholarly("new", "google_scholar")
            self.assertEqual(fetch.await_count, 1)
        operation = AsyncMock(return_value=["cached"])
        self.assertEqual(await providers.cached("a", operation), ["cached"])
        self.assertEqual(await providers.cached("a", operation), ["cached"])
        self.assertEqual(operation.await_count, 1)

    async def test_research_edits_require_acceptance_and_skip_unchanged(self):
        with self.store.edit(self.p["id"]) as p: p["settings"]["research"] = True
        self.models.status["state"] = "unloaded"
        source = {"id":"paper", "url":"https://example.org", "title":"Potential source", "text":"A supporting abstract", "provider":"crossref", "level":"abstract"}
        self.tools.call.return_value = {"sources": [source]}
        await self.agents.research(p["id"], 1)
        saved = self.store.get(p["id"])
        self.assertEqual(saved["draft"]["text"], p["draft"]["text"])
        self.assertEqual(saved["main_id"], p["main_id"])
        self.assertEqual(len(saved["proposals"]), 1)
        self.assertEqual((await self.agents.research(p["id"], 1))["status"], "unchanged")
        with self.store.edit(p["id"]) as updated:
            self.store.replace_draft(updated, updated["draft"]["text"] + " Consult the cited paper.", ["paper"])
        await self.agents.research(p["id"], 2, force=True)
        self.assertEqual(self.store.get(p["id"])["proposals"], [])


if __name__ == "__main__": unittest.main()
