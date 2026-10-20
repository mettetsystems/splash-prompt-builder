"""Revision-bound document discussions; agents may comment, never resolve intent."""
import hashlib
import json
import re
from difflib import SequenceMatcher

from backend.storage import Conflict, now, uid


def anchor(text, start, end, version):
    if not 0 <= start <= end <= len(text):
        raise ValueError("Select a passage in the current working draft.")
    return {"start": start, "end": end, "quote": text[start:end], "version": version}


def thread(kind, message, anchors, *, author="agent", replacement=None, sources=None, generation=None, origin="review"):
    key = json.dumps([origin, kind, [a["quote"] for a in anchors], replacement], ensure_ascii=False)
    return {"id": uid(), "kind": kind, "status": "open", "outdated": False,
            "anchors": anchors, "replacement": replacement, "sources": sources or [],
            "generation": generation, "origin": origin, "fingerprint": hashlib.sha256(key.encode()).hexdigest(),
            "created": now(), "updated": now(),
            "messages": [{"id": uid(), "author": author, "text": message, "created": now()}]}


def add_threads(project, candidates):
    comments = project.setdefault("comments", [])
    added = []
    for candidate in candidates:
        # A background review must not reopen a resolved discussion or erase replies.
        quotes = {a['quote'] for a in candidate['anchors']}
        if any(not c.get("outdated") and (c.get("fingerprint") == candidate["fingerprint"] or
               (c['kind'] == candidate['kind'] == 'conflict' and {a['quote'] for a in c['anchors']} == quotes)) for c in comments):
            continue
        comments.append(candidate)
        added.append(candidate)
    return added


def reanchor_threads(project, old_text, new_text, version):
    matches = SequenceMatcher(None, old_text, new_text, autojunk=len(old_text) > 12000).get_matching_blocks()
    for comment in project.get("comments", []):
        if comment.get("outdated"):
            continue
        adjusted = []
        for prior in comment["anchors"]:
            block = next((b for b in matches if b.size and b.a <= prior["start"] and prior["end"] <= b.a + b.size), None)
            if block is None:
                comment["outdated"] = True
                break
            start = prior["start"] + block.b - block.a
            adjusted.append(anchor(new_text, start, start + len(prior["quote"]), version))
        if not comment.get("outdated"):
            comment["anchors"] = adjusted


def get_thread(project, comment_id):
    result = next((c for c in project.get("comments", []) if c["id"] == comment_id), None)
    if result is None:
        raise KeyError("Comment not found in this project")
    return result


def apply_suggestion(project, comment):
    if comment["status"] != "open" or comment.get("outdated") or comment.get("replacement") is None:
        raise Conflict("This comment has no current edit to apply. Review the latest passage instead.")
    if comment["kind"] != "suggestion" or len(comment["anchors"]) != 1:
        raise ValueError("Conflicts and questions need your clarification; they cannot be applied automatically.")
    text = project["draft"]["text"]
    a = comment["anchors"][0]
    if a["version"] != project["draft"]["version"] or text[a["start"]:a["end"]] != a["quote"]:
        raise Conflict("The passage changed. Review the latest draft before applying an edit.")
    result = text[:a["start"]] + comment["replacement"] + text[a["end"]:]
    if len(result) > 80000:
        raise ValueError("The suggested edit exceeds the draft size limit.")
    return result


def consistency_comments(text, version):
    """Conservative English pattern checks. These flag possible conflicts, not facts."""
    statements = []
    for match in re.finditer(r".+?(?:[.!?](?=\s|$)|\n|$)", text):
        raw = match.group()
        stripped = raw.strip()
        if stripped:
            start = match.start() + len(raw) - len(raw.lstrip())
            statements.append((stripped, anchor(text, start, start + len(stripped), version)))
    facts, directives, lengths, found = {}, {}, [], []
    def conflict(a, b, message):
        found.append(thread("conflict", message, [a, b], author="consistency", origin="consistency"))
    for statement, a in statements:
        clean = statement.rstrip('.!?').strip()
        fact = re.fullmatch(r"(?:the\s+)?([\w -]{1,60}?)\s+(?:is|must be|will be|:)\s+(.{1,150})", clean, re.I)
        if fact:
            key, value = (re.sub(r"\s+", " ", s).casefold() for s in fact.groups())
            # Different descriptions can both be true ("the website is fast" /
            # "the website is colorful"). Compare only scalar values with the
            # same units; leave broader semantic judgments to the reviewer.
            scalar = re.fullmatch(r"([$€£¥]?)\s*([+-]?\d+(?:\.\d+)?)\s*([a-z%]*)", value)
            if scalar:
                fact_key = (key, scalar[1], scalar[3])
                number = float(scalar[2])
                if fact_key in facts and facts[fact_key][0] != number:
                    conflict(facts[fact_key][1], a, f"These passages give different values for ‘{fact[1]}’. Are they alternatives or different contexts? Clarify which applies.")
                facts[fact_key] = (number, a)
        directive = re.fullmatch(r"(do not |never |don't )?(use|include|allow|enable|collect|store|require|deploy|send)\s+(.{1,150})", clean, re.I)
        if directive:
            negative, verb, target = directive.groups()
            key = (verb.casefold(), re.sub(r"\s+", " ", target).casefold())
            if key in directives and directives[key][0] != bool(negative):
                conflict(directives[key][1], a, "One instruction requires this action and another prohibits it. Which direction should the prompt follow?")
            directives[key] = (bool(negative), a)
        length = re.search(r"\b(exactly|at least|at most|no more than|under|over)\s+(\d+)\s+words\b", clean, re.I)
        if length:
            op, count = length[1].casefold(), int(length[2])
            lower = count + (op == 'over') if op in ('exactly', 'at least', 'over') else 0
            upper = count - (op == 'under') if op in ('exactly', 'at most', 'no more than', 'under') else float('inf')
            for lo, hi, other in lengths:
                if max(lower, lo) > min(upper, hi):
                    conflict(other, a, "These word-count requirements cannot both apply to the same output. Specify separate deliverables or choose a single range.")
            lengths.append((lower, upper, a))
    return found[:8]


def rewrite_comments(before, after, version, sources, generation, limit=8):
    """Whole-word suggestions, with exact character offsets and whitespace retained."""
    a, b = [list(re.finditer(r"\S+\s*|\s+", text)) for text in (before, after)]
    candidates = []
    for kind, i, j, k, l in SequenceMatcher(None, [m.group() for m in a], [m.group() for m in b], autojunk=False).get_opcodes():
        if kind == "equal":
            continue
        start = a[i].start() if i < len(a) else len(before)
        end = a[j-1].end() if j > i else start
        replacement = ''.join(m.group() for m in b[k:l])
        # Give insertions a visible neighbouring word to anchor and review.
        if start == end and a:
            if i:
                start = a[i-1].start()
                replacement = before[start:end] + replacement
            else:
                end = a[0].end()
                replacement += before[start:end]
        candidates.append(thread("suggestion", "Suggested wording for clarity and precision. Check that it keeps your intended meaning.",
            [anchor(before, start, end, version)], replacement=replacement, sources=sources,
            generation=generation, origin="enrichment"))
    return candidates if limit is None else candidates[:limit]


def parse_review(raw, text, version, generation):
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    if raw == 'NONE':
        data = {'comments':[]}
    elif raw.startswith('{'):
        data = json.loads(raw)
    else:
        items = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"(?:-\s*)?(CONFLICT|QUESTION)\s+\[?(\d+(?:\s*,\s*\d+)?)\]?\s*[|:]\s*(.{1,2000})", line.strip(), re.I)
            if not match:
                raise ValueError("Use CONFLICT 1,2 | question or QUESTION 1 | question, one comment per line.")
            items.append({'kind':match[1].lower(), 'passages':[int(i.strip()) for i in match[2].split(',')], 'message':match[3]})
        if not items:
            raise ValueError("The reviewer returned no comments or explicit NONE result.")
        data = {'comments':items}
    if not isinstance(data, dict) or set(data) != {"comments"} or not isinstance(data["comments"], list) or len(data["comments"]) > 6:
        raise ValueError("Return an object containing a comments array of at most six items.")
    comments = []
    for item in data["comments"]:
        if isinstance(item, dict) and set(item) == {"kind", "passages", "message"}:
            passages = review_passages(text)
            ids = item['passages']
            if not isinstance(ids, list) or not 1 <= len(ids) <= 2 or any(type(i) is not int or not 1 <= i <= len(passages) for i in ids):
                raise ValueError("Cite one or two valid passage numbers.")
            if item['kind'] == 'conflict' and (len(set(ids)) != 2):
                raise ValueError("A conflict must cite two different passages.")
            item = {'kind':item['kind'], 'quote':passages[ids[0]-1]['quote'],
                    'related_quote':passages[ids[1]-1]['quote'] if len(ids)>1 else '', 'message':item['message']}
        if not isinstance(item, dict) or set(item) != {"kind", "quote", "related_quote", "message"}:
            raise ValueError("Each comment needs kind, quote, related_quote, and message.")
        if item["kind"] not in ("conflict", "question") or not all(isinstance(item[k], str) for k in ("quote", "related_quote", "message")):
            raise ValueError("Invalid comment fields.")
        if not 1 <= len(item["message"].strip()) <= 2000:
            raise ValueError("A comment needs a concise explanation.")
        quotes = [item["quote"]] + ([item["related_quote"]] if item["related_quote"] else [])
        if item["kind"] == "conflict" and len(set(quotes)) != 2:
            raise ValueError("A conflict must quote two different passages.")
        anchors = []
        for quote in quotes:
            if not quote or text.count(quote) != 1:
                raise ValueError("Quote a unique, exact passage from the prompt.")
            start = text.index(quote)
            anchors.append(anchor(text, start, start+len(quote), version))
        comments.append(thread(item["kind"], item["message"].strip(), anchors, generation=generation))
    return comments


def review_passages(text):
    passages = []
    for match in re.finditer(r".+?(?:[.!?](?=\s|$)|\n|$)", text):
        raw = match.group()
        if raw.strip():
            start = match.start()+len(raw)-len(raw.lstrip())
            passages.append({'start':start, 'quote':raw.strip()})
    return passages


REVIEW_RULES = ('Review the numbered prompt passages as a document commenter. Identify contradictions and important ambiguities. '
    'A conflict exists ONLY when two explicit requirements cannot both be satisfied for the same subject and scope. '
    'Separate objectives, content goals, and visual or writing preferences can coexist; never force a choice between them. '
    'For example, "Explain science to students" and "Use a lively style" are compatible: NONE. '
    '"Work entirely offline" and "Require the internet for every action" conflict. '
    'Do not call an unclear phrase a contradiction; use QUESTION only if clarification materially changes the task. '
    'Do not rewrite, answer the prompt, choose between conflicting claims, or invent facts. '
    'Return up to three comments, exactly one per line, using these formats:\n'
    'CONFLICT 1,2 | Which of these competing requirements should apply?\n'
    'QUESTION 3 | What detail should the author clarify?\n'
    'Replace the numbers with actual passage numbers and write a specific question in English. '
    'A CONFLICT cites two passages; a QUESTION cites one. If there are no issues, return NONE. No introduction or other text.')


async def review_document(agents, pid, expected):
    from backend.state import decision_context
    p = agents.store.get(pid)
    agents.store.check_version(p, expected)
    text = p["draft"]["text"]
    rules = agents.system_content(p, REVIEW_RULES + decision_context(p))
    deterministic = consistency_comments(text, expected)
    comments, errors, checked = [], [], 0
    # Keep local checks available even when model review is unavailable or canceled.
    with agents.store.edit(pid) as current:
        agents.check_generation_context(current, p, expected)
        local_added = add_threads(current, deterministic)
    if agents.models.status["state"] == "ready" and text.strip():
        sections = await agents.models.sections(text, rules, "", 384)
        for section in sections[:4]:
            agents.check_generation_context(agents.store.get(pid), p, expected)
            content = section["text"]
            numbered = '\n'.join(f"[{i+1}] {p['quote']}" for i,p in enumerate(review_passages(content)))
            for attempt in range(2):
                response = await agents.models.generate([
                    {"role": "system", "content": rules},
                    {"role": "user", "content": ("Use only CONFLICT 1,2 | question or QUESTION 1 | question, with the correct numbers.\n" if attempt else "") + numbered}],
                    max_tokens=384, priority=1)
                try:
                    candidates = parse_review(response["text"], content, expected, agents.generation_info())
                    for comment in candidates:
                        for a in comment["anchors"]:
                            a["start"] += section["start"]
                            a["end"] += section["start"]
                    comments.extend(candidates)
                    break
                except (ValueError, TypeError) as exc:
                    if attempt:
                        errors.append("AI review returned invalid comments; local consistency checks remain available. " + str(exc)[:180])
            checked += len(content)
    else:
        errors.append("AI review needs a loaded model. Local consistency checks completed.")
    with agents.store.edit(pid) as current:
        agents.check_generation_context(current, p, expected)
        added = add_threads(current, comments[:12])
        if any(c['kind']=='conflict' and c['status']=='open' and not c['outdated'] for c in current['comments']):
            current['proposals'] = [p for p in current['proposals'] if p['rationale'] != 'Language and instruction precision']
        current["comment_review"] = {"version": expected, "created": now(), "checked_characters": checked,
            "excluded_characters": max(0, len(text)-checked), "errors": errors,
            "model": agents.models.status.get('model') if agents.models.status['state'] == 'ready' else None}
    return {"comments_added": len(added)+len(local_added), "review": current["comment_review"], "errors": errors}


async def answer_comment(agents, pid, comment_id, message_id):
    p = agents.store.get(pid)
    comment = get_thread(p, comment_id)
    context = '\n'.join(f"{m['author']}: {m['text']}" for m in comment["messages"][-4:])
    quotes = '\n'.join(a['quote'] for a in comment['anchors'])
    response = await agents.models.generate([
        {"role":"system", "content":agents.system_content(p, "Reply as a helpful prompt-document reviewer in English. Discuss the author's clarification. Do not execute the prompt or silently settle conflicting facts. Suggest concrete wording when useful. You can only comment; the author controls edits and resolution.")},
        {"role":"user", "content":f"Quoted passages:\n{quotes[:1000]}\nDiscussion:\n{context[:1600]}"}], max_tokens=256, priority=1)
    if not response['text'].strip():
        raise ValueError("The reviewer returned an empty reply. Your comment is saved; try asking again.")
    with agents.store.edit(pid) as current:
        agents.check_generation_context(current, p, p['draft']['version'])
        saved = get_thread(current, comment_id)
        if saved['status'] != 'open' or saved['outdated'] or saved['messages'][-1]['id'] != message_id or current['draft']['version'] != p['draft']['version']:
            raise Conflict("The discussion or passage changed. Ask the reviewer again for the current context.")
        saved['messages'].append({"id":uid(), "author":"agent", "text":response['text'].strip(), "created":now(), "generation":agents.generation_info()})
        saved['updated'] = now()
    return {"comment_id": comment_id}
