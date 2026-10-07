"""Read-only evidence retrieval. Public responses are cached; private context is scoped."""
import asyncio
import io
import html
import ipaddress
import json
import re
import socket
import time
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from backend.storage import uid, now

OFFICIAL_HOSTS = ("docs.python.org", "developer.mozilla.org", "typescriptlang.org", "react.dev", "nodejs.org",
                  "docs.rs", "doc.rust-lang.org", "go.dev", "learn.microsoft.com", "docs.github.com")


def clean_html(text):
    return BeautifulSoup(html.unescape(text or ""), "html.parser").get_text(" ", strip=True)


async def public_get(url, limit=2_000_000, headers=None):
    """Pin the connection to a validated public IP, preserving Host and TLS SNI."""
    for _ in range(5):
        parsed = urlsplit(url)
        if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Only public HTTP(S) sources are supported.")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port not in (80, 443):
            raise ValueError("Source URLs must use standard web ports.")
        addresses = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, port, 0, socket.SOCK_STREAM)
        ips = list(dict.fromkeys(a[4][0] for a in addresses))
        if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
            raise ValueError("Local and private network sources are not accessible through research tools.")
        address = f"[{ips[0]}]" if ":" in ips[0] else ips[0]
        target = urlunsplit((parsed.scheme, address, parsed.path or "/", parsed.query, ""))
        request_headers = {"User-Agent": "Splash/2.0 (local prompt research)", **(headers or {}), "Host": parsed.netloc}
        async with httpx.AsyncClient(timeout=15, follow_redirects=False, trust_env=False) as client:
            async with client.stream("GET", target, headers=request_headers, extensions={"sni_hostname": parsed.hostname}) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    url = urljoin(url, response.headers.get("location", ""))
                    continue
                response.raise_for_status()
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > limit:
                        raise ValueError("Source exceeds the retrieval size limit.")
                    chunks.append(chunk)
                return b"".join(chunks), response.headers.get("content-type", ""), url
    raise ValueError("Too many source redirects.")


def parse_document(name, content):
    pages = []
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        if len(reader.pages) > 200:
            raise ValueError("Upload PDFs of at most 200 pages; split larger documents first.")
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": number, "text": text})
        if not pages:
            raise ValueError("This PDF has no extractable text. OCR is not included; paste a transcription.")
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Upload UTF-8 text, code, Markdown, or a text-based PDF.") from exc
        if "\x00" in text:
            raise ValueError("Binary files are not supported.")
        pages = [{"page": 1, "text": text}]
    if sum(len(p["text"]) for p in pages) > 500_000:
        raise ValueError("Extracted context exceeds 500,000 characters; split the document.")
    return {"id": uid(), "name": name[:180], "pages": pages, "created": now(), "origin": None}


def context_chunks(project):
    chunks = []
    for document in project["documents"]:
        for page in document["pages"]:
            for index, start in enumerate(range(0, len(page["text"]), 1400)):
                chunks.append({"id": f"{document['id']}:{page['page']}:{index}", "document_id": document["id"],
                               "name": document["name"], "page": page["page"], "text": page["text"][start:start+1600]})
    return chunks


def retrieve_context(project, query, limit=4):
    terms = set(re.findall(r"\w+", query.casefold()))
    chunks = context_chunks(project)
    ranked = sorted(chunks, key=lambda c: sum(c["text"].casefold().count(t) for t in terms if len(t) > 2), reverse=True)
    selected = ranked[:limit]
    return {"passages": selected, "total_passages": len(chunks), "excluded_passages": max(0, len(chunks)-len(selected))}


def evidence(title, url, text, provider, level="abstract", **extras):
    return {"id": uid(), "title": title or "Untitled source", "url": url, "text": text[:18000],
            "provider": provider, "level": level, "retrieved": now(), **extras}


class Providers:
    def __init__(self):
        self.cache = {}
        self.last_scholar = 0
        self.scholar_blocked = False
        self.next_request = {}

    async def cached(self, key, operation):
        if key in self.cache and time.monotonic() - self.cache[key][0] < 86400:
            return self.cache[key][1]
        result = await operation()
        self.cache[key] = (time.monotonic(), result)
        if len(self.cache) > 500:
            self.cache.pop(next(iter(self.cache)))
        return result

    async def request(self, provider, url):
        if time.monotonic() < self.next_request.get(provider, 0):
            raise ValueError(f"{provider} is cooling down after a request or provider limit.")
        self.next_request[provider] = time.monotonic() + 1
        try:
            return await public_get(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                retry = exc.response.headers.get("retry-after", "60")
                self.next_request[provider] = time.monotonic() + (min(3600, int(retry)) if retry.isdigit() else 60)
            raise

    async def scholarly(self, query, provider="semantic_scholar"):
        query = query.strip()[:240]
        if not query:
            raise ValueError("Search query is empty.")
        if provider not in ("semantic_scholar", "crossref", "google_scholar"):
            raise ValueError("Unknown scholarly provider")
        async def run():
            if provider == "semantic_scholar":
                url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urlencode({"query": query, "limit": 5, "fields": "title,url,abstract,year,externalIds"})
                body, _, _ = await self.request(provider, url)
                return [evidence(r["title"], r["url"], r.get("abstract") or "", provider,
                                 "abstract" if r.get("abstract") else "metadata", year=r.get("year")) for r in json.loads(body).get("data", [])]
            if provider == "crossref":
                url = "https://api.crossref.org/works?" + urlencode({"query.bibliographic": query, "rows": 5})
                body, _, _ = await self.request(provider, url)
                return [evidence((r.get("title") or ["Untitled"])[0], r.get("URL", ""), clean_html(r.get("abstract")), provider,
                                 "abstract" if r.get("abstract") else "metadata", doi=r.get("DOI")) for r in json.loads(body)["message"]["items"]]
            if self.scholar_blocked:
                raise ValueError("Google Scholar blocked automated access. Use another provider; restart the tool server to retry.")
            if time.monotonic() - self.last_scholar < 60:
                raise ValueError("Google Scholar is limited to one request per minute.")
            self.last_scholar = time.monotonic()
            try:
                body, _, _ = await public_get("https://scholar.google.com/scholar?" + urlencode({"q": query, "num": 5}))
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (403, 429):
                    self.scholar_blocked = True
                raise
            html = body.decode("utf-8", errors="replace")
            if any(s in html.lower() for s in ("captcha", "unusual traffic", "automated queries")):
                self.scholar_blocked = True
                raise ValueError("Google Scholar requested human verification; scraping has stopped.")
            soup = BeautifulSoup(html, "html.parser")
            found = []
            for result in soup.select(".gs_r.gs_or.gs_scl")[:5]:
                title, snippet = result.select_one(".gs_rt"), result.select_one(".gs_rs")
                link = title.find("a") if title else None
                if title and link:
                    found.append(evidence(title.get_text(" ", strip=True), link.get("href", ""),
                                          snippet.get_text(" ", strip=True) if snippet else "", provider, "snippet"))
            return found
        return await self.cached((provider, query), run)

    async def official(self, query):
        async def run():
            restricted = query[:200] + " (" + " OR ".join("site:"+h for h in OFFICIAL_HOSTS) + ")"
            body, _, _ = await self.request("official_docs", "https://www.bing.com/search?" + urlencode({"q": restricted, "format": "rss"}))
            results = []
            for item in ET.fromstring(body).findall(".//item"):
                url = item.findtext("link", "")
                host = urlsplit(url).hostname or ""
                if any(host == h or host.endswith("." + h) for h in OFFICIAL_HOSTS):
                    results.append(evidence(item.findtext("title", ""), url, clean_html(item.findtext("description", "")), "official_docs", "snippet"))
            return results[:5]
        return await self.cached(("official_docs", query), run)

    async def fetch(self, url):
        async def run():
            body, content_type, final_url = await public_get(url, limit=8_000_000)
            if "pdf" in content_type:
                document = parse_document("source.pdf", body)
                text = "\n".join(f"Page {p['page']}: {p['text']}" for p in document["pages"])
                return evidence(urlsplit(final_url).path.split("/")[-1], final_url, text, "public_url", "full_text_excerpt")
            soup = BeautifulSoup(body, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else final_url
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = (soup.find("main") or soup.find("article") or soup).get_text(" ", strip=True)
            return evidence(title, final_url, text, "public_url", "full_text_excerpt")
        return await self.cached(("fetch", url), run)
