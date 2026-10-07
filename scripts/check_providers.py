"""Optional live public-provider smoke check; never sends project data."""
import asyncio
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.research import Providers

async def main():
    providers = Providers()
    results = {}
    checks = {
        "crossref": lambda: providers.scholarly("diffusion language models", "crossref"),
        "semantic_scholar": lambda: providers.scholarly("diffusion language models"),
        "official_docs": lambda: providers.official("Python asyncio documentation"),
        "fetch_source": lambda: providers.fetch("https://docs.python.org/3/library/asyncio.html"),
    }
    for name, check in checks.items():
        started = time.perf_counter()
        try:
            result = await check()
            items = result if isinstance(result, list) else [result]
            results[name] = {"status": "passed" if items else "empty", "results": len(items),
                             "example": [{"title": r["title"], "url": r["url"], "level": r["level"]} for r in items[:1]]}
        except Exception as exc:
            results[name] = {"status": "unavailable", "error": str(exc)}
        results[name]["latency_ms"] = round((time.perf_counter() - started) * 1000)
    print(json.dumps(results, indent=2))
    Path("artifacts/provider-check.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__": asyncio.run(main())
