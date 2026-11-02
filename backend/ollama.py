"""Explicit local Ollama management; inspecting a server never loads weights."""
import os
import re
import ipaddress
from urllib.parse import urlsplit

OLLAMA_URL = os.environ.get("SPLASH_OLLAMA_URL", "http://127.0.0.1:11434")

# Approximate quantized download/runtime footprints from Ollama's public catalog.
# Runtime compatibility is checked against free memory with an additional reserve.
OLLAMA_CATALOG = {
    "qwen3.5:4b": {"name": "Qwen3.5 4B", "download_gb": 3.4, "ram_gb": 6, "context": 32768, "cpu_default": True},
    "qwen3.5:9b": {"name": "Qwen3.5 9B", "download_gb": 6.6, "ram_gb": 9, "context": 32768},
    "qwen3.5:27b": {"name": "Qwen3.5 27B", "download_gb": 17, "ram_gb": 22, "context": 32768},
    "gemma4:26b": {"name": "Gemma4 26B", "download_gb": 16, "ram_gb": 21, "context": 32768},
    "qwen3.5:35b": {"name": "Qwen3.5 35B", "download_gb": 22, "ram_gb": 28, "context": 32768},
    "qwen3.5:122b": {"name": "Qwen3.5 122B", "download_gb": 81, "ram_gb": 96, "context": 32768},
}
EMBEDDING_MODEL = "embeddinggemma:300m"


def select_ollama_model(gpus, available_ram_gb, manual=None):
    """Choose the largest catalog model that fits free VRAM (with 20% reserve)."""
    if manual:
        if manual not in OLLAMA_CATALOG:
            raise ValueError("Choose a supported Ollama model from the catalog.")
        return manual
    candidates = []
    eligible = [item for item in gpus if gpu_supports_large_models(item.get("name", ""))]
    gpu = max(eligible, key=lambda item: item.get("free_mb", 0), default=None)
    if gpu:
        free_gb = gpu.get("free_mb", 0) / 1024
        for model, info in OLLAMA_CATALOG.items():
            if info["ram_gb"] * 1.2 <= free_gb and info["ram_gb"] <= available_ram_gb:
                candidates.append(model)
    if candidates:
        return max(candidates, key=lambda model: OLLAMA_CATALOG[model]["ram_gb"])
    if available_ram_gb < OLLAMA_CATALOG["qwen3.5:4b"]["ram_gb"]:
        raise ValueError("No supported Ollama model fits the available system memory.")
    return "qwen3.5:4b"


def gpu_supports_large_models(name):
    match = re.search(r"\bRTX\s*(\d{4})\b", name, re.I)
    return bool(match and int(match.group(1)) >= 2070)


class OllamaClient:
    def __init__(self, base_url=None):
        self.url = base_url or OLLAMA_URL

    @property
    def base_url(self):
        parsed = urlsplit(self.url)
        host = parsed.hostname or ""
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.casefold() == "localhost"
        if parsed.scheme != "http" or not loopback or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("Splash only connects to Ollama on this computer over HTTP loopback.")
        return self.url.rstrip("/")

    async def request(self, method, path, **kwargs):
        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(10 if method == "GET" else 600, connect=3), trust_env=False) as client:
            response = await client.request(method, self.base_url + path, **kwargs)
            response.raise_for_status()
            return response

    async def inspect(self):
        try:
            response = await self.request("GET", "/api/tags")
            return {"available": True, "installed_models": sorted(item["name"] for item in response.json().get("models", []))}
        except Exception as exc:
            return {"available": False, "installed_models": [], "error": str(exc)}
