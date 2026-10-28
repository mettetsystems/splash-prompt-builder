"""Model catalog, device inspection, and a single cancellable inference process."""
import asyncio
import gc
import importlib.util
import ipaddress
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import subprocess
import time
from urllib.parse import urlsplit

from backend.storage import data_directory

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
            if info["ram_gb"] * 0.8 <= free_gb and info["ram_gb"] <= available_ram_gb:
                candidates.append(model)
    if candidates:
        return max(candidates, key=lambda model: OLLAMA_CATALOG[model]["ram_gb"])
    if available_ram_gb < OLLAMA_CATALOG["qwen3.5:4b"]["ram_gb"]:
        raise ValueError("No supported Ollama model fits the available system memory.")
    return "qwen3.5:4b"


def gpu_supports_large_models(name):
    match = re.search(r"\bRTX\s*(\d{4})\b", name, re.I)
    return bool(match and int(match.group(1)) >= 2070)

# Match the PCI ordering used by the hardware scanner instead of CUDA's speed ordering.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def configure_torch_runtime(torch, device):
    """Keep CPU math pools bounded and enable fast NVIDIA math where supported."""
    try:
        requested = int(os.environ.get("SPLASH_INFERENCE_THREADS", "4"))
    except ValueError:
        requested = 4
    available_cpus = (getattr(os, "process_cpu_count", os.cpu_count)() or 1)
    threads = max(1, min(requested, available_cpus))
    torch.set_num_threads(threads)
    # Inter-op work is small for this single-request, single-model worker. A single
    # dispatcher avoids multiplying thread pools on laptops with many logical CPUs.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows this before the first inter-op operation; safe to keep
        # the process's existing value if another import initialized it first.
        pass
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    return threads

CATALOG = {
    "compact": {"repo": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", "revision": "c8d24a3f4adaeef46881b450e1bf7d1005203bd7",
                "name": "Qwen3 diffusion · compact", "download_gb": 1.6, "context": 1024, "ram_gb": 4},
    "dream": {"repo": "Dream-org/Dream-v0-Instruct-7B", "revision": "05334cb9faaf763692dcf9d8737c642be2b2a6ae",
              "name": "Dream 7B · balanced", "download_gb": 15, "context": 2048, "ram_gb": 20},
    "llada": {"repo": "inclusionAI/LLaDA2.1-mini", "revision": "20e64e2ad21644d0e5248586ed9c942cdd45de0f",
              "name": "LLaDA 2.1 mini · extended", "download_gb": 34, "context": 32768, "ram_gb": 40},
}


def hardware():
    import psutil
    result = {"os": platform.system(), "machine": platform.machine(), "cpu": platform.processor() or platform.machine(),
              "cores": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
              "available_ram_gb": round(psutil.virtual_memory().available / 2**30, 1),
              "disk_free_gb": round(shutil.disk_usage(data_directory()).free / 2**30, 1),
              "gpus": [], "gpu_status": "not detected", "metal": "unverified", "profiles": []}
    try:
        run = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        if run.returncode:
            result["gpu_status"] = "GPU inspection unavailable: driver or process permissions; this does not prove hardware is absent."
        else:
            for line in run.stdout.strip().splitlines():
                index, name, total, free, capability, driver = [s.strip() for s in line.split(",")]
                result["gpus"].append({"index": int(index), "name": name, "vram_mb": int(total), "free_mb": int(free),
                                       "capability": capability, "driver": driver})
            result["gpu_status"] = "detected; generation check required"
    except (OSError, subprocess.TimeoutExpired):
        pass
    for gpu in result["gpus"]:
        old = float(gpu["capability"]) < 7.5
        result["profiles"].append({"device": f"cuda:{gpu['index']}", "label": gpu["name"],
            "runtime": "Ollama GPU acceleration" if gpu_supports_large_models(gpu["name"]) else "CPU profile for this GPU class",
            "candidates": ([key for key, model in OLLAMA_CATALOG.items() if key != "qwen3.5:4b"
                            and gpu["free_mb"] >= model["ram_gb"] * 0.8 * 1024
                            and result["available_ram_gb"] >= model["ram_gb"]]
                           if gpu_supports_large_models(gpu["name"]) else ["qwen3.5:4b"]),
            "verified": False})
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        result["profiles"].append({
            "device": "mps", "label": "Apple Metal",
            "runtime": "Unverified until a load check passes; CPU remains available",
            "candidates": ["qwen3.5:4b"], "verified": False,
        })
        if not result["gpus"]:
            result["gpu_status"] = "Apple GPU present; Metal execution is unverified until a load check passes"
    result["profiles"].append({"device": "cpu", "label": "CPU · slower updates", "candidates": ["qwen3.5:4b"], "verified": False})
    return result


class DiffusionAdapter:
    def __init__(self, config):
        import torch
        configure_torch_runtime(torch, config.get("device", "cpu"))
        # Torch 2.14 routes some eager operations through compiled Triton kernels.
        # Keep the portable eager profile independent of a local C/Python toolchain.
        if hasattr(torch, "_native"):
            torch._native.registry.deregister_op_overrides(disable_dispatch_keys="CUDA")
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.key = config["model"]
        self.info = CATALOG[self.key]
        self.device = config.get("device", "cpu")
        self.steps = int(config.get("steps", 64))
        path = config["path"]
        self.context = min(int(config.get("context", self.info["context"])), self.info["context"])
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to this process. Choose CPU or check GPU permissions/runtime.")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Metal execution is unavailable. Choose CPU.")
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        if self.device.startswith("cuda") and torch.cuda.get_device_capability(self.device)[0] >= 8:
            dtype = torch.bfloat16
        # SDPA selects the best supported attention kernel (including Flash
        # Attention on recent CUDA GPUs). LLaDA's compatibility mask requires eager;
        # MPS keeps eager until its attention path is qualified across releases.
        attention = "eager" if self.key == "llada" or self.device == "mps" else "sdpa"
        kwargs = dict(local_files_only=True, trust_remote_code=True, torch_dtype=dtype,
                      attn_implementation=attention)
        if config.get("quantize"):
            if not self.device.startswith("cuda"):
                raise ValueError("NF4 requires a verified CUDA runtime in this release.")
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            kwargs["device_map"] = {"": self.device}
        elif config.get("offload"):
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {int(self.device.split(":")[1]): config.get("gpu_memory", "6GiB"), "cpu": config.get("cpu_memory", "24GiB")}
            kwargs["offload_folder"] = str(data_directory() / "offload")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        self.mask_token_id = None
        if self.key == "compact":
            self.mask_token_id = self.tokenizer.mask_token_id
            if self.mask_token_id is None:
                # Some tokenizer loaders retain the vocabulary entry but fail to
                # restore its mask-token role from the checkpoint metadata.
                token_id = self.tokenizer.convert_tokens_to_ids("<|mask|>")
                if token_id is not None and token_id != self.tokenizer.unk_token_id:
                    self.tokenizer.mask_token = "<|mask|>"
                    self.mask_token_id = self.tokenizer.mask_token_id
            if self.mask_token_id is None:
                raise ValueError(
                    "The Compact checkpoint tokenizer is missing the <|mask|> token. "
                    "Clear the custom checkpoint path and download the pinned Compact checkpoint."
                )
        if self.key == "compact":
            # Import the pinned local module directly: its __main__ demo imports dllm,
            # which HF's static dependency scanner otherwise incorrectly requires.
            from backend.vendor import a2d as module
            model_config = module.A2DQwen3Config.from_pretrained(path, local_files_only=True)
            self.model = module.A2DQwen3LMHeadModel.from_pretrained(path, config=model_config, **kwargs)
        elif self.key == "dream":
            self.model = AutoModel.from_pretrained(path, **kwargs)
        else:
            import transformers.masking_utils as masking
            if not hasattr(masking, "create_bidirectional_mask"):
                from backend.vendor.masking import create_bidirectional_mask
                masking.create_bidirectional_mask = create_bidirectional_mask
            model_config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            if not hasattr(model_config, "rope_parameters"):
                model_config.rope_parameters = {"rope_type":"default", "rope_theta":model_config.rope_theta,
                                                "partial_rotary_factor":model_config.partial_rotary_factor}
            self.model = AutoModelForCausalLM.from_pretrained(path, config=model_config, **kwargs)
        if not config.get("quantize") and not config.get("offload"):
            self.model.to(self.device)
        self.model.eval()

    def generate(self, messages, cancel, max_tokens=256):
        import torch
        tokenizer, model = self.tokenizer, self.model
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            return_tensors="pt", enable_thinking=False).to(self.device)
        max_tokens = min(max_tokens, self.context // 2)
        if ids.shape[1] + max_tokens > self.context:
            raise ValueError(f"Context budget exceeded: {ids.shape[1]} input + {max_tokens} output > {self.context}. Split the input or reduce context.")
        def check(*args):
            if cancel.is_set():
                raise InterruptedError("Generation superseded")
            return args[1] if len(args) > 1 else None
        started = time.perf_counter()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            if self.key == "dream":
                output = model.diffusion_generate(ids, max_new_tokens=max_tokens, steps=self.steps, temperature=0,
                    alg="entropy", generation_tokens_hook_func=check)
                output = output[:, ids.shape[1]:]
            elif self.key == "llada":
                # Check cancellation at each forward pass, including diffusion editing steps.
                hook = model.get_decoder().register_forward_pre_hook(lambda *_: check())
                try:
                    output = model.generate(inputs=ids, gen_length=max_tokens, block_length=32,
                        threshold=0.5, editing_threshold=0, temperature=0, eos_early_stop=True)
                finally:
                    hook.remove()
            else:
                mask = self.mask_token_id
                output = torch.full((1, ids.shape[1] + max_tokens), mask, dtype=torch.long, device=self.device)
                output[:, :ids.shape[1]] = ids
                # Blockwise confidence-based masked diffusion; prompt tokens stay fixed.
                block_size = 32
                blocks = (max_tokens + block_size - 1) // block_size
                for block in range(blocks):
                    start = ids.shape[1] + block * block_size
                    end = min(start + block_size, output.shape[1])
                    steps = min(end - start, max(1, self.steps // blocks))
                    for step in range(steps):
                        check()
                        logits = model(output, use_cache=False).logits[:, start:end, :].float()
                        logits[..., mask] = -float("inf")
                        predicted = logits.argmax(-1)
                        confidence = torch.softmax(logits, -1).gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
                        remaining = output[:, start:end] == mask
                        confidence = confidence.masked_fill(~remaining, -float("inf"))
                        count = (int(remaining.sum()) + (steps - step) - 1) // (steps - step)
                        if count:
                            chosen = confidence.topk(count, dim=-1).indices
                            output[:, start:end].scatter_(1, chosen, predicted.gather(1, chosen))
                output = output[:, ids.shape[1]:]
        check()
        tokens = output[0].tolist()
        eos = tokenizer.eos_token_id
        if eos is not None and eos in tokens:
            tokens = tokens[:tokens.index(eos)]
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return {"text": text, "latency_ms": round((time.perf_counter() - started) * 1000),
                "input_tokens": ids.shape[1], "output_tokens": output.shape[1], "context_limit": self.context,
                "peak_vram_mb": round(torch.cuda.max_memory_allocated(self.device) / 2**20) if self.device.startswith("cuda") else None}

    def sections(self, text, system, suffix, max_tokens):
        """Find exact character boundaries using this model's actual tokenizer."""
        tokenizer = self.tokenizer
        overhead = len(tokenizer.apply_chat_template([{"role":"system", "content":system},
            {"role":"user", "content":"Prompt section:\n" + suffix}], tokenize=True, add_generation_prompt=True, enable_thinking=False))
        budget = self.context - min(max_tokens, self.context // 2) - overhead - 16
        if budget < 64:
            raise ValueError("The model context cannot fit the instructions and reference material. Use a larger context profile.")
        sections, offset = [], 0
        while offset < len(text):
            low, high = 1, len(text) - offset
            while low < high:
                middle = (low + high + 1) // 2
                if len(tokenizer.encode(text[offset:offset+middle], add_special_tokens=False)) <= budget:
                    low = middle
                else:
                    high = middle - 1
            end = offset + low
            if end < len(text):
                # Prefer sentence/paragraph boundaries without losing a single character.
                candidates = [text.rfind(separator, offset + low//2, end) for separator in ("\n", ". ", " ")]
                boundary = max(candidates)
                if boundary > offset:
                    end = boundary + 1
            sections.append({"text": text[offset:end], "start": offset, "end": end})
            offset = end
        return {"sections": sections, "input_budget": budget, "context_limit": self.context}


def worker_main(inbox, outbox, cancel):
    adapter = None
    while True:
        job = inbox.get()
        if job is None:
            return
        try:
            if job["op"] == "load":
                adapter = None
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                adapter = DiffusionAdapter(job["config"])
                result = adapter.generate([{"role": "user", "content": "Rewrite as a clear instruction: Explain caching briefly."}], cancel, 64)
                if not result["text"].strip():
                    raise ValueError("The execution check returned no text. Choose another model or diffusion schedule.")
                result["loaded"] = True
            elif job["op"] == "sections":
                if adapter is None:
                    raise RuntimeError("Load a diffusion model first.")
                result = adapter.sections(job["text"], job["system"], job["suffix"], job["max_tokens"])
            else:
                if adapter is None:
                    raise RuntimeError("Select and load a model in Model settings.")
                result = adapter.generate(job["messages"], cancel, job.get("max_tokens", 256))
            outbox.put({"ok": True, **result})
        except Exception as exc:
            outbox.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


class ModelService:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.status = {"state": "unloaded", "model": None, "engine": "ollama", "error": None}
        self.active_priority = 100

    @property
    def base_url(self):
        parsed = urlsplit(OLLAMA_URL)
        host = parsed.hostname or ""
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.casefold() == "localhost"
        if parsed.scheme != "http" or not loopback or parsed.username or parsed.password:
            raise ValueError("Splash only connects to Ollama on this computer over HTTP loopback.")
        return OLLAMA_URL.rstrip("/")

    async def request(self, method, path, **kwargs):
        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=3), trust_env=False) as client:
            response = await client.request(method, self.base_url + path, **kwargs)
            response.raise_for_status()
            return response

    async def inspect(self):
        try:
            response = await self.request("GET", "/api/tags")
            tags = {item["name"] for item in response.json().get("models", [])}
            self.status.update(available=True, installed=True, installed_models=sorted(tags), error=None)
        except Exception as exc:
            if self.status.get("state") == "ready":
                self.status["state"] = "error"
            self.status.update(available=False, installed=shutil.which("ollama") is not None,
                               installed_models=[], error=f"Ollama is not responding at {self.base_url}: {exc}")
        self.status["engine"] = "ollama"
        return self.status

    async def load(self, config):
        async with self.lock:
            self.status.update(state="loading", engine="ollama", model=config["model"], config=config)
            try:
                result = await self.request("POST", "/api/chat", json={
                    "model": config["model"], "messages": [{"role": "user", "content": "Rewrite this as a clear instruction: Explain caching briefly."}],
                    "stream": False, "options": {"num_ctx": config.get("context", 4096), "num_predict": 64, "temperature": 0,
                        **({"num_gpu": 0} if config.get("cpu_only") else {})},
                    "keep_alive": "10m"})
                text = result.json().get("message", {}).get("content", "").strip()
                if not text:
                    raise RuntimeError("Ollama returned an empty execution check.")
                await self.request("POST", "/api/embed", json={"model": EMBEDDING_MODEL, "input": "intent check", "options": {"num_gpu": 0}})
                self.status.update(state="ready", qualification={"loaded": True, "text": text,
                                    "latency_ms": result.json().get("total_duration", 0) // 1_000_000}, error=None)
            except Exception as exc:
                self.status.update(state="error", error=str(exc))
        return self.status

    async def generate(self, messages, max_tokens=256, priority=0):
        if self.status["state"] != "ready":
            raise RuntimeError("No Ollama model is ready. Open Model & hardware settings to select and load one.")
        if priority > self.active_priority:
            raise InterruptedError("Background inference deferred while editing")
        async with self.lock:
            self.active_priority = priority
            try:
                config = self.status.get("config", {})
                started = time.perf_counter()
                response = await self.request("POST", "/api/chat", json={"model": self.status["model"],
                    "messages": messages, "stream": False, "options": {"num_ctx": config.get("context", 4096),
                    "num_predict": max_tokens, "temperature": 0, **({"num_gpu": 0} if config.get("cpu_only") else {})}})
                payload = response.json()
                result = {"text": payload.get("message", {}).get("content", "").strip(),
                          "latency_ms": round((time.perf_counter()-started)*1000),
                          "input_tokens": payload.get("prompt_eval_count"), "output_tokens": payload.get("eval_count"),
                          "context_limit": config.get("context", 4096)}
                if not result["text"]:
                    raise RuntimeError("Ollama returned an empty response.")
                self.status["last_generation"] = {k:v for k,v in result.items() if k != "text"}
                return result
            finally:
                self.active_priority = 100

    async def sections(self, text, system, suffix="", max_tokens=256):
        if self.status["state"] != "ready":
            raise RuntimeError("Load an Ollama model first.")
        # Keep sentences intact; the cap is generous relative to prompt limits.
        import re
        spans = list(re.finditer(r".+?(?:[.!?。！？]+(?=\s|$)|\n+|$)", text, re.S))
        return [{"text": match.group(), "start": match.start(), "end": match.end()} for match in spans if match.group().strip()]

    async def close(self):
        return None
