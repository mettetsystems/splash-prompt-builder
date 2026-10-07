"""Model catalog, device inspection, and a single cancellable inference process."""
import asyncio
import gc
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import time

from backend.storage import data_directory

# Match the PCI ordering used by the hardware scanner instead of CUDA's speed ordering.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

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
            "runtime": "Pascal / CUDA 11.8 compatibility environment or CPU" if old else "Modern CUDA",
            "candidates": [key for key, model in CATALOG.items() if key == "compact" or gpu["free_mb"] > model["download_gb"] * 1100],
            "verified": False})
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        result["profiles"].append({"device": "mps", "label": "Apple Metal (execution check required)", "candidates": ["compact"], "verified": False})
    result["profiles"].append({"device": "cpu", "label": "CPU · slower updates", "candidates": ["compact"], "verified": False})
    return result


class DiffusionAdapter:
    def __init__(self, config):
        import torch
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
        kwargs = dict(local_files_only=True, trust_remote_code=True, torch_dtype=dtype, attn_implementation="eager")
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
                mask = tokenizer.mask_token_id
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
        context = mp.get_context("spawn")
        self.inbox, self.outbox, self.cancel = context.Queue(), context.Queue(), context.Event()
        self.process = None
        self.lock = asyncio.Lock()
        self.sequence = 0
        self.status = {"state": "unloaded", "model": None, "engine": "diffusion", "error": None}
        self.active_priority = 100

    def invalidate(self):
        self.sequence += 1
        self.cancel.set()

    async def exchange(self, payload):
        if not self.process or not self.process.is_alive():
            if self.process is not None:
                self.process.join(timeout=1)
                self.inbox.close()
                self.outbox.close()
                context = mp.get_context("spawn")
                self.inbox, self.outbox, self.cancel = context.Queue(), context.Queue(), context.Event()
            if payload["op"] != "load":
                self.status.update(state="error", error="The inference process exited. Reload the model.")
                raise RuntimeError(self.status["error"])
            self.process = mp.get_context("spawn").Process(target=worker_main, args=(self.inbox, self.outbox, self.cancel), daemon=True)
            self.process.start()
        self.cancel.clear()
        self.inbox.put(payload)
        started = time.monotonic()
        while True:
            try:
                return self.outbox.get_nowait()
            except queue.Empty:
                if not self.process.is_alive():
                    self.status.update(state="error", error="The inference process exited. Reload the model.")
                    raise RuntimeError("The inference process exited. Reload the model.")
                if time.monotonic() - started > 600:
                    self.process.terminate()
                    raise RuntimeError("Inference exceeded ten minutes. Use a smaller model or shorter input.")
                await asyncio.sleep(.05)

    async def load(self, config):
        self.invalidate()
        async with self.lock:
            self.status = {"state": "loading", "engine": "diffusion", "model": config["model"], "config": config}
            try:
                result = await self.exchange({"op": "load", "config": config})
                if not result["ok"]:
                    raise RuntimeError(result["error"])
                self.status.update(state="ready", qualification=result, error=None)
            except Exception as exc:
                self.status.update(state="error", error=str(exc))
        return self.status

    async def generate(self, messages, max_tokens=256, priority=0):
        if self.status["state"] != "ready":
            raise RuntimeError("No diffusion model is ready. Open Model settings to load and check one.")
        if priority > self.active_priority:
            raise InterruptedError("Background inference deferred while editing")
        self.invalidate()
        ticket = self.sequence
        async with self.lock:
            if ticket != self.sequence:
                raise InterruptedError("Generation superseded")
            self.active_priority = priority
            task = asyncio.create_task(self.exchange({"op": "generate", "messages": messages, "max_tokens": max_tokens}))
            try:
                result = await asyncio.shield(task)
                if ticket != self.sequence:
                    raise InterruptedError("Generation superseded")
                if not result["ok"]:
                    raise RuntimeError(result["error"])
                self.status["last_generation"] = {k:v for k,v in result.items() if k != "text"}
                return result
            except asyncio.CancelledError:
                self.cancel.set()
                await task  # Drain the worker result before another request owns the queue.
                raise
            finally:
                self.active_priority = 100

    async def sections(self, text, system, suffix="", max_tokens=256):
        if self.status["state"] != "ready":
            raise RuntimeError("Load a diffusion model first.")
        async with self.lock:
            task = asyncio.create_task(self.exchange({"op":"sections", "text":text, "system":system, "suffix":suffix, "max_tokens":max_tokens}))
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            if not result["ok"]:
                raise RuntimeError(result["error"])
            return result["sections"]

    async def close(self):
        self.invalidate()
        if self.process and self.process.is_alive():
            self.inbox.put(None)
            await asyncio.to_thread(self.process.join, 3)
            if self.process.is_alive():
                self.process.terminate()
                await asyncio.to_thread(self.process.join, 2)
