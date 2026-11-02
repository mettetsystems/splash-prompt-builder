"""Model catalog, device inspection, and a single cancellable inference process."""
import asyncio
import gc
import importlib.util
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import sys
import time

from backend.storage import data_directory

logger = logging.getLogger(__name__)


def cuda_memory_preflight(config, torch):
    """Estimate load headroom before reading weights; this is not a context-fit guarantee."""
    device = config.get("device", "cpu")
    if not device.startswith("cuda"):
        return None
    # Query the actual logical device in this worker, including CUDA_VISIBLE_DEVICES.
    free, total = torch.cuda.mem_get_info(device)
    gib = 2**30
    if config.get("offload"):
        from accelerate.utils import convert_file_size_to_int
        weights = convert_file_size_to_int(config.get("gpu_memory", "6GiB"))
        basis = "the configured GPU offload budget"
    else:
        from backend.checkpoints import inspect_local
        checked = inspect_local(config["path"], config["model"])
        metadata = json.loads((Path(config["path"]) / "config.json").read_text(encoding="utf-8"))
        stored_dtype = str(metadata.get("dtype", metadata.get("torch_dtype", "bfloat16"))).removeprefix("torch.")
        stored_bytes = {"float32": 4, "float64": 8, "float16": 2, "bfloat16": 2}.get(stored_dtype, 2)
        weights = checked["weight_gb"] * 1e9
        if not metadata.get("quantization_config"):
            weights *= 2 / stored_bytes  # CUDA adapters use 16-bit weights.
            if config.get("quantize"):
                # NF4 also retains unquantized layers, scales, and loading buffers.
                weights *= .35
        basis = "checkpoint size and " + ("NF4 overhead" if config.get("quantize") else "16-bit weights")
    required = int(weights + max(.75 * gib, weights * .1))
    report = {"device": device, "free_gib": round(free / gib, 2), "total_gib": round(total / gib, 2),
              "estimated_required_gib": round(required / gib, 2), "basis": basis}
    if free < required:
        raise RuntimeError(
            f"Not enough free GPU memory on {device} ({torch.cuda.get_device_name(device)}): "
            f"{free / gib:.1f} GiB free of {total / gib:.1f} GiB; estimated load requirement "
            f"{required / gib:.1f} GiB including headroom, based on {basis}. "
            "No checkpoint weights were loaded. Free memory in other GPU applications, or choose a smaller "
            "model or another device in Model & hardware, then retry. The editor remains available. "
            "Reducing context alone will not make the model weights fit.")
    return report


def release_model_memory():
    # Run outside exception handlers: tracebacks can retain a partially built model.
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

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


def decoded_generation(tokenizer, tokens, max_tokens):
    eos = tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos] if eos is not None else []
    message_ends = []
    # The bundled chat templates end assistant turns with these tokens, which
    # are distinct from end-of-text. Do not decode a subsequent hallucinated
    # turn or label a completed assistant message as a truncated response.
    if hasattr(tokenizer, 'convert_tokens_to_ids') and hasattr(tokenizer, 'convert_ids_to_tokens'):
        for marker in ('<|im_end|>', '<|role_end|>'):
            token = tokenizer.convert_tokens_to_ids(marker)
            if isinstance(token, int) and tokenizer.convert_ids_to_tokens(token) == marker:
                message_ends.append(token)
    stop = next((i for i, token in enumerate(tokens) if token in eos_ids + message_ends), None)
    end_kind = 'message_end' if stop is not None and tokens[stop] in message_ends else 'eos'
    generated = len(tokens)
    if stop is not None:
        tokens = tokens[:stop]
    masked = getattr(tokenizer, 'mask_token_id', None) in tokens
    truncated = masked or (stop is None and generated >= max_tokens)
    return {'text': tokenizer.decode(tokens, skip_special_tokens=True).strip(), 'output_tokens': len(tokens),
            'max_output_tokens': max_tokens, 'truncated': truncated,
            'stop_reason': 'masked' if masked else end_kind if stop is not None else 'length' if truncated else 'stopped'}


def generation_schedule(model, steps, max_tokens, quality_mode='quality'):
    if model == 'llada':
        return {'threshold': .7 if quality_mode == 'quality' else .5,
                'editing_threshold': .5 if quality_mode == 'quality' else 0, 'max_post_steps': 16}
    return {'steps': max(steps, (max_tokens + 1) // 2) if quality_mode == 'quality' else steps}


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
            "runtime": "Legacy NVIDIA / PyTorch 2.14.1 CUDA 12.6 (setup --profile nvidia-legacy)" if old else "Modern CUDA (setup --profile nvidia)",
            "candidates": [key for key, model in CATALOG.items() if key == "compact" or gpu["free_mb"] > model["download_gb"] * 1100],
            "verified": False})
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        result["profiles"].append({
            "device": "mps", "label": "Apple Metal",
            "runtime": "Unverified until a load check passes; CPU remains available",
            "candidates": ["compact"], "verified": False,
        })
        if not result["gpus"]:
            result["gpu_status"] = "Apple GPU present; Metal execution is unverified until a load check passes"
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
        self.quality_mode = config.get('quality_mode', 'quality')
        path = config["path"]
        self.context = min(int(config.get("context", self.info["context"])), self.info["context"])
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to this process. Choose CPU or check GPU permissions/runtime.")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Metal execution is unavailable. Choose CPU.")
        self.memory_check = cuda_memory_preflight(config, torch)
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        if self.device.startswith("cuda") and torch.cuda.get_device_capability(self.device)[0] >= 8:
            dtype = torch.bfloat16
        kwargs = dict(local_files_only=True, trust_remote_code=True, dtype=dtype, attn_implementation="eager")
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
        schedule = generation_schedule(self.key, self.steps, max_tokens, self.quality_mode)
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
                output = model.diffusion_generate(ids, max_new_tokens=max_tokens, steps=schedule['steps'], temperature=0,
                    alg="entropy", generation_tokens_hook_func=check)
                output = output[:, ids.shape[1]:]
            elif self.key == "llada":
                # Check cancellation at each forward pass, including diffusion editing steps.
                hook = model.get_decoder().register_forward_pre_hook(lambda *_: check())
                try:
                    output = model.generate(inputs=ids, gen_length=max_tokens, block_length=32,
                        **schedule, temperature=0, eos_early_stop=True)
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
                    steps = min(end - start, max(1, schedule['steps'] // blocks))
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
        decoded = decoded_generation(tokenizer, output[0].tolist(), max_tokens)
        return {**decoded, "latency_ms": round((time.perf_counter() - started) * 1000),
                "input_tokens": ids.shape[1], "context_limit": self.context,
                "quality_mode": self.quality_mode, "schedule": schedule,
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

    def plan_review(self, text, system, references):
        from backend.enrichment import plan_review
        return plan_review(self.tokenizer, self.context, text, system, references)


def worker_main(inbox, outbox, cancel):
    import signal

    def shutdown(signum, frame):
        # Let multiprocessing's finalizers release child-owned semaphores when
        # a slow checkpoint load needs to be interrupted during app shutdown.
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        run_worker(inbox, outbox, cancel)
    finally:
        for channel in (inbox, outbox):
            channel.cancel_join_thread()
            channel.close()


def run_worker(inbox, outbox, cancel):
    adapter = None
    while True:
        job = inbox.get()
        if job is None:
            return
        failure = None
        try:
            if job["op"] == "load":
                adapter = None
                release_model_memory()
                adapter = DiffusionAdapter(job["config"])
                result = adapter.generate([{"role": "user", "content": "Rewrite as a clear instruction: Explain caching briefly."}], cancel, 64)
                if not result["text"].strip():
                    raise ValueError("The execution check returned no text. Choose another model or diffusion schedule.")
                result["loaded"] = True
                result["memory_check"] = adapter.memory_check
            elif job["op"] == "sections":
                if adapter is None:
                    raise RuntimeError("Load a diffusion model first.")
                result = adapter.sections(job["text"], job["system"], job["suffix"], job["max_tokens"])
            elif job["op"] == "plan_review":
                if adapter is None:
                    raise RuntimeError("Load a diffusion model first.")
                result = adapter.plan_review(job["text"], job["system"], job["references"])
            else:
                if adapter is None:
                    raise RuntimeError("Select and load a model in Model settings.")
                result = adapter.generate(job["messages"], cancel, job.get("max_tokens", 256))
            outbox.put({"ok": True, **result})
        except Exception as exc:
            failure = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if failure is not None:
            if job["op"] == "load":
                adapter = None
            # Clear traceback-held tensors and allocator caches before reporting failure.
            release_model_memory()
            outbox.put(failure)


class ModelService:
    def __init__(self):
        context = mp.get_context("spawn")
        self.inbox, self.outbox, self.cancel = context.Queue(), context.Queue(), context.Event()
        self.process = None
        self.lock = asyncio.Lock()
        self.sequence = 0
        self.status = {"state": "unloaded", "model": None, "engine": "diffusion", "error": None}
        self.active_priority = 100
        self.closed = False
        self.endpoint = None
        self.endpoint_request = None

    def invalidate(self):
        self.sequence += 1
        self.cancel.set()
        if self.endpoint_request and not self.endpoint_request.done():
            self.endpoint_request.cancel()

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
            external = config.get('source') == 'endpoint'
            self.status = {"state": "loading", "engine": "endpoint" if external else "diffusion", "model": config["model"], "config": config}
            try:
                self.endpoint = None
                if external:
                    from backend.endpoint_model import EndpointModel
                    await self.stop_worker()
                    adapter = EndpointModel(config)
                    result = await adapter.verify()
                    self.endpoint = adapter
                    self.status['config'] = adapter.config
                else:
                    result = await self.exchange({"op": "load", "config": config})
                    if not result["ok"]:
                        raise RuntimeError(result["error"])
                self.status.update(state="ready", qualification=result, error=None)
            except Exception as exc:
                self.status.update(state="error", error=str(exc))
                logger.error("Model load failed; the editor remains available: %s", exc)
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
            external = self.endpoint is not None
            task = asyncio.create_task(self.endpoint.generate(messages, max_tokens) if external else
                self.exchange({"op": "generate", "messages": messages, "max_tokens": max_tokens}))
            if external:
                self.endpoint_request = task
            try:
                result = await asyncio.shield(task)
                if ticket != self.sequence:
                    raise InterruptedError("Generation superseded")
                if not result["ok"]:
                    raise RuntimeError(result["error"])
                self.status["last_generation"] = {k:v for k,v in result.items() if k != "text"}
                return result
            except asyncio.CancelledError:
                if external:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                else:
                    self.cancel.set()
                    await task  # Drain the worker result before another request owns the queue.
                raise
            finally:
                if self.endpoint_request is task:
                    self.endpoint_request = None
                self.active_priority = 100

    async def sections(self, text, system, suffix="", max_tokens=256):
        result = await self.plan({"op":"sections", "text":text, "system":system, "suffix":suffix, "max_tokens":max_tokens})
        return result["sections"]

    async def plan_review(self, text, system, references):
        return await self.plan({"op":"plan_review", "text":text, "system":system, "references":references})

    async def plan(self, payload):
        if self.status["state"] != "ready":
            raise RuntimeError("Load a diffusion model first.")
        async with self.lock:
            if self.endpoint:
                if payload['op'] == 'sections':
                    return self.endpoint.sections(payload['text'], payload['system'], payload['suffix'], payload['max_tokens'])
                return self.endpoint.plan_review(payload['text'], payload['system'], payload['references'])
            task = asyncio.create_task(self.exchange(payload))
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            if not result["ok"]:
                raise RuntimeError(result["error"])
            return result

    async def stop_worker(self):
        if self.process and self.process.is_alive():
            self.inbox.put(None)
            await asyncio.to_thread(self.process.join, 3)
            if self.process.is_alive():
                self.process.terminate()
                await asyncio.to_thread(self.process.join, 2)
            if self.process.is_alive():
                self.process.kill()
                await asyncio.to_thread(self.process.join, 2)
        if self.process and not self.process.is_alive():
            self.process.close()
            self.process = None
            for channel in (self.inbox, self.outbox):
                channel.cancel_join_thread()
                channel.close()
            context = mp.get_context('spawn')
            self.inbox, self.outbox, self.cancel = context.Queue(), context.Queue(), context.Event()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.invalidate()
        await self.stop_worker()
        self.endpoint = None
        for channel in (self.inbox, self.outbox):
            channel.cancel_join_thread()
            channel.close()
        self.process = self.inbox = self.outbox = self.cancel = None
        self.status.update(state="unloaded")
