"""Download pinned checkpoints and record actual generation results, never simulated scores."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.models import CATALOG, DiffusionAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["compact"], choices=list(CATALOG))
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--output", default="artifacts/model-qualification.json")
    args = parser.parse_args()
    os.environ.setdefault("HF_MODULES_CACHE", str(Path(".models/modules").resolve()))
    results = []
    for key in args.models:
        info = CATALOG[key]
        path = Path(".models") / key
        if args.download:
            from huggingface_hub import snapshot_download
            snapshot_download(info["repo"], revision=info["revision"], local_dir=path)
        for device in args.devices:
            record = {"model": key, "revision": info["revision"], "device": device, "quantize": args.quantize,
                      "steps": args.steps, "output_budget": args.tokens,
                      "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            adapter = None
            try:
                adapter = DiffusionAdapter({"model": key, "path": str(path.resolve()), "device": device,
                                            "quantize": args.quantize, "steps": args.steps})
                import torch
                record["device_name"] = torch.cuda.get_device_name(device) if device.startswith("cuda") else device
                record["samples"] = []
                for prompt in ["Rewrite this as a clear precise instruction, without answering it: Explain caching in two sentences.",
                               "Improve this coding prompt without changing constraints: Build a Python CLI that reads a UTF-8 file. Do not use external dependencies. Include tests."]:
                    record["samples"].append({"prompt": prompt, **adapter.generate([{"role": "user", "content": prompt}], threading.Event(), args.tokens)})
                record["execution"] = "passed"
                record["quality"] = "requires human review; execution is not a quality guarantee"
            except Exception as exc:
                import traceback
                traceback.print_exc()
                record.update(execution="failed", error=f"{type(exc).__name__}: {exc}")
            results.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            del adapter
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
