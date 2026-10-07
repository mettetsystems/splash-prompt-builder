"""Inspect and retrieve compatible diffusion checkpoints without executing model code."""
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import urlsplit

from backend.models import CATALOG

MODEL_TYPES = {"compact": "a2d-qwen3", "dream": "Dream", "llada": "llada2_moe"}
HUB = "https://huggingface.co"
REQUIRED_PACKAGES = ("torch", "transformers", "accelerate", "huggingface-hub", "safetensors")


def dependency_status():
    packages = {}
    for name in (*REQUIRED_PACKAGES, "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.executable, "packages": packages,
            "missing": [name for name in REQUIRED_PACKAGES if not packages[name]]}


def repository_id(value):
    value = value.strip().rstrip("/")
    if "://" in value:
        url = urlsplit(value)
        if url.scheme != "https" or url.netloc != "huggingface.co" or url.query or url.fragment:
            raise ValueError("Use a https://huggingface.co/owner/model repository URL or owner/model ID.")
        value = url.path.strip("/")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value) or ".." in value or "--" in value:
        raise ValueError("Enter a Hugging Face owner/model ID. Put the branch, tag, or commit in Revision.")
    return value


def source_spec(model, repo="", revision=""):
    custom = bool(repo.strip() or revision.strip())
    repo = repository_id(repo) if repo.strip() else CATALOG[model]["repo"]
    revision = revision.strip() or ("main" if custom else CATALOG[model]["revision"])
    if len(revision) > 200 or any(ord(c) < 32 for c in revision):
        raise ValueError("Use a valid Hugging Face branch, tag, or commit revision.")
    key = model if not custom else model + "-" + hashlib.sha256(f"{repo}@{revision}".encode()).hexdigest()[:20]
    return {"id": key, "model": model, "repo": repo, "requested_revision": revision,
            "source": "huggingface" if custom else "catalog"}


def read_config(path, model):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError("Choose a checkpoint folder with a valid config.json (maximum 2 MB).")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("config.json is not valid JSON. Complete the model download first.") from exc
    kind = config.get("model_type") if isinstance(config, dict) else None
    if kind != MODEL_TYPES[model]:
        raise ValueError(f"This checkpoint uses {kind or 'an unknown model type'}; the selected adapter requires {MODEL_TYPES[model]}. Choose the matching model family. Other architectures need a new adapter.")
    return config


def inspect_local(path, model):
    path = Path(path).expanduser().resolve()
    config = read_config(path / "config.json", model)
    weight_files = []
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / name
        if index.is_file():
            if index.stat().st_size > 10_000_000:
                raise ValueError("The weight index is too large to inspect.")
            try:
                mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError("empty weight map")
                weight_files = list(set(mapping.values()))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("Invalid model weight index. Complete the model download first.") from exc
            break
    if not weight_files:
        weight_files = [name for name in ("model.safetensors", "pytorch_model.bin") if (path / name).is_file()]
    if not weight_files:
        raise ValueError("No Hugging Face model weights found. Supply model.safetensors, pytorch_model.bin, or a complete sharded checkpoint. GGUF files are not supported by these adapters.")
    for name in weight_files:
        if not isinstance(name, str) or "\\" in name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("The weight index contains an invalid file path.")
        weight = path / name
        if not weight.is_file() or weight.stat().st_size == 0:
            raise ValueError(f"Missing or empty weight file: {name}. Finish downloading the checkpoint.")
        with weight.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{name} is a Git LFS pointer, not model weights. Pull the LFS files or use Download from Hugging Face.")
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("Tokenizer files are missing. Select the complete Hugging Face checkpoint directory.")
    return {"path": str(path), "model_type": config["model_type"], "weight_files": len(weight_files),
            "weight_gb": round(sum((path / name).stat().st_size for name in weight_files) / 1e9, 2)}


def inspect_remote(directory, spec):
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise ValueError("Hugging Face support is missing. Run python scripts/setup.py --install-models in the Splash project folder.") from exc
    info = HfApi(endpoint=HUB).model_info(spec["repo"], revision=spec["requested_revision"], files_metadata=True, timeout=20)
    revision = info.sha
    if not revision or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("Hugging Face did not return an immutable checkpoint revision.")
    directory = Path(directory)
    config_path = hf_hub_download(spec["repo"], "config.json", revision=revision,
                                  cache_dir=directory / "models" / ".hub-cache", endpoint=HUB)
    config = read_config(config_path, spec["model"])
    path = directory / "models" / spec["model"] if spec["source"] == "catalog" else (
        directory / "models" / "hub" / hashlib.sha256(spec["repo"].encode()).hexdigest()[:20] / revision)
    sizes = [file.size for file in info.siblings or [] if file.size is not None]
    return {**spec, "revision": revision, "path": str(path), "model_type": config["model_type"],
            "download_gb": round(sum(sizes) / 1e9, 2) if sizes else None}


def fetch_checkpoint(record):
    from huggingface_hub import snapshot_download
    snapshot_download(record["repo"], revision=record["revision"], local_dir=record["path"], endpoint=HUB)
    return inspect_local(record["path"], record["model"])


def downloaded_source(records, spec):
    record = records.get(spec["id"])
    if record and record.get("state") == "complete":
        return record
    # After reopening settings, the saved revision is the immutable SHA rather than its old branch alias.
    for record in records.values():
        if (record.get("state") == "complete" and record.get("repo") == spec["repo"] and
                record.get("model") == spec["model"] and record.get("revision") == spec["requested_revision"]):
            return record
    raise ValueError("Download this Hugging Face checkpoint first, or select Local directory for an existing checkpoint.")
