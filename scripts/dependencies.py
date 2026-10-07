"""Dependency checks that also work before Splash's virtual environment exists.

The runtime probe runs in the app interpreter, never in the setup interpreter.
It imports no application modules, downloads no checkpoints, and mutates no data.
"""
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys

TORCH_VERSION = "2.14.1"
BITSANDBYTES_VERSION = "0.50.2"
PROFILES = ("auto", "editor", "cpu", "nvidia-legacy", "nvidia", "metal")
LEVELS = {
    "compact": {"name": "Qwen3 diffusion 0.6B", "download_gb": 1.6, "ram_gb": 4, "vram_gb": 2},
    "balanced": {"name": "Dream 7B", "download_gb": 15, "ram_gb": 20, "vram_gb": 16, "nf4_gb": 6},
    "extended": {"name": "LLaDA 2.1 mini", "download_gb": 34, "ram_gb": 40, "vram_gb": 36, "nf4_gb": 12},
}
INDEXES = {"cpu": "cpu", "nvidia-legacy": "cu126", "nvidia": "cu130"}


def app_python(root):
    return Path(root) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command, timeout=15):
    return subprocess.run([str(part) for part in command], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def requirements(path):
    """Read the repository's pinned requirement files, including -r includes."""
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-r "):
            result.update(requirements(Path(path).parent / line[3:].strip()))
        elif line:
            name, version = line.split("==", 1)
            result[name] = version
    return result


def system_ram_gb():
    try:
        if sys.platform == "win32":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            size = status.total
        elif sys.platform == "darwin":
            size = int(run(["sysctl", "-n", "hw.memsize"]).stdout)
        else:
            size = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return round(size / 2**30, 1)
    except (AttributeError, OSError, ValueError, subprocess.TimeoutExpired):
        return None


def inspect_hardware(root):
    hardware = {"os": platform.system(), "machine": platform.machine().lower(),
                "cpu_threads": os.cpu_count(), "ram_gb": system_ram_gb(),
                "disk_free_gb": round(shutil.disk_usage(root).free / 2**30, 1), "gpus": []}
    try:
        result = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap,driver_version",
                      "--format=csv,noheader,nounits"])
        if result.returncode:
            raise ValueError(result.stderr.strip() or "nvidia-smi failed")
        for row in csv.reader(result.stdout.splitlines()):
            name, total, free, capability, driver = [value.strip() for value in row]
            hardware["gpus"].append({"name": name, "vram_gb": round(float(total) / 1024, 1),
                                     "free_gb": round(float(free) / 1024, 1),
                                     "capability": float(capability), "driver": driver})
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        hardware["gpu_note"] = f"NVIDIA inspection unavailable ({exc}). Check nvidia-smi if this computer has an NVIDIA GPU."
    return hardware


def resolve_profile(requested, hardware):
    if requested != "auto":
        return requested
    gpus = [gpu for gpu in hardware["gpus"] if gpu["capability"] >= 5.0]
    if gpus:
        best = max(gpus, key=lambda gpu: (gpu["vram_gb"], gpu["capability"]))
        return "nvidia-legacy" if best["capability"] < 7.5 else "nvidia"
    if hardware["os"] == "Darwin" and hardware["machine"] in ("arm64", "aarch64"):
        return "metal"
    return "cpu"


def profile_problem(profile, hardware, quantize=False):
    if profile == "editor":
        return "NF4 is a model option; choose an NVIDIA profile." if quantize else None
    system, machine = hardware["os"], hardware["machine"]
    if quantize and not profile.startswith("nvidia"):
        return "NF4 is supported only by the NVIDIA profiles. Omit --quantize for CPU or Metal."
    if profile == "metal" and (system != "Darwin" or machine not in ("arm64", "aarch64")):
        return "The Metal profile requires macOS on Apple Silicon. Select --profile cpu or editor."
    if profile.startswith("nvidia") and (system not in ("Linux", "Windows") or machine not in ("x86_64", "amd64")):
        return "The bundled NVIDIA profiles target Linux/Windows x86-64. Choose CPU or install a vendor-supported runtime manually."
    if system == "Darwin" and machine not in ("arm64", "aarch64"):
        return "The pinned PyTorch runtime has no Intel macOS wheel. Use --profile editor on this machine."
    if system not in ("Linux", "Windows", "Darwin") or machine not in ("x86_64", "amd64", "arm64", "aarch64"):
        return "No bundled model installation profile for this operating system/architecture. Use --profile editor."
    if system == "Windows" and machine not in ("amd64", "x86_64"):
        return "The Windows model profiles currently target x86-64. Use --profile editor."
    if profile.startswith("nvidia"):
        gpus = compatible_gpus(profile, hardware["gpus"])
        if not gpus:
            return f"No supported GPU detected for {profile}. Check the NVIDIA driver and nvidia-smi, or explicitly select --profile cpu."
    return None


def compatible_gpus(profile, gpus):
    if profile == "nvidia-legacy":
        return [gpu for gpu in gpus if 5.0 <= gpu["capability"] < 10.0]
    if profile == "nvidia":
        return [gpu for gpu in gpus if gpu["capability"] >= 7.5]
    return []


def install_commands(root, profile, quantize=False):
    python = str(app_python(root))
    if profile == "editor":
        return []
    # Apple CPU and Metal use the macOS wheel from PyPI, not the Linux CPU index.
    index = INDEXES.get(profile) if platform.system() != "Darwin" else None
    version = TORCH_VERSION + (f"+{index}" if index else "")
    torch = [python, "-m", "pip", "install", f"torch=={version}"]
    if index:
        torch += ["--index-url", f"https://download.pytorch.org/whl/{index}"]
    commands = [torch, [python, "-m", "pip", "install", "-r", str(Path(root) / "backend/requirements-models.txt")]]
    if quantize:
        commands.append([python, "-m", "pip", "install", f"bitsandbytes=={BITSANDBYTES_VERSION}"])
    return commands


def shell_command(command):
    if os.name == "nt":
        # PowerShell requires & when the executable path is quoted.
        return "& " + " ".join("'" + str(part).replace("'", "''") + "'" for part in command)
    return shlex.join([str(part) for part in command])


def runtime_probe(spec):
    """Executed ONLY in the target environment as a short-lived subprocess."""
    result = {"python": sys.executable, "python_version": list(sys.version_info[:3]), "packages": {}}
    for name in spec["packages"]:
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    if spec["profile"] == "editor":
        return result
    try:
        import torch
        result["torch"] = {"version": str(torch.__version__), "cuda": torch.version.cuda}
        if hasattr(torch, "_native"):
            torch._native.registry.deregister_op_overrides(disable_dispatch_keys="CUDA")
    except Exception as exc:
        result["runtime_error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["imports"] = {}
    for name in ("transformers", "accelerate", "huggingface_hub", "safetensors"):
        try:
            if name == "transformers":
                from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            else:
                __import__(name)
            result["imports"][name] = None
        except Exception as exc:
            result["imports"][name] = f"{type(exc).__name__}: {exc}"
    devices = []
    if spec["profile"].startswith("nvidia"):
        try:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append({"device": f"cuda:{index}", "name": props.name,
                                "capability": props.major + props.minor / 10,
                                "vram_gb": round(props.total_memory / 2**30, 1)})
        except Exception as exc:
            result["runtime_error"] = f"GPU discovery failed: {exc}"
    elif spec["profile"] == "metal":
        if torch.backends.mps.is_available():
            devices = [{"device": "mps", "name": "Apple Metal"}]
    else:
        devices = [{"device": "cpu", "name": "CPU"}]
    if spec["profile"].startswith("nvidia"):
        devices = compatible_gpus(spec["profile"], devices)
    result["devices"] = []
    for device in devices:
        try:
            target = device["device"]
            dtype = torch.float32 if target == "cpu" else torch.float16
            with torch.inference_mode():
                x = torch.ones((32, 32), device=target, dtype=dtype)
                y = torch.softmax(x @ x, dim=-1)
                if not torch.isfinite(y).all().item() or abs(y[0].sum().item() - 1) > .01:
                    raise RuntimeError("Tensor calculation returned an invalid result")
                if spec["quantize"]:
                    from bitsandbytes.functional import quantize_4bit, dequantize_4bit
                    packed, state = quantize_4bit(x, quant_type="nf4")
                    restored = dequantize_4bit(packed, state)
                    if not torch.isfinite(restored).all().item():
                        raise RuntimeError("NF4 check returned an invalid result")
            device["ok"] = True
        except Exception as exc:
            device.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        result["devices"].append(device)
    return result


def probe_environment(python, spec):
    if not Path(python).exists():
        return {"probe_error": "Splash's .venv does not exist. Run python scripts/setup.py first.", "packages": {}}
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    try:
        result = subprocess.run([str(python), str(Path(__file__).resolve()), json.dumps(spec)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=45)
        if result.returncode:
            raise ValueError((result.stderr.strip() or result.stdout.strip())[-2000:] or f"exit status {result.returncode}")
        # Third-party imports may print diagnostics to stdout before the sentinel.
        payload = result.stdout.rsplit("SPLASH_PROBE=", 1)[-1].strip()
        return json.loads(payload)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"probe_error": f"Could not check {python}: {exc}", "packages": {}}


def node_checks():
    checks = []
    for name in ("node", "npm"):
        executable = shutil.which(name)
        try:
            result = run([executable, "--version"]) if executable else None
            version = result.stdout.strip() if result and result.returncode == 0 else ""
            numbers = tuple(int(v) for v in re.findall(r"\d+", version)[:3])
            ok = bool(numbers) and (name != "node" or numbers >= (22, 12, 0))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            version, ok = "", False
        checks.append({"name": name, "ok": ok, "detail": version or "not available",
                       "fix": "Install Node.js 22.12 or later with npm."})
    return checks


def make_report(root, hardware, profile, level, quantize, probe, frontend):
    app = requirements(Path(root) / "backend/requirements.txt")
    model = requirements(Path(root) / "backend/requirements-models.txt") if profile != "editor" else {}
    if profile != "editor":
        model["torch"] = TORCH_VERSION
    if quantize:
        model["bitsandbytes"] = BITSANDBYTES_VERSION
    checks = list(frontend)
    def check(name, ok, detail, group="model", fix=""):
        checks.append(dict(name=name, ok=bool(ok), detail=detail, group=group, fix=fix))
    for entry in checks:
        entry["group"] = "app"
    version = probe.get("python_version", [])
    check("App Python", bool(version) and tuple(version) >= (3, 11),
          probe.get("probe_error") or f"{probe.get('python')} · {'.'.join(map(str, version))}", "app",
          "Use Python 3.11 or later and run scripts/setup.py to create Splash's .venv.")
    for name, wanted in (app | model).items():
        installed = probe.get("packages", {}).get(name)
        # PyTorch distribution metadata may include the CUDA/CPU local suffix.
        matches = installed and installed.split("+", 1)[0] == wanted
        check(name, matches, f"{installed or 'missing'} (requires {wanted})", "app" if name in app else "model")
    problem = profile_problem(profile, hardware, quantize)
    if problem:
        check("Hardware profile", False, problem)
    if profile != "editor":
        runtime = probe.get("torch", {})
        cuda = runtime.get("cuda")
        expected_cuda = {"nvidia-legacy": "12.6", "nvidia": "13.0"}.get(profile)
        build_ok = runtime.get("version", "").split("+", 1)[0] == TORCH_VERSION and (profile == "cpu" or cuda == expected_cuda)
        check("PyTorch build", build_ok, probe.get("runtime_error") or
              f"{runtime.get('version', 'not importable')} · CUDA {cuda or 'none'}; selected profile: {profile}")
        for name, error in probe.get("imports", {}).items():
            check(f"Import {name}", error is None, error or "passed")
        devices = probe.get("devices", [])
        check("Device execution", any(d.get("ok") for d in devices),
              "; ".join(f"{d['device']} {d['name']}: {'passed' if d.get('ok') else d.get('error')}" for d in devices)
              or "No usable device. Check the runtime/driver; no automatic CPU substitution.")
    notes = []
    if hardware.get("gpu_note"):
        notes.append(hardware["gpu_note"])
    if profile != "editor":
        model_info = LEVELS[level]
        needed_disk = model_info["download_gb"] + 6
        notes.append(f"{level}: {model_info['name']}; checkpoint ~{model_info['download_gb']} GB. Setup does not download it.")
        if hardware["disk_free_gb"] < needed_disk:
            notes.append(f"LOW DISK: allow at least ~{needed_disk} GB for this checkpoint and packages, plus download/cache overhead.")
        ram = hardware.get("ram_gb")
        if ram is not None and ram < model_info["ram_gb"]:
            notes.append(f"LOW RAM: this level has a conservative {model_info['ram_gb']} GB RAM target; detected {ram} GB.")
        if profile.startswith("nvidia"):
            gpu_list = compatible_gpus(profile, hardware["gpus"])
            needed_vram = model_info.get("nf4_gb", model_info["vram_gb"]) if quantize else model_info["vram_gb"]
            best = max((g["free_gb"] for g in gpu_list), default=0)
            if best < needed_vram:
                notes.append(f"LOW VRAM: allow ~{needed_vram} GB free for {level}{' NF4' if quantize else ''}; detected {best} GB. Choose a smaller level or consider NF4 on a supported GPU.")
        notes.append("Resource figures are estimates. Passing a tensor check does not qualify model quality, latency, or maximum context.")
        notes.append("GTX 1050 and Metal model inference remain unverified; use Load & run check with an actual checkpoint.")
    app_ok = all(c["ok"] for c in checks if c["group"] == "app")
    model_ok = all(c["ok"] for c in checks if c["group"] == "model")
    return {"profile": profile, "level": level, "quantize": quantize, "hardware": hardware,
            "python": str(app_python(root)), "checks": checks, "notes": notes,
            "app_ready": app_ok, "model_ready": model_ok if profile != "editor" else None,
            "ok": app_ok and model_ok, "install_commands": install_commands(root, profile, quantize) if not problem else []}


def check_dependencies(root, requested="auto", level="compact", quantize=False):
    hardware = inspect_hardware(root)
    profile = resolve_profile(requested, hardware)
    packages = requirements(Path(root) / "backend/requirements.txt")
    if profile != "editor":
        packages.update(requirements(Path(root) / "backend/requirements-models.txt"))
        packages.update(torch=TORCH_VERSION)
    if quantize:
        packages["bitsandbytes"] = BITSANDBYTES_VERSION
    probe = probe_environment(app_python(root), {"profile": profile, "quantize": quantize, "packages": list(packages)})
    return make_report(root, hardware, profile, level, quantize, probe, node_checks())


def print_report(report):
    hw = report["hardware"]
    print(f"\nSplash dependency check · {report['profile']} · {report['level']}")
    print(f"{hw['os']} {hw['machine']} · {hw['ram_gb'] or 'unknown'} GB RAM · {hw['disk_free_gb']} GB free disk")
    print(f"App interpreter: {report['python']}")
    for gpu in hw["gpus"]:
        print(f"GPU: {gpu['name']} · capability {gpu['capability']} · {gpu['free_gb']}/{gpu['vram_gb']} GB free · driver {gpu['driver']}")
    for check in report["checks"]:
        print(f"  {'OK' if check['ok'] else 'FAIL':4} {check['name']}: {check['detail']}")
        if not check["ok"] and check.get("fix"):
            print(f"       {check['fix']}")
    for note in report["notes"]:
        print(f"  NOTE {note}")
    if not report["app_ready"]:
        print("App dependencies need attention. Run python scripts/setup.py and resolve the FAIL items above.")
    if report["model_ready"] is False:
        print("Model enrichment is unavailable until the model FAIL items are resolved. Matching install commands:")
        for command in report["install_commands"]:
            print("  " + shell_command(command))
    print("App dependencies: " + ("ready" if report["app_ready"] else "incomplete"))
    if report["profile"] != "editor":
        print("Model runtime: " + ("passed dependency/device checks; load a checkpoint next" if report["model_ready"] else "needs attention"))


if __name__ == "__main__":
    print("SPLASH_PROBE=" + json.dumps(runtime_probe(json.loads(sys.argv[1]))))
