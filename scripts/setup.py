"""Set up Splash and check dependencies for the chosen hardware and model level."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import venv

if __package__:
    from . import dependencies as deps
else:
    import dependencies as deps


def main(argv=None, root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="Include test dependencies")
    parser.add_argument("--profile", choices=deps.PROFILES, default="auto",
                        help="Hardware runtime; auto detects NVIDIA, Apple Silicon, or CPU")
    parser.add_argument("--level", choices=deps.LEVELS, default="compact",
                        help="Model level for RAM/VRAM/disk guidance (default: compact)")
    parser.add_argument("--quantize", action="store_true", help="Include CUDA NF4 dependencies and execution check")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Check only; do not install or change anything")
    modes.add_argument("--install-models", action="store_true", help="Install the selected hardware's pinned model runtime")
    parser.add_argument("--json", action="store_true", help="Machine-readable report; requires --check")
    args = parser.parse_args(argv)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or later is required.")
    if args.json and not args.check:
        parser.error("--json requires --check.")
    if args.profile == "editor" and (args.install_models or args.quantize):
        parser.error("The editor profile does not install models or use NF4. Select a model profile.")
    if args.check:
        report = deps.check_dependencies(root, args.profile, args.level, args.quantize)
        print(json.dumps(report, indent=2)) if args.json else deps.print_report(report)
        return 0 if report["ok"] else 1
    try:
        for check in deps.node_checks():
            if not check["ok"]:
                print(f"{check['name']}: {check['detail']}. {check['fix']}", file=sys.stderr)
                return 1
        hardware = deps.inspect_hardware(root)
        profile = deps.resolve_profile(args.profile, hardware)
        problem = deps.profile_problem(profile, hardware, args.quantize)
        if problem and (args.install_models or args.quantize):
            print(problem, file=sys.stderr)
            return 1
        python = deps.app_python(root)
        if not python.exists():
            venv.create(root / ".venv", with_pip=True)
        interpreter = deps.probe_environment(python, {"profile": "editor", "quantize": False, "packages": []})
        if tuple(interpreter.get("python_version", [])) < (3, 11):
            print(interpreter.get("probe_error") or "Splash's existing .venv uses Python older than 3.11. Recreate it with a supported interpreter.", file=sys.stderr)
            return 1
        requirements = "requirements-dev.txt" if args.dev else "requirements.txt"
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(root / "backend" / requirements)], check=True)
        npm = shutil.which("npm")
        subprocess.run([npm, "--prefix", str(root / "frontend"), "ci"], check=True)
        subprocess.run([npm, "--prefix", str(root / "frontend"), "run", "build"], check=True)
        report = deps.check_dependencies(root, profile, args.level, args.quantize)
        if args.install_models:
            print(f"\nInstalling {profile} model dependencies into {python}. Checkpoints are downloaded separately.", flush=True)
            torch_ready = all(c["ok"] for c in report["checks"] if c["name"] in ("torch", "PyTorch build"))
            for index, command in enumerate(report["install_commands"]):
                if index == 0 and torch_ready:
                    print("Keeping the compatible PyTorch installation.", flush=True)
                    continue
                subprocess.run(command, check=True)
            report = deps.check_dependencies(root, profile, args.level, args.quantize)
        deps.print_report(report)
        if report["app_ready"]:
            print("\nSplash editor is ready. Run: python scripts/start.py")
        if report["model_ready"] is False and not args.install_models:
            command = [sys.executable, str(root / "scripts/setup.py"), "--profile", profile,
                       "--level", args.level, "--install-models"] + (["--quantize"] if args.quantize else [])
            print("To install the matching model runtime:\n  " + deps.shell_command(command))
        return 0 if report["app_ready"] and (not args.install_models or report["model_ready"]) else 1
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Setup could not complete: {exc}\nResolve the error above, then rerun the same setup command. Projects and checkpoints are preserved.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nSetup interrupted. Rerun the command to resume package installation.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
