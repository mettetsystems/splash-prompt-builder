"""Set up Splash, including PyTorch, for the chosen hardware and model level."""
import argparse
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import venv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.endpoints import discover_endpoints, endpoint_config, normalize_url

if __package__:
    from . import dependencies as deps
else:
    import dependencies as deps


def choose_endpoint(found, args):
    """No model-list probe or selection generates text or loads weights."""
    choices = [(server, model) for server in found for model in server['models']]
    for server in found:
        print(f"Model server found: {server['base_url']} ({server['provider']})")
        if server.get('notice'):
            print(server['notice'])
    if args.local_model or not found:
        return None, False
    if args.endpoint:
        matching = [(s, m) for s, m in choices if not args.endpoint_model or m['id'] == args.endpoint_model]
        if len(matching) == 1:
            s, m = matching[0]
            return endpoint_config(s, m['id'], args.endpoint_context), True
        if args.endpoint_model:
            raise ValueError('The requested model is not listed by the endpoint.')
    if not sys.stdin.isatty():
        if args.endpoint:
            raise ValueError('Select a listed model with --endpoint-model for noninteractive setup.')
        print('No terminal input is available. Installing the editor only; choose an endpoint in Model & hardware, or rerun with --local-model.')
        return None, True
    if not choices:
        print('No usable models are listed. Installing the editor only; start a model in the existing server or configure authentication, then rescan in Model & hardware.')
        return None, True
    for i, (server, model) in enumerate(choices, 1):
        print(f"  {i}. {model['id']} at {server['base_url']}")
    while True:
        try:
            answer = input('Use an existing model endpoint? Enter its number [1], L to install a separate local runtime, or E for editor only: ').strip().lower()
        except EOFError:
            return None, True
        if answer == 'l':
            return None, False
        if answer == 'e':
            return None, True
        try:
            index = int(answer or '1') - 1
            if 0 <= index < len(choices):
                server, model = choices[index]
                return endpoint_config(server, model['id'], args.endpoint_context), True
        except ValueError:
            pass
        print('Choose one of the displayed numbers, L, or E.')


def main(argv=None, root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="Include test dependencies")
    parser.add_argument("--profile", choices=deps.PROFILES, default="auto",
                        help="Hardware runtime; auto detects NVIDIA, Apple Silicon, or CPU; editor skips model dependencies")
    parser.add_argument("--level", choices=deps.LEVELS, default="compact",
                        help="Model level for RAM/VRAM/disk guidance (default: compact)")
    parser.add_argument("--quantize", action="store_true", help="Include CUDA NF4 dependencies and execution check")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Check only; do not install or change anything")
    modes.add_argument("--install-models", action="store_true",
                       help="Compatibility alias: model dependencies are installed by default (except --profile editor)")
    parser.add_argument("--json", action="store_true", help="Machine-readable report; requires --check")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--endpoint', help='Reuse an OpenAI-compatible loopback API URL (for example http://localhost:8000/v1)')
    selection.add_argument('--local-model', action='store_true', help='Install the local runtime even when another model server is found')
    parser.add_argument('--endpoint-model', help='Model ID served by --endpoint; required when it lists multiple models')
    parser.add_argument('--endpoint-context', type=int, default=4096, metavar='1024..32768')
    parser.add_argument('--data-dir', type=Path, help='Application data directory for the saved endpoint choice; use the same directory at startup')
    args = parser.parse_args(argv)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or later is required.")
    if args.json and not args.check:
        parser.error("--json requires --check.")
    if args.endpoint_model and not args.endpoint:
        parser.error('--endpoint-model requires --endpoint.')
    if not 1024 <= args.endpoint_context <= 32768:
        parser.error('--endpoint-context must be between 1024 and 32768.')
    if args.profile == "editor" and (args.install_models or args.quantize):
        parser.error("The editor profile does not install models or use NF4. Select a model profile.")
    try:
        found = discover_endpoints([normalize_url(args.endpoint)] if args.endpoint else None)
        if args.endpoint and not found:
            raise ValueError('No compatible model server responded at --endpoint. No local runtime was installed.')
        if args.check:
            check_profile = 'editor' if found and not args.local_model and (args.endpoint or args.profile == 'auto') else args.profile
            report = deps.check_dependencies(root, check_profile, args.level, args.quantize and check_profile != 'editor')
            report['endpoints'] = found
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                deps.print_report(report)
                for endpoint in found:
                    print(f"Existing endpoint: {endpoint['base_url']} · {len(endpoint['models'])} listed models")
            return 0 if report['ok'] else 1
        selected_endpoint, editor_only = choose_endpoint(found, args)
        for check in deps.node_checks():
            if not check["ok"]:
                print(f"{check['name']}: {check['detail']}. {check['fix']}", file=sys.stderr)
                return 1
        hardware = deps.inspect_hardware(root)
        profile = 'editor' if editor_only else deps.resolve_profile(args.profile, hardware)
        install_models = profile != "editor"
        quantize = args.quantize and not editor_only
        problem = deps.profile_problem(profile, hardware, quantize)
        if problem:
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
        report = deps.check_dependencies(root, profile, args.level, quantize)
        if install_models:
            print(f"\nInstalling {profile} model dependencies into {python}. Checkpoints are downloaded separately.", flush=True)
            torch_checks = {c["name"]: c["ok"] for c in report["checks"]}
            torch_ready = all(torch_checks.get(name, False) for name in ("torch", "PyTorch build"))
            for index, command in enumerate(report["install_commands"]):
                if index == 0 and torch_ready:
                    print("Keeping the compatible PyTorch installation.", flush=True)
                    continue
                subprocess.run(command, check=True)
            report = deps.check_dependencies(root, profile, args.level, quantize)
        deps.print_report(report)
        ready = report["app_ready"] and (not install_models or report["model_ready"])
        if ready:
            if selected_endpoint:
                env = dict(os.environ)
                if args.data_dir:
                    env['SPLASH_DATA_DIR'] = str(args.data_dir.expanduser().resolve())
                subprocess.run([str(python), '-m', 'backend.configure_endpoint'], cwd=root, env=env,
                               input=json.dumps(selected_endpoint), text=True, check=True)
                print(f"Using existing endpoint {selected_endpoint['base_url']} · {selected_endpoint['model']}. Splash will not load local weights.")
            label = "Splash and its model runtime are ready" if install_models else "Splash editor is ready (model dependencies skipped)"
            start = ['python', 'scripts/start.py']
            if args.data_dir:
                start += ['--data-dir', str(args.data_dir.expanduser().resolve())]
            command = subprocess.list2cmdline(start) if os.name == 'nt' else shlex.join(start)
            print(f"\n{label}. Run: {command}")
        elif report["app_ready"]:
            print("\nSetup is incomplete: the editor is installed, but the model runtime check failed. Resolve the FAIL items above and rerun setup.", file=sys.stderr)
        return 0 if ready else 1
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Setup could not complete: {exc}\nResolve the error above, then rerun the same setup command. Projects and checkpoints are preserved.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nSetup interrupted. Rerun the command to resume package installation.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
