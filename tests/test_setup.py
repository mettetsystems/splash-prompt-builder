"""Setup must inspect the app environment and never mistake a package for a working GPU."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import venv

from scripts import dependencies as deps, setup

ROOT = Path(__file__).resolve().parents[1]


def hardware(gpus=(), system="Linux", machine="x86_64"):
    return {"os": system, "machine": machine, "ram_gb": 64, "disk_free_gb": 100,
            "cpu_threads": 12, "gpus": list(gpus)}


def gpu(name="GTX 1050", capability=6.1, vram=2):
    return {"name": name, "capability": capability, "vram_gb": vram, "free_gb": vram, "driver": "580.82"}


def probe(cuda="12.6"):
    return {"python": str(deps.app_python(ROOT)), "python_version": [3, 14, 7],
            "packages": deps.requirements(ROOT / "backend/requirements.txt") |
                        deps.requirements(ROOT / "backend/requirements-models.txt") |
                        {"torch": deps.TORCH_VERSION, "bitsandbytes": deps.BITSANDBYTES_VERSION},
            "torch": {"version": deps.TORCH_VERSION, "cuda": cuda},
            "imports": {"transformers": None, "accelerate": None, "huggingface_hub": None, "safetensors": None},
            "devices": [{"device": "cuda:0", "name": "GTX 1050", "ok": True}]}


def report(profile="nvidia-legacy", level="compact", quantize=False, state=None, hw=None):
    return deps.make_report(ROOT, hw or hardware([gpu()]), profile, level, quantize,
                            probe() if state is None else state,
                            [{"name": "node", "ok": True, "detail": "v22.12.0"},
                             {"name": "npm", "ok": True, "detail": "10.0.0"}])


class ProfileTests(unittest.TestCase):
    def test_1050_2070_and_5090_choose_appropriate_wheels(self):
        for card, expected in [(gpu(), "nvidia-legacy"), (gpu("RTX 2070", 7.5, 8), "nvidia"),
                               (gpu("RTX 5090", 12.0, 32), "nvidia")]:
            with self.subTest(card=card["name"]):
                self.assertEqual(deps.resolve_profile("auto", hardware([card])), expected)
        self.assertEqual(deps.resolve_profile("auto", hardware([gpu(), gpu("RTX 5090", 12.0, 32)])), "nvidia")

    def test_cpu_and_apple_silicon(self):
        self.assertEqual(deps.resolve_profile("auto", hardware()), "cpu")
        self.assertEqual(deps.resolve_profile("auto", hardware(system="Darwin", machine="arm64")), "metal")
        self.assertIsNotNone(deps.profile_problem("metal", hardware()))
        self.assertIsNotNone(deps.profile_problem("cpu", hardware(system="Darwin")))

    def test_explicit_gpu_profile_never_silently_falls_back(self):
        self.assertEqual(deps.resolve_profile("nvidia-legacy", hardware()), "nvidia-legacy")
        self.assertIsNotNone(deps.profile_problem("nvidia-legacy", hardware()))
        self.assertIsNotNone(deps.profile_problem("nvidia", hardware([gpu()])))
        self.assertIsNotNone(deps.profile_problem("nvidia-legacy", hardware([gpu("RTX 5090", 12.0, 32)])))
        self.assertIsNotNone(deps.profile_problem("cpu", hardware(), quantize=True))

    def test_install_commands_use_app_python_pinned_wheels_and_optional_nf4(self):
        root = Path("/tmp/Splash with spaces")
        with patch.object(deps.platform, "system", return_value="Linux"):
            commands = deps.install_commands(root, "nvidia-legacy", quantize=True)
            self.assertEqual(commands[0][0], str(deps.app_python(root)))
            self.assertIn("torch==2.14.1+cu126", commands[0])
            self.assertEqual(commands[0][-1], "https://download.pytorch.org/whl/cu126")
            self.assertIn("bitsandbytes==0.50.2", commands[-1])
            self.assertEqual(len(deps.install_commands(root, "nvidia")), 2)
        with patch.object(deps.platform, "system", return_value="Darwin"):
            self.assertNotIn("--index-url", deps.install_commands(root, "cpu")[0])
            self.assertNotIn("--index-url", deps.install_commands(root, "metal")[0])

    def test_node_minimum_is_checked_not_just_presence_of_npm(self):
        for version, expected in [("v22.11.0", False), ("v22.12.0", True), ("v24.0.0", True)]:
            with patch.object(deps.shutil, "which", side_effect=lambda name: name), patch.object(deps, "run") as run:
                run.side_effect = [subprocess.CompletedProcess([], 0, version, ""),
                                   subprocess.CompletedProcess([], 0, "10.0.0", "")]
                self.assertEqual(deps.node_checks()[0]["ok"], expected)


class DependencyReportTests(unittest.TestCase):
    def test_missing_torch_blocks_models_but_editor_remains_ready(self):
        state = probe()
        state["packages"].pop("torch")
        state.pop("torch")
        state["devices"] = []
        state["runtime_error"] = "ModuleNotFoundError: No module named 'torch'"
        result = report(state=state)
        self.assertTrue(result["app_ready"])
        self.assertFalse(result["model_ready"])
        self.assertFalse(result["ok"])
        self.assertIn("cu126", result["install_commands"][0][-1])
        self.assertTrue(report("editor", state=state)["ok"])

    def test_wrong_cuda_build_fails_even_when_torch_imports(self):
        result = report(state=probe(cuda="13.0"))
        self.assertFalse(result["model_ready"])
        self.assertFalse(next(c for c in result["checks"] if c["name"] == "PyTorch build")["ok"])

    def test_gpu_kernel_failure_is_not_a_successful_dependency_check(self):
        state = probe()
        state["devices"][0].update(ok=False, error="no kernel image is available")
        self.assertFalse(report(state=state)["model_ready"])
        state["devices"] = []
        self.assertFalse(report(state=state)["model_ready"])

    def test_broken_transformers_import_and_missing_nf4_fail(self):
        state = probe()
        state["imports"]["transformers"] = "ImportError: incompatible shared library"
        self.assertFalse(report(state=state)["model_ready"])
        state = probe()
        state["packages"].pop("bitsandbytes")
        self.assertTrue(report(state=state)["model_ready"])
        self.assertFalse(report(quantize=True, state=state)["model_ready"])

    def test_capacity_guidance_is_separate_from_runtime_success(self):
        result = report(level="extended")
        self.assertTrue(result["model_ready"])
        self.assertTrue(any("LOW VRAM" in note for note in result["notes"]))
        eight_gb = hardware([gpu("RTX 2070", 7.5, 8)])
        result = report(level="balanced", quantize=True, hw=eight_gb)
        self.assertFalse(any("LOW VRAM" in note for note in result["notes"]))
        self.assertTrue(any("does not qualify model" in note for note in result["notes"]))

    def test_unsupported_platform_has_no_install_command(self):
        result = report("metal")
        self.assertFalse(result["ok"])
        self.assertEqual(result["install_commands"], [])

    def test_cpu_can_use_an_existing_cuda_wheel_without_replacement(self):
        state = probe(cuda="13.0")
        state["devices"] = [{"device": "cpu", "name": "CPU", "ok": True}]
        self.assertTrue(report("cpu", state=state)["model_ready"])

    def test_probe_uses_isolated_target_environment_not_system_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            venv.create(directory, with_pip=False)
            python = Path(directory) / ("Scripts/python.exe" if deps.os.name == "nt" else "bin/python")
            result = deps.probe_environment(python, {"profile": "cpu", "quantize": False, "packages": ["torch"]})
            self.assertIsNone(result["packages"]["torch"])
            self.assertIn("ModuleNotFoundError", result["runtime_error"])
            self.assertEqual(Path(result["python"]), python)

    def test_probe_timeout_is_reported_without_hanging_setup(self):
        with patch.object(deps.subprocess, "run", side_effect=subprocess.TimeoutExpired("probe", 45)):
            result = deps.probe_environment(Path(deps.sys.executable), {"profile": "cpu", "quantize": False, "packages": []})
        self.assertIn("Could not check", result["probe_error"])


class SetupWorkflowTests(unittest.TestCase):
    def test_check_json_has_failure_exit_code_and_never_installs(self):
        state = probe()
        state["packages"].pop("torch")
        with patch.object(setup.deps, "check_dependencies", return_value=report(state=state)), \
             patch.object(setup.venv, "create") as create, patch.object(setup.subprocess, "run") as install, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code = setup.main(["--check", "--json"], ROOT)
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        create.assert_not_called()
        install.assert_not_called()

    def test_install_rejects_incompatible_profile_before_mutations(self):
        with patch.object(setup.deps, "inspect_hardware", return_value=hardware([gpu()])), \
             patch.object(setup.deps, "node_checks", return_value=[]), \
             patch.object(setup.venv, "create") as create, patch.object(setup.subprocess, "run") as install, \
             contextlib.redirect_stderr(io.StringIO()):
            code = setup.main(["--profile", "nvidia", "--install-models"], ROOT)
        self.assertEqual(code, 1)
        create.assert_not_called()
        install.assert_not_called()

    def test_installer_keeps_healthy_torch_and_rechecks_after_install(self):
        good = report()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = deps.app_python(root)
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.object(setup.deps, "inspect_hardware", return_value=hardware([gpu()])), \
                 patch.object(setup.deps, "node_checks", return_value=[]), \
                 patch.object(setup.deps, "probe_environment", return_value=probe()), \
                 patch.object(setup.deps, "check_dependencies", side_effect=[copy.deepcopy(good), good]) as check, \
                 patch.object(setup.subprocess, "run") as install, contextlib.redirect_stdout(io.StringIO()):
                code = setup.main(["--install-models"], root)
        self.assertEqual(code, 0)
        self.assertEqual(check.call_count, 2)
        self.assertFalse(any(any(str(arg).startswith("torch==") for arg in c.args[0]) for c in install.call_args_list))


if __name__ == "__main__":
    unittest.main()
