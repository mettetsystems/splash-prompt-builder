"""Assemble a double-clickable Splash.app. Run this on macOS."""
import argparse
import hashlib
import inspect
import os
import plistlib
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


APP_VERSION = "2.0.0"
BUNDLE_ID = "local.splash.prompt-studio"
PYTHON_RELEASE = "20261009"
PYTHON_ASSETS = {
    "arm64": {
        "name": "cpython-3.12.15+20261009-aarch64-apple-darwin-install_only.tar.gz",
        "sha256": "83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8",
    },
    "x86_64": {
        "name": "cpython-3.12.15+20261009-x86_64-apple-darwin-install_only.tar.gz",
        "sha256": "4c84ef5abe4bc145646175fce9b881936d60126b9a3f9a24c6308e2d035ca5c6",
    },
}


def repo_root():
    return Path(__file__).resolve().parents[2]


def host_arch():
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    if machine == "x86_64":
        return "x86_64"
    raise SystemExit(f"Unsupported Mac architecture: {machine}")


def asset_url(name):
    from urllib.parse import quote
    return (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{PYTHON_RELEASE}/{quote(name)}"
    )


def info_plist(minimum="13.0", include_icon=True):
    payload = {
        "CFBundleName": "Splash",
        "CFBundleDisplayName": "Splash",
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": APP_VERSION,
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundlePackageType": "APPL",
        "CFBundleExecutable": "Splash",
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": minimum,
        "LSMultipleInstancesProhibited": True,
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.developer-tools",
        "NSPrincipalClass": "NSApplication",
    }
    if include_icon:
        payload["CFBundleIconFile"] = "splash"
    return payload


def stage_application(repo, destination):
    index = repo / "frontend" / "dist" / "index.html"
    if not index.is_file():
        raise SystemExit(
            "Frontend build is missing. Install Node.js 22.12 or later and rerun, "
            "or build frontend/dist before packaging."
        )
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copytree(
        repo / "backend",
        destination / "backend",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    (destination / "frontend").mkdir()
    shutil.copytree(repo / "frontend" / "dist", destination / "frontend" / "dist")


def download(url, destination, expected_sha256):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha256:
        print(f"Using cached {destination.name}")
        return
    print(f"Downloading {url}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Checksum mismatch for {destination.name}.")
    temporary.replace(destination)


def safe_extract(archive, destination):
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise SystemExit(f"Refusing to extract {member.name}")
        kwargs = {}
        if "filter" in inspect.signature(tar.extractall).parameters:
            kwargs["filter"] = "data"
        tar.extractall(destination, **kwargs)


def bundled_python_env():
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def install_runtime(python, repo, with_torch):
    env = bundled_python_env()
    requirements = [str(repo / "backend" / "requirements.txt")]
    if with_torch:
        requirements.append(str(repo / "backend" / "requirements-macos.txt"))
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True, env=env)
    subprocess.run([str(python), "-m", "pip", "install", *[item for path in requirements for item in ("-r", path)]],
                   check=True, env=env)


def ensure_frontend(repo, skip_build):
    index = repo / "frontend" / "dist" / "index.html"
    if skip_build:
        if not index.is_file():
            raise SystemExit("frontend/dist is missing and --skip-frontend-build was set.")
        return
    npm = shutil.which("npm")
    if not npm:
        if index.is_file():
            print("npm was not found; using the existing frontend/dist.")
            return
        raise SystemExit("Install Node.js 22.12 or later with npm, then retry.")
    subprocess.run([npm, "--prefix", str(repo / "frontend"), "ci"], check=True)
    subprocess.run([npm, "--prefix", str(repo / "frontend"), "run", "build"], check=True)


def compile_launcher(source, output):
    compiler = shutil.which("clang") or shutil.which("cc")
    if not compiler:
        raise SystemExit("clang is required. Install Xcode Command Line Tools: xcode-select --install")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [compiler, "-fobjc-arc", "-mmacosx-version-min=13.0", "-framework", "Cocoa", "-o", str(output), str(source)],
        check=True,
    )
    output.chmod(output.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_icon(logo, destination):
    if sys.platform != "darwin" or not logo.is_file() or not shutil.which("sips") or not shutil.which("iconutil"):
        return False
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        png = root / "splash.png"
        iconset = root / "splash.iconset"
        iconset.mkdir()
        subprocess.run(["sips", "-s", "format", "png", str(logo), "--out", str(png)],
                       check=True, stdout=subprocess.DEVNULL)
        sizes = {
            "icon_16x16.png": 16, "icon_16x16@2x.png": 32,
            "icon_32x32.png": 32, "icon_32x32@2x.png": 64,
            "icon_128x128.png": 128, "icon_128x128@2x.png": 256,
            "icon_256x256.png": 256, "icon_256x256@2x.png": 512,
            "icon_512x512.png": 512, "icon_512x512@2x.png": 1024,
        }
        for name, size in sizes.items():
            subprocess.run(["sips", "-z", str(size), str(size), str(png), "--out", str(iconset / name)],
                           check=True, stdout=subprocess.DEVNULL)
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(destination)], check=True)
    return destination.is_file()


def sign_app(app):
    if shutil.which("xattr"):
        subprocess.run(["xattr", "-cr", str(app)], check=False)
    if not shutil.which("codesign"):
        raise SystemExit("codesign is required so macOS will launch the app.")
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)


def make_dmg(app, dmg):
    if dmg.exists():
        dmg.unlink()
    subprocess.run(
        ["hdiutil", "create", "-volname", "Splash", "-srcfolder", str(app), "-ov", "-format", "UDZO", str(dmg)],
        check=True,
    )


def require_macos():
    if sys.platform == "darwin":
        return
    raise SystemExit(
        "Splash.app must be built on macOS. Copy this repository to a Mac and run: python3 scripts/package_macos.py"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmg", action="store_true", help="Also write dist/Splash-macos.dmg")
    parser.add_argument("--no-torch", action="store_true", help="Omit PyTorch and the model adapters")
    parser.add_argument("--skip-frontend-build", action="store_true", help="Use the existing frontend/dist")
    parser.add_argument("--output", type=Path, help="App bundle path (default: dist/Splash.app)")
    args = parser.parse_args(argv)
    require_macos()
    arch = host_arch()
    with_torch = not args.no_torch and arch == "arm64"
    if not args.no_torch and arch != "arm64":
        print("PyTorch 2.14.1 publishes an Apple Silicon wheel only. This Intel build can edit and store projects.")
        print("Enrichment needs an Apple Silicon Mac, or a torch build you install yourself.")
    repo = repo_root()
    output = (args.output or (repo / "dist" / "Splash.app")).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    ensure_frontend(repo, args.skip_frontend_build)
    contents = output / "Contents"
    resources = contents / "Resources"
    macos = contents / "MacOS"
    resources.mkdir(parents=True)
    asset = PYTHON_ASSETS[arch]
    archive = repo / "dist" / "cache" / asset["name"]
    download(asset_url(asset["name"]), archive, asset["sha256"])
    safe_extract(archive, resources)
    python = resources / "python" / "bin" / "python3"
    if not python.is_file():
        raise SystemExit("The CPython archive did not contain python/bin/python3.")
    python.chmod(python.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    install_runtime(python, repo, with_torch)
    stage_application(repo, resources / "app")
    shutil.copy2(repo / "deploy" / "macos" / "launch_splash.py", resources / "launch_splash.py")
    icon = resources / "splash.icns"
    include_icon = False
    try:
        include_icon = build_icon(repo / "frontend" / "src" / "assets" / "splash-logo.jpg", icon)
    except subprocess.CalledProcessError as exc:
        print(f"Continuing without an app icon: {exc}")
    plist_path = contents / "Info.plist"
    minimum = "14.0" if with_torch else "13.0"
    with plist_path.open("wb") as handle:
        plistlib.dump(info_plist(minimum, include_icon), handle)
    compile_launcher(repo / "deploy" / "macos" / "launcher.m", macos / "Splash")
    sign_app(output)
    print(f"Built {output}")
    if args.dmg:
        dmg = output.parent / "Splash-macos.dmg"
        make_dmg(output, dmg)
        print(f"Built {dmg}")
    print("Open the app, or copy it to /Applications. Quit from the Splash menu.")
    print("Projects: ~/Library/Application Support/Splash")
    print("Log: ~/Library/Logs/Splash/splash.log")
    print("If macOS blocks the first launch, right-click Splash.app and choose Open.")
    if with_torch:
        print("This build includes PyTorch. Metal stays unverified until Load & run check passes; CPU is the fallback.")
        print("Download a checkpoint from Model & hardware. Checkpoints are not inside the app.")
    else:
        print("This build does not include PyTorch. Projects, editing, and history still work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
