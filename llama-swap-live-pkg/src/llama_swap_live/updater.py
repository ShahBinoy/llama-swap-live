"""Download the latest llama-swap binary from GitHub Releases."""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .colors import info, ok, warn, die, step, cyan


def _detect_platform() -> str:
    system  = platform.system().lower()
    machine = platform.machine().lower()
    os_map   = {"linux": "linux", "darwin": "darwin", "freebsd": "freebsd"}
    arch_map = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
    return f"{os_map.get(system, system)}_{arch_map.get(machine, machine)}"


def _fetch_latest() -> dict:
    url = "https://api.github.com/repos/mostlygeek/llama-swap/releases/latest"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "llama-swap-live"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def _download_with_progress(url: str, dest: Path) -> None:
    is_tty = sys.stdout.isatty()
    with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done  = 0
        while chunk := r.read(65536):
            f.write(chunk)
            done += len(chunk)
            if total and is_tty:
                pct = done * 100 // total
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r    [{bar}] {pct}%", end="", flush=True)
    if is_tty:
        print()


def _current_version(binary: Path) -> str:
    if not binary.exists():
        return "not installed"
    try:
        out = subprocess.check_output(
            [str(binary), "--version"], text=True, stderr=subprocess.STDOUT
        )
        m = re.search(r"version:\s*(\S+)", out)
        return m.group(1) if m else out.strip().split()[0]
    except Exception:
        return "unknown"


def update(install_dir: Path, check_only: bool = False) -> bool:
    """
    Fetch latest release, download binary if needed, install to install_dir.
    Returns True if a new version was installed.
    """
    binary = install_dir / "llama-swap"
    plat   = _detect_platform()

    step("Checking latest llama-swap release on GitHub…")
    try:
        release = _fetch_latest()
    except Exception as e:
        die(f"GitHub API error: {e}")

    tag     = release["tag_name"]
    version = tag.lstrip("v")
    current = _current_version(binary)

    print(f"    Installed : {cyan(current)}")
    print(f"    Latest    : {cyan(version)}  ({tag})")
    print(f"    Platform  : {cyan(plat)}")

    up_to_date = current in (version, tag)

    if up_to_date:
        ok("Already up to date")
        return False

    if check_only:
        warn(f"Update available: {current} → {version}")
        print("  Run without --check to install.")
        return False

    # Find matching asset
    assets = release.get("assets", [])
    asset  = next(
        (a for a in assets if plat in a["name"] and a["name"].endswith(".tar.gz")),
        None,
    )
    if not asset:
        # Fallback: any asset with the platform string
        asset = next((a for a in assets if plat in a["name"]), None)
    if not asset:
        names = [a["name"] for a in assets]
        die(f"No asset found for '{plat}'. Available:\n  " + "\n  ".join(names))

    step(f"Downloading {asset['name']}…")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / asset["name"]
        _download_with_progress(asset["browser_download_url"], archive)

        step("Installing…")
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp)

        candidates = [c for c in Path(tmp).rglob("llama-swap") if c.is_file()]
        if not candidates:
            die("Binary 'llama-swap' not found inside archive")

        install_dir.mkdir(parents=True, exist_ok=True)
        if binary.exists():
            shutil.copy2(binary, binary.with_suffix(".prev"))
            info(f"Old binary backed up → {binary.with_suffix('.prev')}")

        shutil.copy2(candidates[0], binary)
        binary.chmod(0o755)

    new_ver = _current_version(binary)
    ok(f"Installed llama-swap {new_ver} → {binary}")
    return True
