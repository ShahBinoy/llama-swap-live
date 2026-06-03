"""llama-swap process lifecycle: start, stop, restart, status."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from .colors import info, ok, warn, die, step, cyan


def find_pids() -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-x", "llama-swap"], text=True)
        return [int(p) for p in out.strip().splitlines() if p]
    except subprocess.CalledProcessError:
        return []


def stop(timeout: int = 6) -> None:
    pids = find_pids()
    if not pids:
        info("No running llama-swap process found")
        return

    info(f"Stopping llama-swap (PIDs: {pids})…")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not find_pids():
            ok("Stopped cleanly")
            return
        time.sleep(0.4)

    warn("Still alive after SIGTERM — sending SIGKILL")
    for pid in find_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(0.5)


def start(binary: Path, swap_config: Path, listen: str, log_file: Path) -> None:
    if not binary.exists():
        die(f"Binary not found: {binary}\nRun: llama-swap-live --update")
    if not swap_config.exists():
        die(f"Config not found: {swap_config}")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_file, "a")

    proc = subprocess.Popen(
        [str(binary), "--config", str(swap_config), "--listen", listen],
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,
    )
    time.sleep(2)

    if proc.poll() is not None:
        die(f"llama-swap exited immediately — check {log_file}")

    ok(f"llama-swap started (PID {proc.pid})")
    print(f"    Listen : {cyan(listen)}")
    print(f"    Config : {cyan(str(swap_config))}")
    print(f"    Log    : {cyan(str(log_file))}")


def status(binary: Path) -> None:
    pids = find_pids()
    if pids:
        ok(f"llama-swap is RUNNING (PIDs: {', '.join(map(str, pids))})")
    else:
        warn("llama-swap is NOT running")

    if binary.exists():
        try:
            ver = subprocess.check_output(
                [str(binary), "--version"], text=True, stderr=subprocess.STDOUT
            ).strip()
            info(f"Binary version : {ver}")
        except Exception:
            info(f"Binary         : {binary}")
    else:
        warn(f"Binary not found: {binary}")
