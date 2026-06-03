"""HuggingFace model downloading (GGUF + mmproj)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .colors import info, ok, warn, step, die, bold


def _list_mmproj_files(repo_id: str) -> list[str]:
    """Return all mmproj GGUF filenames in the HF repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        return [
            f for f in api.list_repo_files(repo_id)
            if "mmproj" in f.lower() and f.endswith(".gguf")
        ]
    except Exception:
        return []


def _pick_mmproj(files: list[str], quant_hint: Optional[str]) -> Optional[str]:
    if not files:
        return None
    if quant_hint:
        matches = [f for f in files if quant_hint.upper() in f.upper()]
        if matches:
            return matches[0]
    for pref in ("BF16", "bf16", "Q8_0", "q8_0"):
        for f in files:
            if pref in f:
                return f
    return files[0]


def _run_hf_download(args: list[str]) -> None:
    if not shutil.which("huggingface-cli"):
        die("huggingface-cli not found. Install with: pip install huggingface_hub")
    result = subprocess.run(["huggingface-cli", "download"] + args)
    if result.returncode != 0:
        die(f"huggingface-cli download failed (exit {result.returncode})")


def download(
    repo_id: str,
    quant: Optional[str],
    dest_dir: Path,
    multimodal: bool,
    mmproj_quant: Optional[str],
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Download model GGUF and optionally mmproj.
    Returns (gguf_path, mmproj_path | None).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ── Main GGUF ─────────────────────────────────────────────────────────────
    step(f"Downloading {bold(repo_id)}" + (f":{quant}" if quant else ""))
    info(f"Destination: {dest_dir}")

    dl_args = [repo_id, "--local-dir", str(dest_dir)]
    if quant:
        dl_args += ["--include", f"*{quant}*.gguf", "--exclude", "*mmproj*"]
    _run_hf_download(dl_args)

    gguf_files = sorted(
        f for f in dest_dir.rglob("*.gguf")
        if "mmproj" not in f.name.lower()
        and (not quant or quant.lower() in f.name.lower())
    )
    gguf_path = gguf_files[0] if gguf_files else None
    if gguf_path:
        ok(f"Model GGUF: {gguf_path.name}")
    else:
        warn("No GGUF found in destination directory after download")

    # ── mmproj ────────────────────────────────────────────────────────────────
    mmproj_path: Optional[Path] = None
    if multimodal:
        step("Resolving mmproj file...")
        remote = _list_mmproj_files(repo_id)
        chosen = _pick_mmproj(remote, mmproj_quant)

        if not chosen:
            warn("No mmproj GGUF found in repo — skipping")
        else:
            info(f"Selected: {chosen}")
            _run_hf_download([
                repo_id,
                "--local-dir", str(dest_dir),
                "--include", f"*{Path(chosen).name}",
            ])
            # huggingface-cli may nest inside author/model-name subdir
            candidates = [dest_dir / chosen] + list(dest_dir.rglob(Path(chosen).name))
            for c in candidates:
                if c.exists():
                    mmproj_path = c
                    ok(f"mmproj:     {mmproj_path.name}")
                    break
            if not mmproj_path:
                warn("mmproj file not found after download")

    return gguf_path, mmproj_path
