"""
llama-server engine.

Behaviour is identical to the pre-refactor code:
  - Downloads GGUF (and optionally mmproj) via huggingface-cli
  - Selects the best llama-XXk macro using context estimation
  - Builds the cmd: block with ${macro} + --model + --alias + --cache-type flags
  - Removes via shutil.rmtree on the model directory
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ..colors import die, info, ok, step, warn
from .context import estimate_context, print_estimate
from . import EngineBase


def _fetch_arch(repo_id: str) -> dict:
    """
    Fetch model architecture config from HuggingFace Hub.
    Returns a dict with the relevant keys; empty dict on failure.
    """
    try:
        from huggingface_hub import hf_hub_download
        import json
        cfg_path = hf_hub_download(repo_id, "config.json", timeout=15)
        with open(cfg_path) as f:
            cfg = json.load(f)
        return {
            k: cfg[k] for k in (
                "num_hidden_layers",
                "num_key_value_heads",
                "num_attention_heads",
                "hidden_size",
                "head_dim",
            ) if k in cfg
        }
    except Exception as e:
        warn(f"Could not fetch model config.json: {e}")
        return {}


def _run_hf_download(args: list[str]) -> None:
    if not shutil.which("huggingface-cli"):
        die("huggingface-cli not found. Install: pip install huggingface_hub")
    result = subprocess.run(["huggingface-cli", "download"] + args)
    if result.returncode != 0:
        die(f"huggingface-cli download failed (exit {result.returncode})")


def _list_mmproj_files(repo_id: str) -> list[str]:
    try:
        from huggingface_hub import HfApi
        return [
            f for f in HfApi().list_repo_files(repo_id)
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


class LlamaServerEngine(EngineBase):
    name = "llama-server"

    def pull(
        self,
        repo_id:      str,
        quant:        Optional[str],
        multimodal:   bool,
        mmproj_quant: Optional[str],
        engine_cfg:   dict,
        lsl_cfg:      dict,
    ) -> tuple[Optional[Path], Optional[Path]]:
        model_root = Path(engine_cfg["model-root"]).expanduser()

        # Derive bucket + dest dir
        from ..buckets import detect as detect_bucket, ask_bucket, name_to_bucket
        bucket = name_to_bucket(repo_id.split("/")[-1]) or detect_bucket(repo_id) or ask_bucket()

        author     = repo_id.split("/")[0]
        model_name = repo_id.split("/")[-1]
        dest_dir   = model_root / bucket / author / model_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        step(f"Downloading {repo_id}" + (f":{quant}" if quant else ""))
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
            ok(f"GGUF: {gguf_path.name}")
        else:
            warn("No GGUF found after download")

        mmproj_path: Optional[Path] = None
        if multimodal:
            step("Resolving mmproj…")
            remote  = _list_mmproj_files(repo_id)
            chosen  = _pick_mmproj(remote, mmproj_quant)
            if not chosen:
                warn("No mmproj found in repo — skipping")
            else:
                info(f"Selected: {chosen}")
                _run_hf_download([
                    repo_id, "--local-dir", str(dest_dir),
                    "--include", f"*{Path(chosen).name}",
                ])
                candidates = [dest_dir / chosen] + list(dest_dir.rglob(Path(chosen).name))
                for c in candidates:
                    if c.exists():
                        mmproj_path = c
                        ok(f"mmproj: {mmproj_path.name}")
                        break

        return gguf_path, mmproj_path

    def remove(
        self,
        model_key:  str,
        model_dir:  Optional[Path],
        engine_cfg: dict,
    ) -> None:
        if model_dir and model_dir.exists():
            shutil.rmtree(model_dir)
            ok(f"Deleted: {model_dir}")
        elif model_dir:
            warn(f"Directory not found: {model_dir}")
        else:
            warn(f"[{model_key}] No model directory — skipping disk deletion")

    def build_cmd(
        self,
        repo_id:     str,
        quant:       Optional[str],
        model_path:  Optional[Path],
        mmproj_path: Optional[Path],
        multimodal:  bool,
        engine_cfg:  dict,
        lsl_cfg:     dict,
        macro:       Optional[str] = None,
    ) -> str:
        if model_path is None:
            die("llama-server engine requires a model path")

        # ── resolve macro via context estimation ──────────────────────────────
        if not macro:
            ls_cfg     = lsl_cfg.get("llama-swap", {})
            gpu_gb     = float(ls_cfg.get("gpu-capacity", 24))
            kv_bits    = int(ls_cfg.get("kv-quant", 8))

            # Try to get param count from HF if not already known
            param_count = self._param_count_from_path(model_path)
            arch        = _fetch_arch(repo_id)

            if param_count and arch:
                result = estimate_context(param_count, quant, kv_bits, gpu_gb, arch)
                print_estimate(result, gpu_gb)
                macro = result["selected_macro"]
            else:
                # Fall back to bucket-based macro
                from ..buckets import name_to_bucket, BUCKET_MACRO
                bucket = name_to_bucket(repo_id.split("/")[-1]) or "20B-31B"
                macro  = BUCKET_MACRO.get(bucket, "llama-16k")
                info(f"Using bucket-based macro: {macro}")

        macro_ref = "${" + macro + "}"
        model_key = f"{repo_id}:{quant}" if quant else repo_id

        lines = [macro_ref, f"--model {model_path}"]
        if mmproj_path:
            lines.append(f"--mmproj {mmproj_path}")
        lines += [
            f"--alias {model_key}",
            "--cache-type-k q8_0",
            "--cache-type-v q8_0",
        ]
        return "\n".join(lines) + "\n"

    def build_name(
        self,
        repo_id:    str,
        quant:      Optional[str],
        bucket:     str,
        multimodal: bool,
        ctx_k:      Optional[int],
        engine_cfg: dict,
        lsl_cfg:    dict,
    ) -> str:
        size_label = bucket.split("-")[-1]
        author     = repo_id.split("/")[0]
        model_name = repo_id.split("/")[-1]
        ctx_str    = f"{ctx_k}K" if ctx_k else "?"
        capability = "Multi" if multimodal else "Writing"
        return f"{size_label} | {author} | {model_name} | {ctx_str} | {capability}"

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _param_count_from_path(model_path: Path) -> Optional[int]:
        """Try to derive param count from the GGUF filename via regex."""
        import re
        m = re.search(r"[^0-9](\d+(?:\.\d+)?)B", model_path.name, re.IGNORECASE)
        if m:
            return int(float(m.group(1)) * 1e9)
        return None
