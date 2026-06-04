"""
rapid-mlx engine.

Key differences from llama-server:
  - Model identifier is a HuggingFace repo path (e.g. mlx-community/Qwen3.5-27B-4bit)
    OR a registered rapid-mlx short alias (e.g. qwen3.5-27b).
    rapid-mlx resolves both internally; we pass whatever the user gave with -r.
  - `pull`  → `rapid-mlx pull <repo_or_alias>`  (rapid-mlx manages its own cache)
  - `remove`→ `rapid-mlx rm  <repo_or_alias>`
  - `cmd:`  → `rapid-mlx serve <repo_or_alias> --port <proxy-port> [optional flags]`
  - No GGUF path, no macro, no --model flag — rapid-mlx handles all of that.
  - Context window is managed internally by MLX; we don't set --ctx-size.
  - The --mllm flag enables vision/multimodal models.
  - Optional performance flags from lsl.yaml engines.rapid-mlx:
      prefill-step-size  → --prefill-step-size N
      max-tokens         → --max-tokens N
      kv-bits            → --kv-cache-quantization N
        (falls back to lsl.llama-swap.kv-quant if not set per-engine)
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from ..colors import die, info, ok, step, warn
from . import EngineBase


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check)


class RapidMlxEngine(EngineBase):
    name = "rapid-mlx"

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _exec(engine_cfg: dict) -> str:
        """Path to the rapid-mlx executable."""
        return str(engine_cfg.get("exec", "rapid-mlx"))

    @staticmethod
    def _proxy_port(engine_cfg: dict) -> int:
        return int(engine_cfg.get("proxy-port", 8080))

    @staticmethod
    def _optional_flags(engine_cfg: dict, lsl_cfg: dict) -> list[str]:
        """
        Build optional rapid-mlx serve flags from engine config.
        Only includes a flag when the key is explicitly set — never
        injects a default so that rapid-mlx's own defaults are respected.
        """
        flags: list[str] = []

        prefill = engine_cfg.get("prefill-step-size")
        if prefill:
            flags += ["--prefill-step-size", str(prefill)]

        max_tokens = engine_cfg.get("max-tokens")
        if max_tokens:
            flags += ["--max-tokens", str(max_tokens)]

        # kv-bits: prefer engine-level, fall back to top-level kv-quant
        kv_bits = engine_cfg.get("kv-bits") or lsl_cfg.get("llama-swap", {}).get("kv-quant")
        if kv_bits:
            flags += ["--kv-cache-quantization", str(kv_bits)]

        return flags

    # ── EngineBase implementation ─────────────────────────────────────────────

    def pull(
        self,
        repo_id:      str,
        quant:        Optional[str],   # not used by rapid-mlx
        multimodal:   bool,
        mmproj_quant: Optional[str],   # not used by rapid-mlx
        engine_cfg:   dict,
        lsl_cfg:      dict,
    ) -> tuple[Optional[Path], Optional[Path]]:
        """
        Call `rapid-mlx pull <repo_id>` to pre-download the model.
        rapid-mlx manages its own cache; we return (None, None).
        """
        binary = self._exec(engine_cfg)
        step(f"Pulling {repo_id} via rapid-mlx…")
        result = _run([binary, "pull", repo_id], check=False)
        if result.returncode == 0:
            ok(f"rapid-mlx pull complete: {repo_id}")
        else:
            warn(
                f"rapid-mlx pull exited {result.returncode} — "
                "model may still be available (downloaded on first serve)"
            )
        # rapid-mlx manages its own cache; no local path to return
        return None, None

    def remove(
        self,
        model_key:  str,
        model_dir:  Optional[Path],    # not used — rapid-mlx manages its cache
        engine_cfg: dict,
    ) -> None:
        """Call `rapid-mlx rm <repo_id>` to delete the cached model."""
        binary   = self._exec(engine_cfg)
        # model_key may be "owner/repo" or "owner/repo:quant" — strip quant suffix
        repo_id  = model_key.split(":")[0]
        step(f"Removing {repo_id} from rapid-mlx cache…")
        result = _run([binary, "rm", repo_id], check=False)
        if result.returncode == 0:
            ok(f"rapid-mlx rm: {repo_id}")
        else:
            warn(f"rapid-mlx rm exited {result.returncode} for {repo_id}")

    def build_cmd(
        self,
        repo_id:     str,
        quant:       Optional[str],    # not used — rapid-mlx handles quantisation
        model_path:  Optional[Path],   # always None for rapid-mlx
        mmproj_path: Optional[Path],   # always None for rapid-mlx
        multimodal:  bool,
        engine_cfg:  dict,
        lsl_cfg:     dict,
        macro:       Optional[str] = None,  # not used
    ) -> str:
        binary = self._exec(engine_cfg)
        port   = self._proxy_port(engine_cfg)

        parts = [binary, "serve", repo_id, "--port", str(port)]

        if multimodal:
            parts.append("--mllm")

        parts += self._optional_flags(engine_cfg, lsl_cfg)

        return " ".join(parts) + "\n"

    def build_name(
        self,
        repo_id:    str,
        quant:      Optional[str],
        bucket:     str,
        multimodal: bool,
        ctx_k:      Optional[int],    # not used — MLX manages context internally
        engine_cfg: dict,
        lsl_cfg:    dict,
    ) -> str:
        size_label = bucket.split("-")[-1]
        # repo_id may be "mlx-community/Qwen3.5-27B-4bit" or a short alias
        parts      = repo_id.split("/")
        author     = parts[0] if len(parts) > 1 else "rapid-mlx"
        model_name = parts[-1]
        capability = "Multi" if multimodal else "Writing"
        return f"{size_label} | {author} | {model_name} | MLX | {capability}"
