"""
Engine abstraction layer.

Each engine is a module in this package implementing the EngineProtocol.
Adding a third engine = one new file that subclasses EngineBase.

Registry:  ENGINES["llama-server"]  →  LlamaServerEngine()
           ENGINES["rapid-mlx"]     →  RapidMlxEngine()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class EngineBase(ABC):
    """
    Contract every engine must satisfy.

    All methods receive the full lsl.yaml document (as a plain dict)
    so engines can read any top-level config they need without
    requiring extra arguments to be threaded through every call site.
    """

    # Canonical engine name — must match the key in lsl.yaml engines:
    name: str = ""

    # ── lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def pull(
        self,
        repo_id:     str,
        quant:       Optional[str],
        multimodal:  bool,
        mmproj_quant: Optional[str],
        engine_cfg:  dict,
        lsl_cfg:     dict,
    ) -> tuple[Optional[Path], Optional[Path]]:
        """
        Download the model.
        Returns (primary_path, mmproj_path).
        rapid-mlx returns (None, None) — it manages its own cache.
        llama-server returns (gguf_path, mmproj_path|None).
        """

    @abstractmethod
    def remove(
        self,
        model_key:  str,
        model_dir:  Optional[Path],
        engine_cfg: dict,
    ) -> None:
        """
        Delete model files from disk.
        llama-server: shutil.rmtree(model_dir)
        rapid-mlx:    subprocess `rapid-mlx rm <repo_id>`
        """

    # ── config generation ─────────────────────────────────────────────────────

    @abstractmethod
    def build_cmd(
        self,
        repo_id:    str,
        quant:      Optional[str],
        model_path: Optional[Path],
        mmproj_path: Optional[Path],
        multimodal: bool,
        engine_cfg: dict,
        lsl_cfg:    dict,
        macro:      Optional[str] = None,   # llama-server: override macro
    ) -> str:
        """
        Return the literal block scalar string for the cmd: field
        in the llama-swap config.
        """

    @abstractmethod
    def build_name(
        self,
        repo_id:    str,
        quant:      Optional[str],
        bucket:     str,
        multimodal: bool,
        ctx_k:      Optional[int],          # context window in K tokens, e.g. 28
        engine_cfg: dict,
        lsl_cfg:    dict,
    ) -> str:
        """
        Return the display name string following the pipe-delimited convention:
        "<size> | <author> | <model> | <ctx>K | <capability>"
        """


# ── Registry ───────────────────────────────────────────────────────────────────

def _load_engines() -> dict[str, EngineBase]:
    from .llama_server import LlamaServerEngine
    from .rapid_mlx import RapidMlxEngine
    engines: list[EngineBase] = [LlamaServerEngine(), RapidMlxEngine()]
    return {e.name: e for e in engines}


# Lazily populated on first access
_REGISTRY: Optional[dict[str, EngineBase]] = None


def get_engine(name: str) -> EngineBase:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _load_engines()
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown engine '{name}'. Available: {available}")
    return _REGISTRY[name]
