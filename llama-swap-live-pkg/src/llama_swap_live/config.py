"""
Manager config: ~/.llama-swap/lsl.yaml

Replaces the old config.yaml. Generates a scaffold if absent.
The old config.yaml is NOT migrated — users are notified.
"""
from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

from ruamel.yaml import YAML

from .colors import die, info, ok, warn

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.default_flow_style = False

DEFAULT_CONFIG_PATH = Path.home() / ".llama-swap" / "lsl.yaml"
OLD_CONFIG_PATH     = Path.home() / ".llama-swap" / "config.yaml"

# ── Scaffold template ──────────────────────────────────────────────────────────
# Values match the example in the feature spec exactly.
_SCAFFOLD = """\
llama-swap:
  exec: "{home}/bin/llama/llama-swap"
  swap-config: "{home}/bin/llama/config.yaml"
  listen-host: "0.0.0.0"
  listen-port: 8080
  log-file: "{home}/.llama-swap/llama-swap.log"
  default-engine: "llama-server"
  gpu-capacity: 32
  kv-quant: 8

engines:
  llama-server:
    exec: "{home}/bin/llama/bin/llama-server"
    model-root: "{home}/.cache/llama.cpp/models"
    proxy-host: "0.0.0.0"
    proxy-port: 1234

  rapid-mlx:
    exec: "{home}/bin/rapid-mlx"
    model-root: "{home}/.cache/rapid-mlx/models"
    proxy-host: "0.0.0.0"
    proxy-port: 8000
    # prefill-step-size: 8192   # uncomment to set --prefill-step-size
    # max-tokens: 4096          # uncomment to set --max-tokens
    # kv-bits: 8                # overrides llama-swap.kv-quant for this engine
"""


def _scaffold_content() -> str:
    home = str(Path.home())
    return _SCAFFOLD.replace("{home}", home)


def load(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    # Notify about old config but do not migrate
    if OLD_CONFIG_PATH.exists() and not path.exists():
        warn(
            f"Found old config at {OLD_CONFIG_PATH} — "
            f"lsl.yaml is the new format. A scaffold has been generated."
        )

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_scaffold_content())
        ok(f"Created lsl.yaml scaffold → {path}")
        info("Edit it to match your paths, then re-run.")

    with open(path) as f:
        doc = _yaml.load(f) or {}

    return doc


def expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


# ── Convenience accessors ──────────────────────────────────────────────────────

def ls_cfg(doc: dict) -> dict:
    """Return the llama-swap: block."""
    return doc.get("llama-swap", {})


def engine_cfg(doc: dict, engine_name: str) -> dict:
    """Return the engines.<name>: block, empty dict if absent."""
    return (doc.get("engines") or {}).get(engine_name, {})


def default_engine(doc: dict) -> str:
    """Return the configured default engine, falling back to llama-server."""
    return ls_cfg(doc).get("default-engine", "llama-server")


def listen_addr(doc: dict) -> str:
    """Return host:port string for llama-swap --listen."""
    ls = ls_cfg(doc)
    host = ls.get("listen-host", "0.0.0.0")
    port = ls.get("listen-port", 8080)
    return f"{host}:{port}"
