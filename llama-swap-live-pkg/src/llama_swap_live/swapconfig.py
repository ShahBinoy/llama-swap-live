"""
Inject and remove model entries in the llama-swap server config via ruamel.yaml.

Engine-agnostic: receives the fully-built cmd string and name from the engine.
Adds an `engine:` field to each entry so --rm can dispatch to the right engine.
"""
from __future__ import annotations

import shutil
from io import StringIO
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from .colors import cyan, ok, step, warn

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 4096
_yaml.indent(mapping=4, sequence=4, offset=4)


def _load(path: Path) -> tuple[YAML, object]:
    with open(path) as f:
        return _yaml, _yaml.load(f)


def _dump(doc: object, path: Path) -> None:
    buf = StringIO()
    _yaml.dump(doc, buf)
    path.write_text(buf.getvalue())


def inject(
    swap_config_path: Path,
    repo_id:          str,
    quant:            Optional[str],
    bucket:           str,
    cmd_str:          str,            # literal block scalar from engine.build_cmd()
    display_name:     str,            # from engine.build_name()
    engine_name:      str,            # "llama-server" | "rapid-mlx"
) -> None:
    """
    Insert a new model entry into the llama-swap config.
    Preserves all comments, ordering and formatting via ruamel round-trip.
    """
    if not swap_config_path.exists():
        warn(f"llama-swap config not found: {swap_config_path} — skipping")
        return

    step("Updating llama-swap config…")

    model_key = f"{repo_id}:{quant}" if quant else repo_id

    _, doc = _load(swap_config_path)

    if "models" not in doc or doc["models"] is None:
        doc["models"] = {}

    if model_key in doc["models"]:
        warn(f"'{model_key}' already exists in config — skipping")
        return

    doc["models"][model_key] = {
        "engine": engine_name,
        "cmd":    LiteralScalarString(cmd_str),
        "name":   display_name,
        "proxy":  "http://127.0.0.1:1234",
    }

    bak = swap_config_path.with_suffix(".yaml.bak")
    shutil.copy2(swap_config_path, bak)
    _dump(doc, swap_config_path)

    ok("Added to llama-swap config")
    print(f"    Key    : {cyan(model_key)}")
    print(f"    Name   : {cyan(display_name)}")
    print(f"    Engine : {cyan(engine_name)}")
    print(f"    Backup : {bak}")


def remove_keys(swap_config_path: Path, keys: list[str]) -> None:
    """Remove a list of model keys. Called by remover.py."""
    _, doc = _load(swap_config_path)
    models = doc.get("models") or {}
    for key in keys:
        if key in models:
            del models[key]
    _dump(doc, swap_config_path)
