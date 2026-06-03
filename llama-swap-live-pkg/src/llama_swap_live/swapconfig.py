"""Inject and manage model entries in the llama-swap server config via ruamel.yaml."""
from __future__ import annotations

import re
import shutil
from io import StringIO
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML

from .buckets import BUCKET_MACRO
from .colors import cyan, ok, step, warn

# Round-trip YAML instance — preserves comments, ordering, blank lines, quotes
_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 4096          # prevent unwanted line-wrapping of long paths
_yaml.indent(mapping=4, sequence=4, offset=4)  # match user's 4-space config style


def _load(path: Path) -> tuple[YAML, object]:
    """Load the swap config, returning (yaml_instance, document)."""
    with open(path) as f:
        return _yaml, _yaml.load(f)


def _dump(doc: object, path: Path) -> None:
    """Write the document back, preserving all ruamel.yaml round-trip metadata."""
    buf = StringIO()
    _yaml.dump(doc, buf)
    path.write_text(buf.getvalue())


def _build_cmd(macro: str, gguf_path: Path, mmproj_path: Optional[Path], alias: str) -> str:
    """
    Build the literal block scalar string for the cmd: field.
    Each argument on its own line, consistent with the style in the user's config.
    """
    lines = [
        "${" + macro + "}",
        f"--model {gguf_path}",
    ]
    if mmproj_path:
        lines.append(f"--mmproj {mmproj_path}")
    lines += [
        f"--alias {alias}",
        "--cache-type-k q8_0",
        "--cache-type-v q8_0",
    ]
    # ruamel block scalar: join with \n, trailing \n required
    return "\n".join(lines) + "\n"


def inject(
    swap_config_path: Path,
    repo_id: str,
    quant: Optional[str],
    bucket: str,
    gguf_path: Path,
    mmproj_path: Optional[Path],
    macro: Optional[str],
    display_name: Optional[str],
) -> None:
    """
    Add a new model entry to the llama-swap config.
    Uses ruamel.yaml round-trip so comments/formatting are fully preserved.
    """
    if not swap_config_path.exists():
        warn(f"llama-swap config not found: {swap_config_path} — skipping")
        return

    step("Updating llama-swap config...")

    model_key    = f"{repo_id}:{quant}" if quant else repo_id
    resolved_mac = macro or BUCKET_MACRO.get(bucket, "llama-16k")

    author     = repo_id.split("/")[0]
    model_name = repo_id.split("/")[-1]

    # Name follows the established pattern:
    # "<size> | <author> | <model> | <ctx> | <capability>"
    # We set size + author + model here; ctx comes from the macro name (e.g. llama-28k → 28K)
    if not display_name:
        size_label = bucket.split("-")[-1]           # "31B" from "20B-31B"
        ctx_label  = resolved_mac.replace("llama-", "").upper()   # "28K"
        capability = "Multi" if mmproj_path else "Writing"
        display_name = f"{size_label} | {author} | {model_name} | {ctx_label} | {capability}"

    _, doc = _load(swap_config_path)

    if "models" not in doc or doc["models"] is None:
        doc["models"] = {}

    if model_key in doc["models"]:
        warn(f"'{model_key}' already exists in config — skipping")
        return

    # Build the entry as a plain dict; ruamel will serialise it correctly
    from ruamel.yaml.scalarstring import LiteralScalarString
    entry = {
        "cmd": LiteralScalarString(_build_cmd(resolved_mac, gguf_path, mmproj_path, model_key)),
        "name": display_name,
        "proxy": "http://127.0.0.1:1234",
    }

    doc["models"][model_key] = entry

    bak = swap_config_path.with_suffix(".yaml.bak")
    shutil.copy2(swap_config_path, bak)
    _dump(doc, swap_config_path)

    ok("Added to llama-swap config")
    print(f"    Key    : {cyan(model_key)}")
    print(f"    Name   : {cyan(display_name)}")
    print(f"    Macro  : {cyan(resolved_mac)}")
    print(f"    Backup : {bak}")


def remove_keys(swap_config_path: Path, keys: list[str]) -> None:
    """Remove a list of model keys from the config. Called by remover.py."""
    _, doc = _load(swap_config_path)
    models = doc.get("models") or {}
    for key in keys:
        if key in models:
            del models[key]
    _dump(doc, swap_config_path)
