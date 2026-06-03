"""Manager config (~/.llama-swap/config.yaml) load/save via ruamel.yaml."""
from __future__ import annotations

import os
from pathlib import Path

from ruamel.yaml import YAML

from .colors import die, ok, warn

_yaml = YAML()
_yaml.default_flow_style = False
_yaml.preserve_quotes = True

DEFAULT_CONFIG_PATH = Path.home() / ".llama-swap" / "config.yaml"

DEFAULTS: dict[str, str] = {
    "llama-swap-bin": str(Path.home() / "bin" / "llama"),
    "model-root":     str(Path.home() / ".cache" / "llama.cpp" / "models"),
    "swap-config":    str(Path.home() / ".llama-swap.yaml"),
    "listen":         "0.0.0.0:8080",
    "log-file":       str(Path.home() / ".llama-swap" / "llama-swap.log"),
}


def load(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not path.exists():
        warn(f"Config not found at {path} — creating with defaults")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            _yaml.dump(dict(DEFAULTS), f)
        ok(f"Created default config → {path}")
        return dict(DEFAULTS)

    with open(path) as f:
        data = _yaml.load(f) or {}

    # Back-fill any missing keys without touching existing ones
    changed = False
    for k, v in DEFAULTS.items():
        if k not in data:
            data[k] = v
            changed = True
    if changed:
        with open(path, "w") as f:
            _yaml.dump(data, f)

    return dict(data)


def expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))
