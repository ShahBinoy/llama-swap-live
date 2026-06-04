"""
Interactive model removal.

Reads the engine: field from each config entry to dispatch disk deletion
to the correct engine (shutil.rmtree for llama-server, rapid-mlx rm for rapid-mlx).
Missing engine: field defaults to "llama-server" for backward compatibility.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .colors import (
    banner, bold, cyan, err, green, info, ok, red, step, warn, yellow,
)
from .swapconfig import _load, remove_keys


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    key:          str
    display_name: str
    size_token:   str           # first pipe segment from name: e.g. "31B"
    engine_name:  str           # "llama-server" | "rapid-mlx" | ...
    model_path:   Optional[Path]
    mmproj_path:  Optional[Path]

    @property
    def model_dir(self) -> Optional[Path]:
        """Parent directory of --model path. None for rapid-mlx entries."""
        return self.model_path.parent if self.model_path else None

    @property
    def dir_size_mb(self) -> float:
        d = self.model_dir
        if not d or not d.exists():
            return 0.0
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)


# ── Parsing ────────────────────────────────────────────────────────────────────

_FLAG_RE = re.compile(r"--(\w[\w-]*)\s+(\S+)")


def _extract_paths(cmd_str: str) -> tuple[Optional[Path], Optional[Path]]:
    flags: dict[str, str] = {}
    for flag, value in _FLAG_RE.findall(cmd_str):
        flags.setdefault(flag, value)
    model  = Path(flags["model"])  if "model"  in flags else None
    mmproj = Path(flags["mmproj"]) if "mmproj" in flags else None
    return model, mmproj


def _size_token(name: str) -> str:
    parts = [p.strip() for p in name.split("|")]
    if parts and re.match(r"^\d+B$", parts[0], re.IGNORECASE):
        return parts[0].upper()
    return "Unknown"


def parse_models(swap_config_path: Path) -> list[ModelEntry]:
    _, doc = _load(swap_config_path)
    models = doc.get("models") or {}

    entries: list[ModelEntry] = []
    for key, block in models.items():
        if not isinstance(block, dict):
            continue
        cmd_str      = str(block.get("cmd", ""))
        display_name = str(block.get("name", key)).strip().strip('"\'')
        # Default to llama-server for entries without engine: field (backward compat)
        engine_name  = str(block.get("engine", "llama-server"))
        model_path, mmproj_path = _extract_paths(cmd_str)

        entries.append(ModelEntry(
            key          = str(key),
            display_name = display_name,
            size_token   = _size_token(display_name),
            engine_name  = engine_name,
            model_path   = model_path,
            mmproj_path  = mmproj_path,
        ))
    return entries


# ── Display ────────────────────────────────────────────────────────────────────

def _dir_label(entry: ModelEntry) -> str:
    if entry.engine_name != "llama-server":
        return cyan(f"managed by {entry.engine_name}")
    d = entry.model_dir
    if d is None:
        return yellow("(no --model path)")
    if not d.exists():
        return yellow(f"{d}  [NOT ON DISK]")
    return cyan(f"{d}  ({entry.dir_size_mb:.0f} MB)")


def show_model_list(entries: list[ModelEntry]) -> None:
    groups: dict[str, list[tuple[int, ModelEntry]]] = {}
    for idx, e in enumerate(entries, 1):
        groups.setdefault(e.size_token, []).append((idx, e))

    print()
    for size_label, group in groups.items():
        print(bold(f"  ── {size_label} " + "─" * (58 - len(size_label))))
        for idx, e in group:
            engine_tag = f"[{e.engine_name}]" if e.engine_name != "llama-server" else ""
            print(f"  {bold(cyan(f'{idx:<3}'))}  {e.display_name}  {cyan(engine_tag)}")
            print(f"         key : {e.key}")
            print(f"         dir : {_dir_label(e)}")
        print()


# ── Shared-directory detection ─────────────────────────────────────────────────

def _find_shared_dirs(
    selected:    list[ModelEntry],
    all_entries: list[ModelEntry],
) -> set[Path]:
    selected_keys  = {e.key for e in selected}
    remaining_dirs: set[Path] = set()
    for e in all_entries:
        if e.key not in selected_keys and e.model_dir:
            remaining_dirs.add(e.model_dir)
    shared: set[Path] = set()
    for e in selected:
        if e.model_dir and e.model_dir in remaining_dirs:
            shared.add(e.model_dir)
    return shared


# ── Main entry ─────────────────────────────────────────────────────────────────

def cmd_remove(
    swap_config_path: Path,
    lsl_doc:          dict,
    no_confirm:       bool = False,
) -> None:
    banner("llama-swap-live — Remove Models")

    all_entries = parse_models(swap_config_path)
    if not all_entries:
        info("No models found in config.")
        return

    show_model_list(all_entries)

    try:
        raw = input(cyan("  Enter numbers to remove (comma-separated, e.g. 1,3): ")).strip()
    except (EOFError, KeyboardInterrupt):
        print(); ok("Cancelled."); return

    if not raw:
        ok("Nothing selected — exiting."); return

    selected: list[ModelEntry] = []
    seen: set[int] = set()
    bad: list[str] = []

    for token in raw.split(","):
        token = token.strip()
        if not token: continue
        if not token.isdigit(): bad.append(token); continue
        idx = int(token)
        if idx < 1 or idx > len(all_entries): bad.append(str(idx)); continue
        if idx not in seen:
            seen.add(idx)
            selected.append(all_entries[idx - 1])

    if bad:
        warn(f"Ignored invalid selections: {', '.join(bad)}")
    if not selected:
        ok("Nothing valid selected — exiting."); return

    shared_dirs = _find_shared_dirs(selected, all_entries)

    # ── confirmation ───────────────────────────────────────────────────────────
    print()
    print(bold(red(f"  ⚠  {len(selected)} model(s) selected for removal:")))
    print()
    for e in selected:
        d = e.model_dir
        if e.engine_name != "llama-server":
            disk_action = cyan(f"rapid-mlx rm {e.key.split(':')[0]}")
        elif d in shared_dirs:
            disk_action = yellow("CONFIG ONLY  (directory shared with another model)")
        elif d and d.exists():
            disk_action = red(f"DELETE {d}  ({e.dir_size_mb:.0f} MB)")
        elif d:
            disk_action = yellow(f"{d}  [not on disk]")
        else:
            disk_action = yellow("CONFIG ONLY  (no --model path)")

        print(f"    {bold(e.display_name)}  [{e.engine_name}]")
        print(f"      key  : {e.key}")
        print(f"      disk : {disk_action}")
        print()

    if shared_dirs:
        print(yellow("  ℹ  Shared directories will NOT be deleted from disk."))
        print()

    if not no_confirm:
        try:
            ans = input(bold(red("  Type 'yes' to confirm: "))).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); ok("Aborted."); return
        if ans != "yes":
            ok("Aborted — nothing deleted."); return

    # ── backup ─────────────────────────────────────────────────────────────────
    bak = swap_config_path.with_suffix(".yaml.bak")
    shutil.copy2(swap_config_path, bak)
    info(f"Config backed up → {bak}")

    # ── disk deletion — dispatched per engine ──────────────────────────────────
    step("Deleting…")
    from .engines import get_engine
    from .config import engine_cfg as get_engine_cfg

    removed  = 0
    skipped  = 0
    missing  = 0
    errors   = 0

    for e in selected:
        eng_cfg = get_engine_cfg(lsl_doc, e.engine_name)
        try:
            engine = get_engine(e.engine_name)
        except ValueError as exc:
            warn(f"  [{e.key}] {exc} — skipping disk deletion")
            skipped += 1
            continue

        d = e.model_dir

        if e.engine_name == "llama-server":
            if d in shared_dirs:
                warn(f"  [{e.key}] Shared directory — skipping: {d}")
                skipped += 1
                continue
            if d and not d.exists():
                warn(f"  [{e.key}] Not on disk: {d}")
                missing += 1
                continue

        try:
            engine.remove(e.key, d, eng_cfg)
            removed += 1
        except Exception as exc:
            err(f"  [{e.key}] Remove failed: {exc}")
            errors += 1

    # ── config removal ─────────────────────────────────────────────────────────
    try:
        remove_keys(swap_config_path, [e.key for e in selected])
        ok(f"  Removed {len(selected)} entr{'y' if len(selected)==1 else 'ies'} from config")
    except Exception as exc:
        err(f"  Config update failed: {exc}")
        errors += 1

    # ── summary ────────────────────────────────────────────────────────────────
    print()
    print(bold("  Summary"))
    print(f"    Removed   : {green(str(removed))}")
    if skipped: print(f"    Skipped   : {yellow(str(skipped))}")
    if missing: print(f"    Not found : {yellow(str(missing))}")
    if errors:  print(f"    Errors    : {red(str(errors))}")
    print(f"    Backup    : {bak}")
    print()
    ok("Done.")
