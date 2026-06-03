"""
Interactive model removal.

Flow:
  1. Parse models: section via ruamel.yaml (preserves comments/formatting).
  2. Extract size token from name: field (first pipe-delimited segment).
  3. Display numbered list grouped by size token, sorted.
  4. Accept comma-separated index input.
  5. Cross-check selected model directories against remaining entries
     — shared directories: warn + config-only removal, no disk delete.
  6. Confirm, backup, delete directories, remove config keys.
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
    display_name: str           # full name: field value
    size_token:   str           # first pipe segment e.g. "31B"
    model_path:   Optional[Path]
    mmproj_path:  Optional[Path]

    @property
    def model_dir(self) -> Optional[Path]:
        """Parent directory of --model path — the primary deletion target."""
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
    """Extract --model and --mmproj paths from the cmd literal block string.
    
    If a value looks like a model alias (single token, no path separators,
    not an existing file) it is treated as None — Rapid-MLX entries
    don't have local model directories to remove.
    """
    flags: dict[str, str] = {}
    for flag, value in _FLAG_RE.findall(cmd_str):
        flags.setdefault(flag, value)   # first occurrence wins

    def _as_path(val: Optional[str]) -> Optional[Path]:
        if val is None:
            return None
        p = Path(val)
        # If it looks like a model alias (no separators, not an existing file)
        # treat it as a non-path — Rapid-MLX manages its own storage.
        if "/" not in val and "\\" not in val and not p.exists():
            return None
        return p

    return _as_path(flags.get("model")), _as_path(flags.get("mmproj"))


def _size_token(name: str) -> str:
    """
    Extract the size label from the name field.
    Pattern: "<size> | <author> | ..."  → "<size>"
    Falls back to "Unknown" if the name doesn't follow the convention.
    """
    parts = [p.strip() for p in name.split("|")]
    if parts and re.match(r"^\d+B$", parts[0], re.IGNORECASE):
        return parts[0].upper()
    return "Unknown"


def parse_models(swap_config_path: Path) -> list[ModelEntry]:
    """
    Load the swap config via ruamel.yaml and return ModelEntry objects
    in the order they appear under models:.
    """
    _, doc = _load(swap_config_path)
    models = doc.get("models") or {}

    entries: list[ModelEntry] = []
    for key, block in models.items():
        if not isinstance(block, dict):
            continue

        cmd_str      = str(block.get("cmd", ""))
        display_name = str(block.get("name", key)).strip().strip('"\'')
        model_path, mmproj_path = _extract_paths(cmd_str)

        entries.append(ModelEntry(
            key          = str(key),
            display_name = display_name,
            size_token   = _size_token(display_name),
            model_path   = model_path,
            mmproj_path  = mmproj_path,
        ))

    return entries


# ── Display ────────────────────────────────────────────────────────────────────

def _dir_label(entry: ModelEntry) -> str:
    d = entry.model_dir
    if d is None:
        return yellow("(no --model path)")
    if not d.exists():
        return yellow(f"{d}  [NOT ON DISK]")
    return cyan(f"{d}  ({entry.dir_size_mb:.0f} MB)")


def show_model_list(entries: list[ModelEntry]) -> None:
    """
    Print a numbered list grouped by size token (derived from name field).
    """
    # Group by size token, preserving encounter order
    groups: dict[str, list[tuple[int, ModelEntry]]] = {}
    for idx, e in enumerate(entries, 1):
        groups.setdefault(e.size_token, []).append((idx, e))

    print()
    for size_label, group in groups.items():
        print(bold(f"  ── {size_label} " + "─" * (60 - len(size_label))))
        for idx, e in group:
            num  = bold(cyan(f"  {idx:<3}"))
            name = e.display_name
            print(f"{num}  {name}")
            print(f"         key : {e.key}")
            print(f"         dir : {_dir_label(e)}")
        print()


# ── Shared-directory detection ─────────────────────────────────────────────────

def _find_shared_dirs(
    selected: list[ModelEntry],
    all_entries: list[ModelEntry],
) -> set[Path]:
    """
    Return the set of model directories that are referenced by at least one
    entry NOT in the selected set.  These must not be deleted from disk.
    """
    selected_keys = {e.key for e in selected}
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

def cmd_remove(swap_config_path: Path, no_confirm: bool = False) -> None:
    banner("llama-swap-live — Remove Models")

    all_entries = parse_models(swap_config_path)
    if not all_entries:
        info("No models found in config.")
        return

    show_model_list(all_entries)

    # ── index selection ────────────────────────────────────────────────────────
    try:
        raw = input(cyan("  Enter numbers to remove (comma-separated, e.g. 1,3): ")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        ok("Cancelled.")
        return

    if not raw:
        ok("Nothing selected — exiting.")
        return

    selected: list[ModelEntry] = []
    seen: set[int] = set()
    bad: list[str] = []

    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            bad.append(token); continue
        idx = int(token)
        if idx < 1 or idx > len(all_entries):
            bad.append(str(idx)); continue
        if idx not in seen:
            seen.add(idx)
            selected.append(all_entries[idx - 1])

    if bad:
        warn(f"Ignored invalid selections: {', '.join(bad)}")
    if not selected:
        ok("Nothing valid selected — exiting.")
        return

    # ── shared-directory analysis ──────────────────────────────────────────────
    shared_dirs = _find_shared_dirs(selected, all_entries)

    # ── confirmation table ─────────────────────────────────────────────────────
    print()
    print(bold(red(f"  ⚠  {len(selected)} model(s) selected for removal:")))
    print()
    for e in selected:
        d = e.model_dir
        disk_action: str
        if d in shared_dirs:
            disk_action = yellow("CONFIG ONLY  (directory shared with another model)")
        elif d and d.exists():
            disk_action = red(f"DELETE {d}  ({e.dir_size_mb:.0f} MB)")
        elif d:
            disk_action = yellow(f"{d}  [not on disk — config only]")
        else:
            disk_action = yellow("CONFIG ONLY  (no --model path found)")

        print(f"    {bold(e.display_name)}")
        print(f"      key  : {e.key}")
        print(f"      disk : {disk_action}")
        print()

    if shared_dirs:
        print(yellow("  ℹ  Shared directories will NOT be deleted from disk."))
        print()

    if not no_confirm:
        try:
            ans = input(bold(red(f"  Type 'yes' to confirm: "))).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            ok("Aborted.")
            return
        if ans != "yes":
            ok("Aborted — nothing deleted.")
            return

    # ── backup ─────────────────────────────────────────────────────────────────
    bak = swap_config_path.with_suffix(".yaml.bak")
    shutil.copy2(swap_config_path, bak)
    info(f"Config backed up → {bak}")

    # ── disk deletion ──────────────────────────────────────────────────────────
    step("Deleting…")
    removed_dirs = 0
    skipped_dirs = 0
    missing_dirs = 0
    errors       = 0

    for e in selected:
        d = e.model_dir

        if d is None:
            warn(f"  [{e.key}] No --model path — skipping disk deletion")
            skipped_dirs += 1
            continue

        if d in shared_dirs:
            warn(f"  [{e.key}] Shared directory — skipping disk deletion: {d}")
            skipped_dirs += 1
            continue

        if not d.exists():
            warn(f"  [{e.key}] Directory not on disk: {d}")
            missing_dirs += 1
            continue

        try:
            shutil.rmtree(d)
            ok(f"  Deleted: {d}")
            removed_dirs += 1
        except Exception as exc:
            err(f"  Failed to delete {d}: {exc}")
            errors += 1

    # ── config removal (ruamel round-trip via swapconfig) ─────────────────────
    try:
        remove_keys(swap_config_path, [e.key for e in selected])
        ok(f"  Removed {len(selected)} entr{'y' if len(selected)==1 else 'ies'} from config")
    except Exception as exc:
        err(f"  Config update failed: {exc}")
        errors += 1

    # ── summary ────────────────────────────────────────────────────────────────
    print()
    print(bold("  Summary"))
    print(f"    Directories deleted  : {green(str(removed_dirs))}")
    if skipped_dirs:
        print(f"    Skipped (shared)     : {yellow(str(skipped_dirs))}")
    if missing_dirs:
        print(f"    Not found on disk    : {yellow(str(missing_dirs))}")
    if errors:
        print(f"    Errors               : {red(str(errors))}")
    print(f"    Config backup        : {bak}")
    print()
    ok("Done.")
