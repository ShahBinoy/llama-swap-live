"""Entry-point: argument parsing and command dispatch."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .colors import banner, bold, cyan, die, info, ok, step, warn
from .config import (
    DEFAULT_CONFIG_PATH, default_engine, engine_cfg, expand,
    listen_addr, load, ls_cfg,
)


# ── command handlers ───────────────────────────────────────────────────────────

def cmd_start(doc: dict) -> None:
    from .process import find_pids, start, stop
    ls = ls_cfg(doc)

    binary     = expand(ls["exec"])
    swap_conf  = expand(ls["swap-config"])
    listen     = listen_addr(doc)
    log        = expand(ls["log-file"])

    banner("llama-swap-live — Start")

    existing = find_pids()
    if existing:
        ans = input(cyan(f"  llama-swap already running (PIDs: {existing}). Restart? [y/N] ")).strip().lower()
        if ans != "y":
            ok("Nothing to do."); return
        stop()

    start(binary, swap_conf, listen, log)


def cmd_restart(doc: dict) -> None:
    from .process import start, stop
    ls = ls_cfg(doc)

    banner("llama-swap-live — Restart")
    stop()
    start(expand(ls["exec"]), expand(ls["swap-config"]), listen_addr(doc), expand(ls["log-file"]))


def cmd_status(doc: dict) -> None:
    from .process import status
    banner("llama-swap-live — Status")
    status(expand(ls_cfg(doc)["exec"]))


def cmd_update(doc: dict, check_only: bool) -> None:
    from .process import start, stop
    from .updater import update
    ls = ls_cfg(doc)

    banner("llama-swap-live — Update")
    install_dir = expand(ls["exec"]).parent
    installed   = update(install_dir, check_only=check_only)

    if installed:
        step("Restarting llama-swap…")
        stop()
        start(expand(ls["exec"]), expand(ls["swap-config"]), listen_addr(doc), expand(ls["log-file"]))


def cmd_remove(doc: dict, no_confirm: bool = False) -> None:
    from .remover import cmd_remove as _rm
    _rm(
        swap_config_path = expand(ls_cfg(doc)["swap-config"]),
        lsl_doc          = doc,
        no_confirm       = no_confirm,
    )


def cmd_add(doc: dict, args: argparse.Namespace) -> None:
    from .buckets import ask_bucket, detect as detect_bucket, name_to_bucket
    from .engines import get_engine
    from .swapconfig import inject

    banner("llama-swap-live — Add Model")

    repo_id      = args.repo
    quant        = args.quant
    multimodal   = args.mm
    mmproj_quant = args.mmproj_quant
    force_bucket = args.size_bucket
    force_macro  = args.macro
    custom_name  = args.name
    eng_name     = args.engine or default_engine(doc)

    try:
        engine = get_engine(eng_name)
    except ValueError as e:
        die(str(e))

    eng_cfg   = engine_cfg(doc, eng_name)
    swap_conf = expand(ls_cfg(doc)["swap-config"])
    ls        = ls_cfg(doc)

    print(f"\n    Repo   : {cyan(repo_id)}")
    if quant:
        print(f"    Quant  : {cyan(quant)}")
    print(f"    Engine : {cyan(eng_name)}")
    print(f"    Multi  : {cyan('yes' if multimodal else 'no')}")

    # ── resolve size bucket ────────────────────────────────────────────────────
    if force_bucket:
        bucket = force_bucket
        info(f"Using forced bucket: {bold(bucket)}")
    else:
        bucket = name_to_bucket(repo_id.split("/")[-1]) or detect_bucket(repo_id) or ask_bucket()

    print(f"    Bucket : {cyan(bucket)}")

    # ── pull / download ────────────────────────────────────────────────────────
    model_path, mmproj_path = engine.pull(
        repo_id      = repo_id,
        quant        = quant,
        multimodal   = multimodal,
        mmproj_quant = mmproj_quant,
        engine_cfg   = eng_cfg,
        lsl_cfg      = doc,
    )

    # ── show downloaded files (llama-server only) ──────────────────────────────
    if model_path and model_path.parent.exists():
        step("Downloaded files:")
        for f in sorted(model_path.parent.iterdir()):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name}  ({size_mb:.0f} MB)")

    # ── build cmd string ───────────────────────────────────────────────────────
    cmd_str = engine.build_cmd(
        repo_id     = repo_id,
        quant       = quant,
        model_path  = model_path,
        mmproj_path = mmproj_path,
        multimodal  = multimodal,
        engine_cfg  = eng_cfg,
        lsl_cfg     = doc,
        macro       = force_macro,
    )

    # ── build display name ─────────────────────────────────────────────────────
    # Extract ctx_k from the selected macro for llama-server (e.g. "llama-28k" → 28)
    ctx_k: int | None = None
    if eng_name == "llama-server":
        import re
        m = re.search(r"llama-(\d+)k", cmd_str)
        if m:
            ctx_k = int(m.group(1))

    display_name = custom_name or engine.build_name(
        repo_id    = repo_id,
        quant      = quant,
        bucket     = bucket,
        multimodal = multimodal,
        ctx_k      = ctx_k,
        engine_cfg = eng_cfg,
        lsl_cfg    = doc,
    )

    # ── inject into llama-swap config ──────────────────────────────────────────
    inject(
        swap_config_path = swap_conf,
        repo_id          = repo_id,
        quant            = quant,
        bucket           = bucket,
        cmd_str          = cmd_str,
        display_name     = display_name,
        engine_name      = eng_name,
    )

    print()
    ok("Done.")


# ── argument parser ────────────────────────────────────────────────────────────

EPILOG = """
Commands:
  (default)     Start llama-swap in the background
  --status      Show running process and installed version
  --restart     Gracefully restart the process
  --rm          Interactively remove models (config + disk)
  --update      Download latest llama-swap binary from GitHub and restart
  --add         Download a model and register it in llama-swap config

Examples:
  llama-swap-live
  llama-swap-live --status
  llama-swap-live --restart
  llama-swap-live --update
  llama-swap-live --update --check
  llama-swap-live --rm

  # llama-server (default engine)
  llama-swap-live --add -r bartowski/Qwen3-30B-A3B-GGUF -q Q6_K_L
  llama-swap-live --add -r mradermacher/gemma-4-31B-GGUF -q Q6_K --mm
  llama-swap-live --add -r llmfan46/gemma-4-GGUF -q Q6_K --mm --mmproj-quant BF16

  # rapid-mlx engine (Apple Silicon)
  llama-swap-live --add -r mlx-community/Qwen3.5-27B-4bit --engine rapid-mlx
  llama-swap-live --add -r mlx-community/gemma-3-27b-it-4bit --engine rapid-mlx --mm
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llama-swap-live",
        description="Unified manager for llama-swap and HuggingFace model downloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )

    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        metavar="PATH",
        help=f"lsl.yaml path (default: {DEFAULT_CONFIG_PATH})",
    )

    # ── mode flags ─────────────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--add",     action="store_true", help="Download & register a model")
    mode.add_argument("--rm",      action="store_true", help="Interactively remove models from config + disk")
    mode.add_argument("--update",  action="store_true", help="Update llama-swap binary from GitHub")
    mode.add_argument("--restart", action="store_true", help="Restart llama-swap process")
    mode.add_argument("--status",  action="store_true", help="Show process status and binary version")

    # ── --add options ──────────────────────────────────────────────────────────
    g = p.add_argument_group("--add options")
    g.add_argument("-r", "--repo",       metavar="REPO",   help="HuggingFace repo ID (owner/name)")
    g.add_argument("-q", "--quant",      metavar="QUANT",  help="Quantisation filter e.g. Q6_K_L  (llama-server only)")
    g.add_argument("--engine",           metavar="ENGINE", help="Engine to use: llama-server (default) | rapid-mlx")
    g.add_argument("--mm",               action="store_true", help="Multi-modal: download mmproj / pass --mllm")
    g.add_argument("--mmproj-quant",     metavar="Q",      help="mmproj quant preference (default: BF16, llama-server only)")
    g.add_argument("--size-bucket",      metavar="BUCKET", help="Force size bucket e.g. 20B-31B")
    g.add_argument("--macro",            metavar="MACRO",  help="Override llama-swap macro (llama-server only)")
    g.add_argument("--name",             metavar="NAME",   help="Override display name in config")

    # ── --rm options ───────────────────────────────────────────────────────────
    r = p.add_argument_group("--rm options")
    r.add_argument("--no-confirm", action="store_true",
                   help="Skip 'yes' confirmation prompt (use with care in scripts)")

    # ── --update options ───────────────────────────────────────────────────────
    u = p.add_argument_group("--update options")
    u.add_argument("--check", action="store_true", help="Check for newer version without installing")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    doc = load(Path(args.config))

    if args.add:
        if not args.repo:
            parser.error("--add requires -r/--repo")
        cmd_add(doc, args)
    elif args.rm:
        cmd_remove(doc, no_confirm=args.no_confirm)
    elif args.update:
        cmd_update(doc, check_only=args.check)
    elif args.restart:
        cmd_restart(doc)
    elif args.status:
        cmd_status(doc)
    else:
        cmd_start(doc)


if __name__ == "__main__":
    main()
