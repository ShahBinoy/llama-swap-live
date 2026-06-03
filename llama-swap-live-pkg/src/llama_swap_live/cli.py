"""Entry-point: argument parsing and command dispatch."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .colors import banner, die, ok, warn
from .config import DEFAULT_CONFIG_PATH, expand, load


# ── command handlers ───────────────────────────────────────────────────────────

def cmd_start(cfg: dict) -> None:
    from .process import find_pids, start, stop

    binary = expand(cfg["llama-swap-bin"]) / "llama-swap"
    config = expand(cfg["swap-config"])
    listen = cfg["listen"]
    log    = expand(cfg["log-file"])

    banner("llama-swap-live — Start")

    existing = find_pids()
    if existing:
        from .colors import yellow
        ans = input(yellow(f"  llama-swap already running (PIDs: {existing}). Restart? [y/N] ")).strip().lower()
        if ans != "y":
            ok("Nothing to do.")
            return
        stop()

    start(binary, config, listen, log)


def cmd_restart(cfg: dict) -> None:
    from .process import start, stop

    banner("llama-swap-live — Restart")
    stop()
    start(
        expand(cfg["llama-swap-bin"]) / "llama-swap",
        expand(cfg["swap-config"]),
        cfg["listen"],
        expand(cfg["log-file"]),
    )


def cmd_status(cfg: dict) -> None:
    from .process import status

    banner("llama-swap-live — Status")
    status(expand(cfg["llama-swap-bin"]) / "llama-swap")


def cmd_update(cfg: dict, check_only: bool) -> None:
    from .process import start, stop
    from .updater import update

    banner("llama-swap-live — Update")
    install_dir = expand(cfg["llama-swap-bin"])
    installed   = update(install_dir, check_only=check_only)

    if installed:
        from .colors import step
        step("Restarting llama-swap…")
        stop()
        start(
            install_dir / "llama-swap",
            expand(cfg["swap-config"]),
            cfg["listen"],
            expand(cfg["log-file"]),
        )


def cmd_remove(cfg: dict, no_confirm: bool = False) -> None:
    from .remover import cmd_remove as _rm
    _rm(expand(cfg["swap-config"]), no_confirm=no_confirm)


def cmd_add(cfg: dict, args: argparse.Namespace) -> None:
    from .buckets import ask_bucket, detect
    from .downloader import download
    from .swapconfig import inject

    banner("llama-swap-live — Add Model")

    repo_id      = args.repo
    quant        = args.quant
    multimodal   = args.mm
    mmproj_quant = args.mmproj_quant
    force_bucket = args.size_bucket
    force_macro  = args.macro
    display_name = args.name
    engine       = _normalize_engine(args.engine)

    model_root = expand(cfg["model-root"])
    swap_cfg   = expand(cfg["swap-config"])

    from .colors import cyan, bold, step, info
    print(f"\n    Repo   : {cyan(repo_id)}")
    if quant:
        print(f"    Quant  : {cyan(quant)}")
    print(f"    Multi  : {cyan('yes' if multimodal else 'no')}")
    print(f"    Engine : {cyan(engine)}")

    # Resolve size bucket
    if force_bucket:
        bucket = force_bucket
        info(f"Using forced bucket: {bold(bucket)}")
    else:
        bucket = detect(repo_id)
        if not bucket:
            bucket = ask_bucket()

    author     = repo_id.split("/")[0]
    model_name = repo_id.split("/")[-1]

    print(f"    Bucket : {cyan(bucket)}")

    if engine == "rapid-mlx":
        # Rapid-MLX manages its own downloads — just inject the config entry
        step("Skipping download (Rapid-MLX manages its own models)")
        inject(
            swap_config_path=swap_cfg,
            repo_id=repo_id,
            quant=None,
            bucket=bucket,
            gguf_path=None,
            mmproj_path=None,
            macro=force_macro,
            display_name=display_name,
            engine=engine,
        )
    else:
        dest_dir = model_root / bucket / author / model_name
        print(f"    Dest   : {cyan(str(dest_dir))}")

        gguf_path, mmproj_path = download(
            repo_id, quant, dest_dir, multimodal, mmproj_quant
        )

        step("Downloaded files:")

        for f in sorted(dest_dir.iterdir()):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name}  ({size_mb:.0f} MB)")

        if gguf_path:
            inject(
                swap_config_path=swap_cfg,
                repo_id=repo_id,
                quant=quant,
                bucket=bucket,
                gguf_path=gguf_path,
                mmproj_path=mmproj_path,
                macro=force_macro,
                display_name=display_name,
                engine=engine,
            )
        else:
            warn("No GGUF found — skipping config update")

    print()
    ok("Done.")


def _normalize_engine(raw: str) -> str:
    """Normalise engine aliases to canonical form: llama or rapid-mlx."""
    if raw in ("rapid-mlx", "mlx"):
        return "rapid-mlx"
    return "llama"


# ── argument parser ────────────────────────────────────────────────────────────

EPILOG = """
Commands:
  (default)     Start llama-swap in the background
  --status      Show running process and installed version
  --restart     Gracefully restart the process
  --rm          Interactively select and remove models (config + disk)
  --update      Download latest binary from GitHub and restart
  --add         Download a HuggingFace model and register it

Examples:
  llama-swap-live
  llama-swap-live --status
  llama-swap-live --restart
  llama-swap-live --update
  llama-swap-live --update --check
  llama-swap-live --rm

  llama-swap-live --add -r bartowski/Qwen3-30B-A3B-GGUF -q Q6_K_L
  llama-swap-live --add -r mradermacher/gemma-4-31B-GGUF -q Q6_K --mm
  llama-swap-live --add -r llmfan46/gemma-4-GGUF -q Q6_K --mm --mmproj-quant BF16
  llama-swap-live --add -r some/35B-MoE-GGUF -q Q5_K --size-bucket 32B-48B
  llama-swap-live --add -r mlx-community/Qwen3.5-27B-4bit -e rapid-mlx
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
        help=f"Manager config (default: {DEFAULT_CONFIG_PATH})",
    )

    # ── mode flags (mutually exclusive) ───────────────────────────────────────
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--add",     action="store_true", help="Download & register a HuggingFace model")
    mode.add_argument("--rm",      action="store_true", help="Interactively remove models from config + disk")
    mode.add_argument("--update",  action="store_true", help="Update llama-swap binary from GitHub")
    mode.add_argument("--restart", action="store_true", help="Restart llama-swap process")
    mode.add_argument("--status",  action="store_true", help="Show process status and binary version")

    # ── --add options ──────────────────────────────────────────────────────────
    g = p.add_argument_group("--add options")
    g.add_argument("-r", "--repo",        metavar="REPO",   help="HuggingFace repo ID (owner/name)")
    g.add_argument("-q", "--quant",       metavar="QUANT",  help="Quantisation filter e.g. Q6_K_L")
    g.add_argument("--mm",                action="store_true", help="Multi-modal: also download mmproj file")
    g.add_argument("--mmproj-quant",      metavar="Q",      help="mmproj quant preference (default: BF16)")
    g.add_argument("--size-bucket",       metavar="BUCKET", help="Force size bucket e.g. 20B-31B")
    g.add_argument("--macro",             metavar="MACRO",  help="Override macro name (e.g. llama-28k or rapid-mlx-28k)")
    g.add_argument("--name",              metavar="NAME",   help="Override display name in config")
    g.add_argument("-e", "--engine",      metavar="ENGINE",
                   choices=["llama", "llama-server", "rapid-mlx", "mlx"],
                   default="llama",
                   help="Target inference engine: llama (llama-server) or rapid-mlx (mlx)")

    # ── --rm options ───────────────────────────────────────────────────────────
    r = p.add_argument_group("--rm options")
    r.add_argument(
        "--no-confirm", action="store_true",
        help="Skip the 'yes' confirmation prompt (use with care in scripts)",
    )

    # ── --update options ───────────────────────────────────────────────────────
    u = p.add_argument_group("--update options")
    u.add_argument("--check", action="store_true", help="Check for a newer version without installing")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    cfg = load(Path(args.config))

    if args.add:
        if not args.repo:
            parser.error("--add requires -r/--repo")
        cmd_add(cfg, args)
    elif args.rm:
        cmd_remove(cfg, no_confirm=args.no_confirm)
    elif args.update:
        cmd_update(cfg, check_only=args.check)
    elif args.restart:
        cmd_restart(cfg)
    elif args.status:
        cmd_status(cfg)
    else:
        cmd_start(cfg)


if __name__ == "__main__":
    main()
