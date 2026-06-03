"""Terminal colour helpers."""
from __future__ import annotations
import sys

IS_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if IS_TTY else text

def red(t: str) -> str:    return _c("0;31", t)
def green(t: str) -> str:  return _c("0;32", t)
def yellow(t: str) -> str: return _c("1;33", t)
def blue(t: str) -> str:   return _c("0;34", t)
def cyan(t: str) -> str:   return _c("0;36", t)
def bold(t: str) -> str:   return _c("1", t)

def info(msg: str)  -> None: print(blue(f"  ℹ {msg}"))
def ok(msg: str)    -> None: print(green(f"  ✓ {msg}"))
def warn(msg: str)  -> None: print(yellow(f"  ⚠ {msg}"), file=sys.stderr)
def err(msg: str)   -> None: print(red(f"  ✗ {msg}"), file=sys.stderr)
def step(msg: str)  -> None: print(bold(f"\n▶ {msg}"))
def die(msg: str, code: int = 1) -> None:
    err(msg)
    sys.exit(code)

def banner(title: str) -> None:
    w = 46
    print(bold(blue("═" * w)))
    print(bold(blue(f"  {title}")))
    print(bold(blue("═" * w)))
