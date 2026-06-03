"""llama-swap-live — unified manager for llama-swap and HuggingFace model downloads."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("llama-swap-live")
except PackageNotFoundError:
    # Package not installed (e.g. running directly from source tree)
    __version__ = "0.0.0-dev"
