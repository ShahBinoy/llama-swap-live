"""Model parameter-size bucket detection and mapping."""
from __future__ import annotations

import re
from typing import Optional

from .colors import info, warn, cyan, bold, step

# (upper_bound_billions, bucket_label)
BUCKETS = [
    (8,    "7B-8B"),
    (13,   "10B-13B"),
    (19,   "14B-19B"),
    (31,   "20B-31B"),
    (48,   "32B-48B"),
    (71,   "48B-71B"),
    (9999, "72B-120B"),
]

BUCKET_MACRO: dict[str, str] = {
    "7B-8B":    "llama-32k",
    "10B-13B":  "llama-28k",
    "14B-19B":  "llama-28k",
    "20B-31B":  "llama-28k",
    "32B-48B":  "llama-16k",
    "48B-71B":  "llama-16k",
    "72B-120B": "llama-8k",
}

RAPID_MLX_BUCKET_MACRO: dict[str, str] = {
    "7B-8B":    "rapid-mlx-32k",
    "10B-13B":  "rapid-mlx-28k",
    "14B-19B":  "rapid-mlx-28k",
    "20B-31B":  "rapid-mlx-28k",
    "32B-48B":  "rapid-mlx-16k",
    "48B-71B":  "rapid-mlx-16k",
    "72B-120B": "rapid-mlx-8k",
}


def params_to_bucket(params: int) -> str:
    billions = params / 1e9
    for threshold, label in BUCKETS:
        if billions <= threshold:
            return label
    return "72B-120B"


def name_to_bucket(model_name: str) -> Optional[str]:
    """
    Parse parameter count from model name strings like:
      gemma-4-31B-it, Qwen3.6-27B, llama-3-70B-instruct, 35B-A3B (MoE)
    Returns bucket label or None.
    """
    patterns = [
        r"[-_.](\d+)[Bb](?:[^0-9A-Za-z]|$)",  # -27B, _8b, .6-27B
        r"^(\d+)[Bb][-_]",                      # 8B-v2  (leading)
        r"-(\d+)[Bb]$",                          # name-27B (trailing)
    ]
    for pat in patterns:
        m = re.search(pat, model_name)
        if m:
            n = int(m.group(1))
            for threshold, label in BUCKETS:
                if n <= threshold:
                    return label
    return None


def ask_bucket() -> str:
    """Interactive prompt when auto-detection fails."""
    labels = [label for _, label in BUCKETS]
    print(warn.__doc__ or "")  # silence
    print(f"\n  {bold('Could not determine model parameter size automatically.')}")
    print("  Please choose a size bucket:\n")
    for i, label in enumerate(labels, 1):
        print(f"    {bold(str(i))}) {label}")
    while True:
        try:
            choice = input(cyan("\n  Enter number [1-7]: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(1)
        if choice.isdigit() and 1 <= int(choice) <= len(labels):
            return labels[int(choice) - 1]
        print("  Invalid choice, please try again.")


def detect(repo_id: str) -> Optional[str]:
    """
    Multi-strategy detection:
      1. Parse model name (offline)
      2. HF API safetensors metadata
      3. HF API model tags
    Returns bucket label or None.
    """
    model_name = repo_id.split("/")[-1]

    bucket = name_to_bucket(model_name)
    if bucket:
        info(f"Detected size from model name → {bold(bucket)}")
        return bucket

    # Try HF API (optional dep)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        warn("huggingface_hub not installed — cannot query HF API for model size")
        return None

    step("Querying HuggingFace API for model size...")
    try:
        api = HfApi()
        model_info = api.model_info(repo_id, timeout=15)
    except Exception as e:
        warn(f"HF API error: {e}")
        return None

    # Strategy 2: safetensors parameter map
    params: Optional[int] = None
    source = ""
    try:
        st = getattr(model_info, "safetensors", None)
        if st:
            p = getattr(st, "parameters", None)
            if p and isinstance(p, dict):
                params = sum(p.values())
                source = "safetensors metadata"
    except Exception:
        pass

    # Strategy 3: tags like "7b", "27B"
    if not params:
        try:
            for tag in (getattr(model_info, "tags", None) or []):
                m = re.search(r"^(\d+(?:\.\d+)?)[Bb]$", str(tag))
                if m:
                    params = int(float(m.group(1)) * 1e9)
                    source = "model tags"
                    break
        except Exception:
            pass

    if params:
        bucket = params_to_bucket(params)
        info(f"Detected {params/1e9:.1f}B params via {source} → {bold(bucket)}")
        return bucket

    return None
