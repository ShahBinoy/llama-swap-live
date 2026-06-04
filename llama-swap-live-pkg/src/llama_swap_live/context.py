"""
Context window estimation for llama-server models.

Given:
  - param_count       : total model parameters (int)
  - quant             : quantisation string e.g. "Q6_K_L", "Q8_0", "IQ4_XS"
  - kv_quant_bits     : KV cache quantisation bits from lsl.yaml (e.g. 8)
  - gpu_capacity_gb   : VRAM available from lsl.yaml (e.g. 32)
  - arch              : model architecture dict from HF config.json
                        (num_hidden_layers, num_key_value_heads, head_dim or
                         hidden_size/num_attention_heads)

Returns:
  - max_ctx           : estimated maximum safe context in tokens
  - selected_macro    : the tightest llama-XXk macro that fits
  - model_weights_gb  : estimated VRAM consumed by weights
  - kv_budget_gb      : VRAM left for KV cache

Formula:
  model_weights_gb  = params * bits_per_weight / 8 / 1e9
  kv_per_token_gb   = 2 * layers * kv_heads * head_dim * kv_bits / 8 / 1e9
  kv_budget_gb      = gpu_capacity_gb - model_weights_gb - HEADROOM_GB
  max_ctx           = kv_budget_gb / kv_per_token_gb
"""
from __future__ import annotations

import re
from typing import Optional

# ── Quant bits-per-weight table ────────────────────────────────────────────────
# Keys are prefix patterns matched against the uppercased quant string.
# Values are (bits_per_weight: float).
# Ordered from most specific to least specific.
_QUANT_BITS: list[tuple[str, float]] = [
    # Integer quants
    ("IQ1",   1.56),
    ("IQ2",   2.31),
    ("IQ3",   3.35),
    ("IQ4",   4.25),
    ("Q2_K",  2.63),
    ("Q3_K",  3.35),
    ("Q4_0",  4.50),
    ("Q4_1",  4.50),
    ("Q4_K",  4.50),   # Q4_K_S, Q4_K_M, Q4_K_L
    ("Q5_0",  5.00),
    ("Q5_1",  5.00),
    ("Q5_K",  5.50),   # Q5_K_S, Q5_K_M, Q5_K_P
    ("Q6_K",  6.50),
    ("Q8_0",  8.00),
    ("Q8_K",  8.00),
    ("BF16",  16.0),
    ("F16",   16.0),
    ("F32",   32.0),
]

# Context sizes available as llama-swap macros, in ascending order
# Format: (ctx_tokens, macro_name)
_MACROS: list[tuple[int, str]] = [
    (16_384,  "llama-16k"),
    (20_480,  "llama-20k"),
    (22_528,  "llama-22k"),
    (24_576,  "llama-24k"),
    (28_672,  "llama-28k"),
    (32_768,  "llama-32k"),
    (49_152,  "llama-48k"),
    (65_536,  "llama-64k"),
    (81_920,  "llama-80k"),
    (98_304,  "llama-96k"),
    (131_072, "llama-128k"),
]

# Safety headroom reserved for CUDA/Metal runtime, activations, etc.
_HEADROOM_GB = 1.0


def quant_bits(quant: Optional[str]) -> float:
    """Return bits-per-weight for a quant string. Defaults to 6.5 if unknown."""
    if not quant:
        return 6.5
    q = quant.upper().strip()
    for prefix, bits in _QUANT_BITS:
        if q.startswith(prefix):
            return bits
    # Fallback: try to parse a leading number e.g. "4bit" → 4.0
    m = re.match(r"(\d+(?:\.\d+)?)", q)
    if m:
        return float(m.group(1))
    return 6.5   # safe-ish default


def _head_dim_from_arch(arch: dict) -> Optional[int]:
    """
    Derive per-head dimension from the HF config dict.
    Most models expose head_dim directly; others need hidden_size / num_heads.
    """
    # Direct field (Gemma, Qwen3, newer HF configs)
    if "head_dim" in arch:
        return int(arch["head_dim"])
    hs = arch.get("hidden_size")
    nh = arch.get("num_attention_heads")
    if hs and nh:
        return int(hs) // int(nh)
    return None


def estimate_context(
    param_count:    int,
    quant:          Optional[str],
    kv_quant_bits:  int,
    gpu_capacity_gb: float,
    arch:           dict,
) -> dict:
    """
    Estimate safe context window and return a result dict with all workings.

    Returns:
        {
          "bits_per_weight":  float,
          "model_weights_gb": float,
          "kv_budget_gb":     float,
          "max_ctx":          int,
          "selected_macro":   str,
          "macro_ctx":        int,    # actual ctx of selected macro
          "layers":           int | None,
          "kv_heads":         int | None,
          "head_dim":         int | None,
          "warning":          str | None,  # set when arch info was incomplete
        }
    """
    bpw = quant_bits(quant)
    model_weights_gb = (param_count * bpw) / 8.0 / 1e9

    kv_budget_gb = gpu_capacity_gb - model_weights_gb - _HEADROOM_GB

    layers   = arch.get("num_hidden_layers")
    kv_heads = arch.get("num_key_value_heads") or arch.get("num_attention_heads")
    head_dim = _head_dim_from_arch(arch)

    warning: Optional[str] = None
    max_ctx = 0

    if layers and kv_heads and head_dim and kv_budget_gb > 0:
        # Each KV token: 2 (K+V) × layers × kv_heads × head_dim × kv_bits/8 bytes
        kv_bytes_per_token = 2 * layers * kv_heads * head_dim * (kv_quant_bits / 8.0)
        kv_budget_bytes = kv_budget_gb * 1e9
        max_ctx = int(kv_budget_bytes / kv_bytes_per_token)
    elif kv_budget_gb <= 0:
        warning = "Model weights exceed GPU capacity — no VRAM left for KV cache"
        max_ctx = 0
    else:
        # Architecture info incomplete — fall back to a rough heuristic:
        # assume ~0.5 GB per 1K tokens at kv_quant=8 for a 27B model scale
        scale = (param_count / 27e9) ** 0.5
        kv_gb_per_1k = 0.5 * scale * (kv_quant_bits / 8.0)
        max_ctx = int((kv_budget_gb / kv_gb_per_1k) * 1000) if kv_gb_per_1k > 0 else 0
        warning = "Incomplete architecture info — using heuristic estimate"

    # Select the largest macro whose ctx fits within max_ctx
    selected_macro = _MACROS[0][1]
    macro_ctx      = _MACROS[0][0]
    for ctx_tokens, macro_name in _MACROS:
        if ctx_tokens <= max_ctx:
            selected_macro = macro_name
            macro_ctx      = ctx_tokens
        else:
            break

    return {
        "bits_per_weight":  bpw,
        "model_weights_gb": round(model_weights_gb, 2),
        "kv_budget_gb":     round(max(kv_budget_gb, 0.0), 2),
        "max_ctx":          max_ctx,
        "selected_macro":   selected_macro,
        "macro_ctx":        macro_ctx,
        "layers":           layers,
        "kv_heads":         kv_heads,
        "head_dim":         head_dim,
        "warning":          warning,
    }


def print_estimate(result: dict, gpu_capacity_gb: float) -> None:
    """Pretty-print the context estimation workings."""
    from ..colors import bold, cyan, green, yellow, warn as _warn

    print()
    print(bold("  Context Estimation"))
    gpu_str  = cyan(f"{gpu_capacity_gb:.0f} GB")
    bpw_str  = cyan("{:.2f}".format(result["bits_per_weight"]))
    wgt_str  = cyan("{:.2f} GB".format(result["model_weights_gb"]))
    kv_str   = cyan("{:.2f} GB".format(result["kv_budget_gb"]))
    ctx_str  = green("{:,} tokens".format(result["max_ctx"]))
    mac_str  = green(result["selected_macro"])
    mac_ctx  = result["macro_ctx"] // 1024
    print(f"    GPU capacity       : {gpu_str}")
    print(f"    Bits per weight    : {bpw_str}")
    print(f"    Model weights      : {wgt_str}")
    print(f"    KV budget          : {kv_str}")
    if result["layers"]:
        lkh = cyan("{} / {} / {}".format(result["layers"], result["kv_heads"], result["head_dim"]))
        print(f"    Layers/KV-heads/dim: {lkh}")
    print(f"    Max context        : {ctx_str}")
    print(f"    Selected macro     : {mac_str}  ({mac_ctx}K tokens)")
    if result["warning"]:
        print(f"    {yellow('⚠  ' + result['warning'])}")
    print()
