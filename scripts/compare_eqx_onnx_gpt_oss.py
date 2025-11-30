#!/usr/bin/env python3
# scripts/compare_eqx_onnx_gpt_oss.py

"""Compare Equinox GPT-OSS logits against an exported ONNX (single prompt, fixed seq)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

# Default to CPU to avoid GPU OOM during quick parity checks.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

from jax2onnx.plugins.examples.eqx.gpt_oss import GPTOSSConfig, Transformer


def _load_tokenizer():
    try:
        from gpt_oss import tokenizer as gpt_tokenizer  # type: ignore

        enc = gpt_tokenizer.get_tokenizer()

        def encode(text: str) -> List[int]:
            return enc.encode(text)

        def decode(token_ids: Iterable[int]) -> str:
            return enc.decode(list(token_ids))

        return encode, decode
    except Exception:
        pass

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def encode(text: str) -> List[int]:
            return enc.encode(text)

        def decode(token_ids: Iterable[int]) -> str:
            return enc.decode(list(token_ids))

        return encode, decode
    except Exception:

        def encode(text: str) -> List[int]:
            return [ord(ch) for ch in text]

        def decode(token_ids: Iterable[int]) -> str:
            chars = []
            for tid in token_ids:
                val = tid % 256
                chars.append(chr(val) if 32 <= val <= 126 else f"<0x{val:02x}>")
            return "".join(chars)

        return encode, decode


def _read_meta(eqx_path: Path) -> dict:
    meta_path = eqx_path.with_suffix(eqx_path.suffix + ".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata file missing for Equinox archive: {meta_path}"
        )
    return json.loads(meta_path.read_text())


def _ort_dtype_from_type(type_str: str) -> np.dtype | None:
    if not type_str.startswith("tensor(") or not type_str.endswith(")"):
        return None
    t = type_str[7:-1]
    mapping = {
        "float": np.float32,
        "float16": np.float16,
        "float64": np.float64,
        "double": np.float64,
        "bfloat16": np.float32,
        "int64": np.int64,
        "int32": np.int32,
        "int16": np.int16,
        "int8": np.int8,
        "uint8": np.uint8,
    }
    return mapping.get(t)


DEBUG_KEYS = [
    "input",
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_out",
    "mlp_norm",
    "gate_logits",
    "expert_indices",
    "expert_weights",
    "mlp_proj1",
    "mlp_act",
    "mlp_proj2",
    "mlp_output",
    "output",
]


def _load_eqx(eqx_path: Path, meta: dict) -> Transformer:
    config_data = meta.get("config")
    if not isinstance(config_data, dict):
        raise ValueError(f"Invalid config metadata in {eqx_path}.meta.json")
    config = GPTOSSConfig(**config_data)
    param_dtype_key = str(meta.get("param_dtype", "bfloat16"))
    dtype_map = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}
    if param_dtype_key not in dtype_map:
        raise ValueError(f"Unsupported param dtype '{param_dtype_key}' in metadata")
    param_dtype = dtype_map[param_dtype_key]
    seed_value = int(meta.get("seed", 0))
    template = Transformer(
        config=config,
        key=jax.random.PRNGKey(seed_value),
        param_dtype=param_dtype,
    )
    model = eqx.tree_deserialise_leaves(eqx_path, template, is_leaf=lambda _: True)
    return model


def _run_eqx(
    model: Transformer, tokens: np.ndarray
) -> Tuple[np.ndarray, Optional[List[dict]]]:
    logits, debug = model.debug(jnp.asarray(tokens))
    logits = np.asarray(jax.device_get(logits), dtype=np.float32)
    debug_np: List[dict] = []
    for entry in debug:
        debug_np.append({k: np.asarray(jax.device_get(v)) for k, v in entry.items()})
    return logits, debug_np


def _topo_sort_custom_functions(model: onnx.ModelProto) -> None:
    """Reorder embedded FunctionProtos so dependencies appear before callers."""

    funcs = list(model.functions)

    def key_for(f):
        return (f.domain, f.name)

    deps: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for f in funcs:
        k = key_for(f)
        deps[k] = {
            (n.domain, n.op_type)
            for n in f.node
            if n.domain and n.domain != "" and n.domain.startswith("custom.")
        }

    ordered: list[onnx.FunctionProto] = []
    placed: set[tuple[str, str]] = set()
    remaining = {key_for(f): f for f in funcs}

    # Simple topo sort; fall back to remaining order if a cycle is found.
    while remaining:
        ready = [
            k for k, dep in deps.items() if dep.issubset(placed) and k in remaining
        ]
        if not ready:
            ordered.extend(remaining.values())
            break
        for k in ready:
            ordered.append(remaining.pop(k))
            placed.add(k)

    del model.functions[:]
    model.functions.extend(ordered)


def _parse_onnx_debug(outputs: List[np.ndarray]) -> Tuple[np.ndarray, List[dict]]:
    logits = np.asarray(outputs[0], dtype=np.float32)
    debug_blocks: List[dict] = []
    if len(outputs) > 1:
        num_blocks = (len(outputs) - 1) // len(DEBUG_KEYS)
        idx = 1
        for _ in range(num_blocks):
            dbg: dict = {}
            for key in DEBUG_KEYS:
                arr = np.asarray(outputs[idx])
                # Drop leading batch axis if it is size 1 to align with Equinox captures.
                if arr.shape and arr.shape[0] == 1:
                    arr = arr[0]
                dbg[key] = arr
                idx += 1
            debug_blocks.append(dbg)
    return logits, debug_blocks


def _run_onnx_reference(
    onnx_path: Path, tokens: np.ndarray
) -> Tuple[np.ndarray, List[dict]]:
    model = onnx.load(str(onnx_path))
    _topo_sort_custom_functions(model)
    session = ReferenceEvaluator(model)
    input_name = session.input_names[0]
    outputs = session.run(None, {input_name: tokens})
    return _parse_onnx_debug(outputs)


def _run_onnx_ort(onnx_path: Path, tokens: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed")
    sess_opts = ort.SessionOptions()
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )
    meta = session.get_inputs()[0]
    input_name = meta.name
    input_dtype = _ort_dtype_from_type(getattr(meta, "type", ""))
    input_shape = list(getattr(meta, "shape", []) or [])
    feed = tokens
    if input_dtype is not None and feed.dtype != input_dtype:
        feed = feed.astype(input_dtype, copy=False)
    if len(feed.shape) == 1 and len(input_shape) == 2:
        feed = feed[None, :]
    outputs = session.run(None, {input_name: feed})
    return _parse_onnx_debug(outputs)


def _diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    delta = np.abs(a - b)
    return float(delta.max()), float(delta.mean()), float(np.median(delta))


def _diff_debug(
    eqx_blocks: List[dict], onnx_blocks: List[dict]
) -> Tuple[List[Tuple[int, str, float, float, float]], List[str]]:
    diffs: List[Tuple[int, str, float, float, float]] = []
    issues: List[str] = []
    n = min(len(eqx_blocks), len(onnx_blocks))
    for i in range(n):
        e = eqx_blocks[i]
        o = onnx_blocks[i]
        for key in DEBUG_KEYS:
            if key not in e or key not in o:
                issues.append(
                    f"block{i}.{key} missing (eqx:{key in e}, onnx:{key in o})"
                )
                continue
            if e[key].shape != o[key].shape:
                issues.append(
                    f"block{i}.{key} shape mismatch eqx{e[key].shape} vs onnx{o[key].shape}"
                )
                continue
            mx, mn, md = _diff(o[key], e[key])
            diffs.append((i, key, mx, mn, md))
    return diffs, issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eqx",
        required=True,
        type=Path,
        help="Path to Equinox .eqx file (with .meta.json).",
    )
    parser.add_argument("--onnx", required=True, type=Path, help="Path to ONNX file.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="GPT-OSS config JSON (for vocab size).",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=32,
        help="Sequence length to pad/truncate to.",
    )
    parser.add_argument(
        "--runtime",
        choices=("ort", "reference"),
        default="ort",
        help="Backend for ONNX execution (default: ort).",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help=(
            "Trim the Equinox model to the first N layers (helps when the Equinox archive "
            "contains more layers than the ONNX export)."
        ),
    )
    parser.add_argument(
        "--trace-block",
        type=int,
        default=None,
        help="If set, print a step-by-step divergence trace for this specific block index.",
    )
    parser.add_argument(
        "--emit-debug",
        action="store_true",
        help="Expect debug outputs in the ONNX and compare block-level tensors.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    encode, _decode = _load_tokenizer()
    tokens = encode(args.prompt)
    seq_len = int(args.sequence_length)
    if len(tokens) == 0:
        tokens = [0]
    window = tokens[:seq_len]
    padded = np.zeros((seq_len,), dtype=np.int32)
    padded[: len(window)] = np.array(window, dtype=np.int32)[:seq_len]

    meta = _read_meta(args.eqx.expanduser())
    eqx_model = _load_eqx(args.eqx.expanduser(), meta)
    if args.max_layers is not None:
        print(
            f"Trimming Equinox model from {len(eqx_model.blocks)} to {int(args.max_layers)} layers."
        )
        eqx_model = eqx.tree_at(
            lambda m: m.blocks, eqx_model, eqx_model.blocks[: int(args.max_layers)]
        )
    eqx_logits, eqx_debug = _run_eqx(eqx_model, padded)

    if args.runtime == "reference":
        onnx_logits, onnx_debug = _run_onnx_reference(args.onnx.expanduser(), padded)
    else:
        onnx_logits, onnx_debug = _run_onnx_ort(args.onnx.expanduser(), padded)

    # Align shapes if batch dim present
    if onnx_logits.ndim == 3 and onnx_logits.shape[0] == 1:
        onnx_logits = onnx_logits[0]
    if eqx_logits.ndim == 3 and eqx_logits.shape[0] == 1:
        eqx_logits = eqx_logits[0]

    max_diff, mean_diff, median_diff = _diff(onnx_logits, eqx_logits)
    last_idx = min(len(window) - 1, onnx_logits.shape[0] - 1, eqx_logits.shape[0] - 1)
    last_max, last_mean, last_med = _diff(onnx_logits[last_idx], eqx_logits[last_idx])

    print(f"Prompt tokens (len={len(window)}): {window}")
    print(f"Logits shape ONNX: {onnx_logits.shape}, Equinox: {eqx_logits.shape}")
    print(
        f"Diff all tokens: max={max_diff:.4e}, mean={mean_diff:.4e}, median={median_diff:.4e}"
    )
    print(
        f"Diff last token: max={last_max:.4e}, mean={last_mean:.4e}, median={last_med:.4e}"
    )
    if args.emit_debug:
        print(f"Debug blocks ONNX: {len(onnx_debug)}, Equinox: {len(eqx_debug)}")
        diffs, issues = _diff_debug(eqx_debug, onnx_debug)
        diffs = sorted(diffs, key=lambda t: t[2], reverse=True)
        if args.trace_block is not None:
            idx = int(args.trace_block)
            print(f"\n--- Trace for Block {idx} ---")
            block_diffs = [d for d in diffs if d[0] == idx]
            diff_map = {d[1]: d for d in block_diffs}
            print(f"{'Key':<20} | {'Max Diff':<12} | {'Mean Diff':<12}")
            print("-" * 50)
            for key in DEBUG_KEYS:
                if key in diff_map:
                    _, _, mx, mn, _ = diff_map[key]
                    print(f"{key:<20} | {mx:<12.4e} | {mn:<12.4e}")
            print("-" * 50 + "\n")
        print("Top debug diffs (block, key, max, mean, median):")
        if not diffs:
            print("  (none captured)")
        for blk, key, mx, mn, md in diffs[:10]:
            print(f"  block{blk}.{key}: max={mx:.4e}, mean={mn:.4e}, median={md:.4e}")
        if issues:
            print("Debug issues:")
            for msg in issues[:10]:
                print(f"  {msg}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
