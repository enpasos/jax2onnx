#!/usr/bin/env python3
# scripts/run_onnx_only.py

"""Run an exported GPT-OSS ONNX model without Flax parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import onnx
from onnx import compose
from onnx.reference import ReferenceEvaluator


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
            ids = list(token_ids)
            try:
                return enc.decode(ids)
            except KeyError:
                fallback = []
                for tid in ids:
                    val = tid % 256
                    if 32 <= val <= 126:
                        fallback.append(chr(val))
                    else:
                        fallback.append(f"<0x{val:02x}>")
                return "".join(fallback)

        return encode, decode
    except Exception:

        def encode(text: str) -> List[int]:
            return [ord(ch) for ch in text]

        def decode(token_ids: Iterable[int]) -> str:
            chars = []
            for tid in token_ids:
                val = tid % 256
                if 32 <= val <= 126:
                    chars.append(chr(val))
                else:
                    chars.append(f"<0x{val:02x}>")
            return "".join(chars)

        return encode, decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX GPT-OSS model only.")
    parser.add_argument("--onnx", required=True, type=Path, help="Path to .onnx file")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="JSON config (used for vocab_size/tokenizer fallback).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt to tokenize (simple byte-level fallback).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=32,
        help="Sequence length to pad/truncate to.",
    )
    parser.add_argument(
        "--generate-steps",
        type=int,
        default=64,
        help="Number of autoregressive tokens to generate (default: 64).",
    )
    parser.add_argument(
        "--expand-functions",
        action="store_true",
        help=(
            "Inline custom functions (TokenEmbedding, TransformerBlock, etc.) into the main graph. "
            "Useful when a runtime cannot resolve custom domains in-place."
        ),
    )
    parser.add_argument(
        "--runtime",
        choices=["reference", "ort"],
        default="reference",
        help="Backend to execute the ONNX model. 'ort' (onnxruntime) is faster and more memory efficient.",
    )
    return parser.parse_args()


def _ort_dtype_from_type(type_str: str) -> np.dtype | None:
    if not type_str.startswith("tensor(") or not type_str.endswith(")"):
        return None
    t = type_str[7:-1]
    mapping = {
        "float": np.float32,
        "float16": np.float16,
        "float64": np.float64,
        "double": np.float64,
        "bfloat16": np.float32,  # coerce to float32 for ORT feeds
        "int64": np.int64,
        "int32": np.int32,
        "int16": np.int16,
        "int8": np.int8,
        "uint8": np.uint8,
    }
    return mapping.get(t)


def _topo_sort_custom_functions(model: onnx.ModelProto) -> None:
    """Reorder embedded FunctionProtos so dependencies appear before callers."""

    funcs = list(model.functions)

    def key_for(f):
        return (f.domain, f.name)

    deps: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for f in funcs:
        k = key_for(f)
        deps[k] = {
            (n.domain, n.op_type) for n in f.node if n.domain.startswith("custom.")
        }

    ordered: list[onnx.FunctionProto] = []
    placed: set[tuple[str, str]] = set()
    remaining = {key_for(f): f for f in funcs}

    # Simple Kahn-style pass; fall back to insertion order if we hit a cycle.
    while remaining:
        ready = [
            k for k, dep in deps.items() if dep.issubset(placed) and k in remaining
        ]
        if not ready:
            # Cycle or unresolved dependency; append the rest as-is.
            ordered.extend(remaining.values())
            break
        for k in ready:
            ordered.append(remaining.pop(k))
            placed.add(k)

    del model.functions[:]
    model.functions.extend(ordered)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().read_text())
    vocab_size = int(config["vocab_size"])
    seq_len = int(args.sequence_length)
    encode, decode = _load_tokenizer()
    tokens = encode(args.prompt)
    generated: List[int] = []

    onnx_path = args.onnx.expanduser().resolve()
    if args.runtime == "reference":
        model = onnx.load(str(onnx_path))
        _topo_sort_custom_functions(model)
        if args.expand_functions:
            model = compose.expand_functions(model)
        # ReferenceEvaluator consumes custom FunctionProtos embedded in the model.
        session = ReferenceEvaluator(model)
        input_name = session.input_names[0]

        def run_inference(feed):
            return session.run(None, {input_name: feed})[0]

    else:
        # Prefer onnxruntime for large models; it handles external data efficiently.
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "onnxruntime is required for --runtime ort. Install with "
                "'poetry run pip install onnxruntime'"
            ) from exc
        sess_opts = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts, providers=providers
        )
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        input_dtype = _ort_dtype_from_type(getattr(input_meta, "type", ""))
        input_shape = list(getattr(input_meta, "shape", []) or [])

        def run_inference(feed):
            if input_dtype is not None and feed.dtype != input_dtype:
                feed = feed.astype(input_dtype, copy=False)
            # If the model expects a batch dimension but we provided 1D, add it.
            if len(feed.shape) == 1 and len(input_shape) == 2:
                feed = feed[None, :]
            return session.run(None, {input_name: feed})[0]

    for _ in range(max(1, args.generate_steps)):
        if len(tokens) == 0:
            tokens.append(0)
        window = tokens[-seq_len:]
        padded = np.zeros((seq_len,), dtype=np.int32)
        padded[: len(window)] = np.array(window, dtype=np.int32)[:seq_len]
        logits = run_inference(padded)
        # Handle outputs with or without batch/time axes.
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim == 2 and logits.shape[0] == 1:
            last_logits = logits[0]
        else:
            last_index = min(len(window) - 1, logits.shape[0] - 1)
            last_logits = logits[last_index]
        pred_id = int(np.argmax(last_logits))
        pred_id = pred_id % max(1, vocab_size)
        tokens.append(pred_id)
        generated.append(pred_id)
        print(f"Iteration logits (first 10): {last_logits[:10]}")
        print(f"Predicted token id: {pred_id}")

    decoded = decode(generated)
    print(f"Prompt: {args.prompt!r}")
    print(f"Generated tokens: {generated}")
    print(f"Decoded tokens: {decoded}")
    _print_plaintext(decoded)


def _print_plaintext(decoded: str) -> None:
    import json
    import re

    normalized = decoded
    if normalized.count("'") > normalized.count('"'):
        normalized = normalized.replace("'", '"')
    candidates = [normalized]
    for text in candidates:
        matches = re.findall(r'"text"\s*:\s*"([^"]+)"', text)
        if matches:
            print("Extracted text:")
            for line in matches:
                print(line)
            return
        try:
            data = json.loads(text)
            texts = _extract_text_from_json(data)
            if texts:
                print("Extracted text:")
                for line in texts:
                    print(line)
                return
        except Exception:
            continue
    print("No structured text segment found.")


def _extract_text_from_json(obj) -> List[str]:
    result: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "text" and isinstance(value, str):
                result.append(value)
            else:
                result.extend(_extract_text_from_json(value))
    elif isinstance(obj, list):
        for item in obj:
            result.extend(_extract_text_from_json(item))
    return result


if __name__ == "__main__":
    main()
