#!/usr/bin/env python3
"""Run an exported GPT-OSS ONNX model without Flax parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import onnx
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().read_text())
    vocab_size = int(config["vocab_size"])
    seq_len = int(args.sequence_length)
    encode, decode = _load_tokenizer()
    tokens = encode(args.prompt)
    generated: List[int] = []

    model = onnx.load(str(args.onnx.expanduser().resolve()))
    session = ReferenceEvaluator(model)

    for _ in range(max(1, args.generate_steps)):
        if len(tokens) == 0:
            tokens.append(0)
        window = tokens[-seq_len:]
        padded = np.zeros((seq_len,), dtype=np.int32)
        padded[: len(window)] = np.array(window, dtype=np.int32)[:seq_len]
        logits = session.run(None, {session.input_names[0]: padded})[0]
        last_index = len(window) - 1
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
