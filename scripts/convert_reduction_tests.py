#!/usr/bin/env python
# scripts/convert_reduction_tests.py

"""One-off helper to port missing reduction testcases into converter plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from jax2onnx.plugins.jax.lax._reduction_registry import REDUCTION_TESTS

TARGETS: Dict[str, str] = {
    "reduce_max": "jax2onnx/plugins/jax/lax/reduce_max.py",
    "reduce_min": "jax2onnx/plugins/jax/lax/reduce_min.py",
    "reduce_or": "jax2onnx/plugins/jax/lax/reduce_or.py",
    "reduce_and": "jax2onnx/plugins/jax/lax/reduce_and.py",
    "reduce_prod": "jax2onnx/plugins/jax/lax/reduce_prod.py",
    "reduce_sum": "jax2onnx/plugins/jax/lax/reduce_sum.py",
    "reduce_xor": "jax2onnx/plugins/jax/lax/reduce_xor.py",
}


def format_testcase(spec) -> str:
    lines: List[str] = ["        {"]
    lines.append(f'            "testcase": "{spec.testcase}",')
    if spec.values is not None:
        lines.append(
            '            "input_values": [' + repr(spec.values.tolist()) + "],"
        )
    elif spec.dtype is not None:
        lines.append(
            '            "input_shapes": [(2, 3)],'  # placeholder; edit manually if needed
        )
    else:
        lines.append('            "input_shapes": [(3, 3)],')
    lines.append('            "": True,')
    lines.append("        },")
    return "\n".join(lines)


def main() -> None:
    for comp, path in TARGETS.items():
        specs = REDUCTION_TESTS.get(comp, [])
        if not specs:
            continue
        tc_lines = ["    testcases=["]
        for spec in specs:
            tc_lines.append(format_testcase(spec))
        tc_lines.append("    ],")
        block = "\n".join(tc_lines)
        file = Path(path)
        text = file.read_text()
        new_text = text.replace("    testcases=[", block)
        file.write_text(new_text)


if __name__ == "__main__":
    main()
