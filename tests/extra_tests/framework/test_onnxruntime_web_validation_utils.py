# tests/extra_tests/framework/test_onnxruntime_web_validation_utils.py

from pathlib import Path
import shutil
import subprocess

import pytest


def test_float16_comparison_handles_numeric_and_bit_typed_arrays() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for onnxruntime-web validation")

    repo_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [node, "--input-type=module"],
        cwd=repo_root,
        input="""
import assert from "node:assert/strict";
import {
  compareTensor,
  float16BitsToNumber,
} from "./scripts/onnxruntime_web_validation_utils.mjs";

// Native Float16Array iterates over numbers, unlike the Uint16Array fallback.
// Model that behavior on older Node versions without Float16Array support.
class NumericFloat16Array extends Uint16Array {
  *[Symbol.iterator]() {
    for (const bits of super.values()) {
      yield float16BitsToNumber(bits);
    }
  }
}

const bits = [0x2e7f, 0xbc00, 0x0001, 0x8000, 0x7c00, 0xfc00, 0x7e00];
const expected = {type: "float16", dims: [bits.length], data: bits};
const paddedBits = Uint16Array.from([0x4500, ...bits, 0x4600]);
const actualData = [
  paddedBits.subarray(1, -1),
  new NumericFloat16Array(paddedBits.buffer, 2, bits.length),
];
if (typeof Float16Array !== "undefined") {
  actualData.push(new Float16Array(paddedBits.buffer, 2, bits.length));
}
for (const data of actualData) {
  const actual = {type: "float16", dims: [bits.length], data};
  assert.equal(compareTensor("fp16", actual, expected, 0, 0), 0);
  const wrong = {...expected, data: [0, ...bits.slice(1)]};
  assert.throws(
    () => compareTensor("fp16", actual, wrong, 0, 0),
    /mismatch at flat index 0/,
  );
}
""",
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
