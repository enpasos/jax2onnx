# tests/extra_tests/converter/test_conversion_primitive_recording.py

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from jax2onnx.converter import conversion_api


def _add_one(x):
    return x + 1


def test_record_primitive_calls_file_writes_lowering_records(tmp_path) -> None:
    log_path = tmp_path / "primitive_calls.json"

    conversion_api.to_onnx(
        fn=_add_one,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_params=None,
        model_name="primitive_calls",
        opset=21,
        enable_double_precision=False,
        record_primitive_calls_file=str(log_path),
    )

    records = json.loads(log_path.read_text(encoding="utf-8"))
    assert records
    add_record = next(record for record in records if record["primitive_name"] == "add")
    assert add_record["sequence_id"] >= 0
    assert add_record["plugin_file_hint"]
    assert add_record["inputs_aval"]
    assert add_record["outputs_aval"]
    assert add_record["inputs_jax_vars"]
    assert add_record["outputs_jax_vars"]
    assert add_record["outputs_onnx_names"]
