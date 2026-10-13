# tests/extra_tests/framework/test_normalization_benchmark.py

"""Benchmark reports must measure the requested dtype and distinguish startup."""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import pytest

from jax2onnx.plugins.examples.nnx.transformer_decoder_with_sequential import (
    TransformerDecoder,
)
from jax2onnx.plugins.plugin_system import construct_and_call, with_rng_seed
from scripts import benchmark_decoder_normalization as benchmark


@pytest.fixture(autouse=True)
def _restore_jax_settings() -> Iterator[None]:
    original_x64 = bool(jax.config.read("jax_enable_x64"))
    original_platforms = jax.config.jax_platforms
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", original_x64)
        jax.config.update("jax_platforms", original_platforms)


def test_decoder_dtype_configuration_preserves_defaults() -> None:
    def build(**kwargs: object) -> TransformerDecoder:
        return construct_and_call(
            TransformerDecoder,
            num_layers=1,
            embed_dim=16,
            num_heads=4,
            ff_dim=32,
            rngs=with_rng_seed(0),
            **kwargs,
        ).instantiate()

    default = build()
    explicit_default = build(dtype=None, param_dtype=jnp.float32)
    half = build(dtype=jnp.float16, param_dtype=jnp.float16)
    x = jnp.asarray(np.random.default_rng(1).normal(size=(1, 2, 16)), jnp.float32)
    np.testing.assert_array_equal(default(x, x), explicit_default(x, x))
    assert default(x, x).dtype == jnp.float32
    assert half(x.astype(jnp.float16), x.astype(jnp.float16)).dtype == jnp.float16
    for model, dtype in ((default, jnp.float32), (half, jnp.float16)):
        parameters = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
        assert parameters
        assert all(parameter.dtype == dtype for parameter in parameters)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("workload", ["decoder", "rmsnorm"])
def test_benchmark_runs_typed_static_and_symbolic_exports(
    dtype: str, workload: str, tmp_path: Path
) -> None:
    output = tmp_path / "report.json"
    benchmark.main(
        [
            "--dtype",
            dtype,
            "--workload",
            workload,
            "--embed-dim",
            "32",
            "--num-heads",
            "4",
            "--intra-op-threads",
            "2",
            "--batch-size",
            "1",
            "--decoder-length",
            "3",
            "--encoder-length",
            "2",
            "--iterations",
            "1",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["model"]["dtype"] == dtype
    assert report["model"]["reference_output_dtype"] == dtype
    assert report["environment"]["intra_op_num_threads"] == 2
    measurement = report["measurement"]
    assert measurement["reference"].startswith(f"JAX CPU {dtype}")
    assert measurement["error_calculation_dtype"] == "float64"
    assert measurement["relative_error_floor_dtype"] == dtype
    assert measurement["startup_samples_per_case"] == 1
    expected_tolerance = 2e-3 if dtype == "float16" else 1e-4
    assert measurement["parity_atol"] == expected_tolerance
    assert measurement["parity_rtol"] == expected_tolerance
    assert len(report["cases"]) == 4
    assert {case["shape_mode"] for case in report["cases"]} == {"static", "symbolic"}
    native_op = "LayerNormalization" if workload == "decoder" else "RMSNormalization"
    for case in report["cases"]:
        assert case["output_dtype"] == dtype
        assert case["session_initialization_ms"] > 0
        assert case["first_run_ms"] > 0
        assert case["median_ms"] == case["repeat_mean_ms"][0]
        assert case["median_ms"] > 0
        assert case["errors_against_jax"][
            "pointwise_relative_denominator_floor"
        ] >= float(np.finfo(dtype).eps)
        assert (case["exported_node_counts"].get(native_op, 0) > 0) == (
            case["normalization_mode"] == "prefer_native"
        )


def test_benchmark_rejects_incompatible_decoder_head_width() -> None:
    with pytest.raises(SystemExit, match="2"):
        benchmark.main(["--embed-dim", "31", "--num-heads", "4"])


def test_error_metrics_use_reference_dtype_epsilon() -> None:
    reference: NDArray[np.float16] = np.asarray([0.0, 1.0], dtype=np.float16)
    output: NDArray[np.float16] = np.asarray(
        [np.finfo(np.float16).eps, 1.0], dtype=np.float16
    )
    errors = benchmark._errors(output, reference)
    assert errors["pointwise_relative_denominator_floor"] == np.finfo(np.float16).eps
    assert errors["max_pointwise_relative_error"] == 1.0
