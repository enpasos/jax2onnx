# tests/extra_tests/test_qwix_quantized_export.py

"""Optional W8A8 recipe for Qwix 0.1.8 at commit 9ee1079245682ed82e5da00ce855e4daf77a086b.

Use an isolated environment containing the repository's JAX 0.10.2 / Flax 0.12.8
stack. Ordinary converter use needs no Qwix. The optional recipe can be run with::

    python -m pip install --no-deps git+https://github.com/google/qwix@9ee1079245682ed82e5da00ce855e4daf77a086b
    python -m pytest -q tests/extra_tests/test_qwix_quantized_export.py
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx


def test_qwix_w8a8_dense_relu_dense_exports_before_native_tracing() -> None:
    try:
        version("qwix")
    except PackageNotFoundError:
        pytest.skip("Optional Qwix distribution is not installed")
    qwix = pytest.importorskip("qwix")

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, inputs: jax.Array) -> jax.Array:
            hidden = nn.Dense(
                8,
                use_bias=False,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                name="dense0",
            )(inputs)
            return nn.Dense(
                3,
                use_bias=False,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                name="dense1",
            )(nn.relu(hidden))

    inputs = np.asarray(
        [[1.0, -0.5, 2.0, -1.5], [-1.0, 0.25, 0.5, 1.25]], dtype=np.float32
    )
    fp_params = {
        "dense0": {
            "kernel": jnp.asarray(
                np.linspace(-0.8, 0.9, 32, dtype=np.float32).reshape(4, 8)
            )
        },
        "dense1": {
            "kernel": jnp.asarray(
                np.linspace(-0.6, 0.75, 24, dtype=np.float32).reshape(8, 3)
            )
        },
    }
    rule = qwix.QuantizationRule(
        op_names=("dot_general",),
        weight_qtype=jnp.int8,
        act_qtype=jnp.int8,
        act_static_scale=False,
        weight_calibration_method="absmax",
        act_calibration_method="absmax",
        tile_size=None,
    )
    ptq_model = qwix.quantize_model(MLP(), qwix.PtqProvider([rule], disable_jit=True))
    # Abstract initialization prepares the parameter template; it does not run serve.
    abstract = jax.eval_shape(
        ptq_model.init,
        jax.random.key(0),
        jax.ShapeDtypeStruct(inputs.shape, jnp.float32),
    )
    ptq_params = qwix.quantize_params(fp_params, abstract["params"])
    prepared_kernels = []
    for name in ("dense0", "dense1"):
        quantized = ptq_params[name]["kernel"].array
        assert isinstance(quantized, qwix.QArray)
        assert quantized.qvalue.dtype == jnp.int8
        assert quantized.scale.dtype == jnp.float32
        assert quantized.zero_point is None
        prepared_kernels.append(np.asarray(quantized.qvalue))

    def serve(values: jax.Array) -> jax.Array:
        return ptq_model.apply({"params": ptq_params}, values)

    # Export must be the first trace of this serving callable. A previously cached
    # unpatched Linen trace can otherwise hide an ineligible Dense shortcut.
    model = to_onnx(
        serve,
        [jax.ShapeDtypeStruct(inputs.shape, jnp.float32)],
        enable_double_precision=False,
    )
    onnx.checker.check_model(model, full_check=True)
    integer_dots = [
        node for node in model.graph.node if node.op_type == "MatMulInteger"
    ]
    assert len(integer_dots) == 2
    assert not any(node.op_type in {"Gemm", "Einsum"} for node in model.graph.node)
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert tuple(
        dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim
    ) == (2, 3)

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        )
    }
    value_types.update(
        (tensor.name, tensor.data_type) for tensor in inferred.graph.initializer
    )
    for node in integer_dots:
        assert value_types[node.input[0]] == onnx.TensorProto.INT8
        assert value_types[node.input[1]] == onnx.TensorProto.INT8
        assert value_types[node.output[0]] == onnx.TensorProto.INT32

    constants = [
        onnx.numpy_helper.to_array(tensor) for tensor in model.graph.initializer
    ]
    constants.extend(
        onnx.numpy_helper.to_array(attribute.t)
        for node in model.graph.node
        if node.op_type == "Constant"
        for attribute in node.attribute
        if attribute.type == onnx.AttributeProto.TENSOR
    )
    for kernel in prepared_kernels:
        assert any(
            constant.dtype == np.int8
            and constant.shape == kernel.shape
            and np.array_equal(constant, kernel)
            for constant in constants
        )

    native_input = jnp.asarray(inputs)
    eager = np.asarray(serve(native_input))

    def fresh_serve(values: jax.Array) -> jax.Array:
        return serve(values)

    compiled = jax.jit(fresh_serve)
    results = [
        ("eager", eager),
        ("fresh_jit", np.asarray(compiled(native_input))),
        ("warm_jit", np.asarray(compiled(native_input))),
    ]
    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        (actual,) = session.run(None, {session.get_inputs()[0].name: inputs})
        results.append((str(optimization), np.asarray(actual)))

    for role, result in results:
        assert result.dtype == np.float32
        assert result.shape == (2, 3)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result, eager, rtol=1e-6, atol=1e-6, err_msg=role)
