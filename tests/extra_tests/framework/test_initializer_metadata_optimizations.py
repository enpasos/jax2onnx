# tests/extra_tests/framework/test_initializer_metadata_optimizations.py

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx
from jax2onnx.converter.ir_optimizations import optimize_graph


def _run(model: onnx.ModelProto, **inputs: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return np.asarray(session.run(None, inputs)[0])


@pytest.mark.parametrize("annotated", [False, True])
def test_constant_cse_preserves_distinct_annotated_sources(annotated: bool) -> None:
    nodes: list[ir.Node] = []
    values: list[ir.Value] = []
    for name in ("left", "right"):
        tensor = ir.tensor(np.asarray([2.0], dtype=np.float32))
        if annotated:
            tensor.metadata_props["caller.path"] = name
        value = ir.Value(
            name=name,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1]),
            const_value=tensor,
        )
        nodes.append(
            ir.Node(
                "",
                "Constant",
                [],
                outputs=[value],
                attributes=[ir.AttrTensor("value", tensor)],
            )
        )
        values.append(value)
    output = ir.val("sum", dtype=ir.DataType.FLOAT, shape=[1])
    nodes.append(ir.Node("", "Add", values, outputs=[output]))
    model = ir.Model(
        ir.Graph(
            [],
            [output],
            nodes=nodes,
            opset_imports={"": 18},
            name="annotated_constants",
        ),
        ir_version=10,
    )
    optimize_graph(model)
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    np.testing.assert_array_equal(_run(proto), np.asarray([4.0], dtype=np.float32))
    assert len(proto.graph.initializer) == (2 if annotated else 1)
    if annotated:
        assert {
            dict((entry.key, entry.value) for entry in tensor.metadata_props)[
                "caller.path"
            ]
            for tensor in proto.graph.initializer
        } == {"left", "right"}
        add = next(node for node in proto.graph.node if node.op_type == "Add")
        assert add.input[0] != add.input[1]


@pytest.mark.parametrize("cast_source", [False, True])
def test_broadcast_keeps_one_annotated_scalar_and_update_affects_every_element(
    cast_source: bool,
) -> None:
    dtype = jnp.float16 if cast_source else jnp.float32
    weight = jnp.asarray(2.0, dtype=dtype)

    def fn(x: jax.Array) -> jax.Array:
        value = weight.astype(jnp.float32) if cast_source else weight
        return x + jnp.broadcast_to(value, (2, 3))

    def metadata(source: object) -> dict[str, str] | None:
        return {"caller.path": "weight"} if source is weight else None

    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct((2, 3), jnp.float32)],
        input_names=["x"],
        initializer_metadata=metadata,
    )
    annotated = [tensor for tensor in model.graph.initializer if tensor.metadata_props]
    assert len(annotated) == 1
    parameter = annotated[0]
    assert list(parameter.dims) == []
    assert {entry.key: entry.value for entry in parameter.metadata_props} == {
        "caller.path": "weight"
    }
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_array_equal(_run(model, x=x), x + 2)
    parameter.raw_data = np.asarray(
        5.0, dtype=np.float16 if cast_source else np.float32
    ).tobytes()
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_array_equal(_run(model, x=x), x + 5)


def test_eqx_conv_bias_reshape_keeps_source_metadata_and_updates() -> None:
    # Exercise a pure FP32 reshape even when the caller enables JAX x64.
    module = eqx.nn.Conv2d(
        2, 3, 3, padding=1, dtype=jnp.float32, key=jax.random.key(37)
    )
    assert module.bias is not None
    source_bias = np.asarray(module.bias)
    assert source_bias.shape == (3, 1, 1)

    def metadata(source: object) -> dict[str, str] | None:
        if source is module.weight:
            return {"caller.path": "conv.weight"}
        if source is module.bias:
            return {"caller.path": "conv.bias"}
        return None

    model = to_onnx(
        module,
        [jax.ShapeDtypeStruct((2, 4, 4), jnp.float32)],
        input_names=["x"],
        initializer_metadata=metadata,
    )
    parameters = {
        entry.value: tensor
        for tensor in model.graph.initializer
        for entry in tensor.metadata_props
        if entry.key == "caller.path"
    }
    assert parameters.keys() == {"conv.weight", "conv.bias"}
    bias = parameters["conv.bias"]
    assert list(bias.dims) == [3]
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(bias), source_bias.reshape(3)
    )
    x = np.arange(32, dtype=np.float32).reshape(2, 4, 4) / 16
    expected = np.asarray(module(jnp.asarray(x)))
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_allclose(_run(model, x=x), expected, rtol=1e-6, atol=1e-6)
    bias.raw_data = (source_bias.reshape(3) + np.float32(0.125)).tobytes()
    np.testing.assert_allclose(_run(model, x=x), expected + 0.125, rtol=1e-6, atol=1e-6)
