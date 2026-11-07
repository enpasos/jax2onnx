# tests/extra_tests/framework/test_initializer_reshape_metadata.py

from __future__ import annotations

from pathlib import Path
from typing import Literal

from flax import nnx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins._utils import inline_reshape_initializer
from jax2onnx.plugins.equinox.eqx.nn.linear import _inline_scalar_bias


def _source(ctx: IRContext, array: np.ndarray, name: str = "weight") -> ir.Value:
    source = ctx.builder.add_initializer_from_array(name=name, array=array)
    assert source.const_value is not None
    source.const_value.metadata_props.update({"trainable": "true", "source_path": name})
    source.const_value.doc_string = "Source tensor"
    source.metadata_props["application.owner"] = "caller"
    source.doc_string = "Source value"
    return source


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, ml_dtypes.bfloat16]
)
def test_reshape_cache_preserves_source_dtype_payload_and_metadata(dtype: type) -> None:
    ctx = IRContext(opset=18)
    array = np.arange(6).astype(dtype).reshape(2, 3)
    source = _source(ctx, array)
    tensor = source.const_value
    assert tensor is not None

    shaped = inline_reshape_initializer(ctx, source, (6,), "view")
    repeated = inline_reshape_initializer(ctx, source, (6,), "other_hint")
    unchanged_shape = inline_reshape_initializer(ctx, source, (2, 3), "same_shape")

    assert repeated is shaped
    assert shaped is not source and unchanged_shape is not source
    assert unchanged_shape is not shaped
    assert shaped.shape == ir.Shape((6,))
    assert shaped.dtype == source.dtype == tensor.dtype
    assert shaped.const_value is not None and shaped.const_value is not tensor
    assert shaped.const_value.metadata_props == tensor.metadata_props
    assert shaped.const_value.doc_string == "Source tensor"
    assert shaped.metadata_props == source.metadata_props
    assert shaped.doc_string == "Source value"
    np.testing.assert_array_equal(shaped.const_value.numpy(), array.reshape(6))
    np.testing.assert_array_equal(tensor.numpy(), array)
    assert source.shape == ir.Shape((2, 3))
    shaped.const_value.metadata_props["source_path"] = "derived-only"
    shaped.metadata_props["application.owner"] = "derived-only"
    assert tensor.metadata_props["source_path"] == "weight"
    assert source.metadata_props["application.owner"] == "caller"


def test_equal_payloads_with_distinct_values_get_distinct_constant_views() -> None:
    ctx = IRContext(opset=18)
    array = np.ones((2, 3), np.float32)
    first = _source(ctx, array, "first")
    second = _source(ctx, array, "second")

    first_view = inline_reshape_initializer(ctx, first, (6,), "view")
    second_view = inline_reshape_initializer(ctx, second, (6,), "view")

    assert first_view is not second_view
    assert first_view.const_value.metadata_props["source_path"] == "first"
    assert second_view.const_value.metadata_props["source_path"] == "second"


def test_reshape_cache_is_local_to_graph_and_function_body() -> None:
    parent = IRContext(opset=18)
    source = _source(parent, np.arange(6, dtype=np.float32).reshape(2, 3))
    parent_view = inline_reshape_initializer(parent, source, (6,), "view")
    children = [parent.create_child_context(scope="function") for _ in range(2)]
    views = [
        inline_reshape_initializer(child, source, (6,), "view") for child in children
    ]

    assert len({parent_view, *views}) == 3
    for child, view in zip(children, views):
        assert view.graph is child.builder.graph
        assert view.producer().op_type == "Constant"
        assert not child.builder.graph.initializers
        assert view.const_value.metadata_props == source.const_value.metadata_props
        assert inline_reshape_initializer(child, source, (6,), "again") is view


def test_reshape_cache_is_excluded_from_exported_graphs_and_clones() -> None:
    ctx = IRContext(opset=18)
    source = _source(ctx, np.arange(6, dtype=np.float32).reshape(2, 3))
    view = inline_reshape_initializer(ctx, source, (6,), "view")
    ctx.builder.outputs.append(view)

    model = ctx.builder.to_ir_model(name="views", ir_version=10)
    clone = model.graph.clone()

    assert ctx.builder.reshaped_initializers == {(source, (6,)): view}
    assert not model.graph.meta
    assert not clone.meta
    assert clone.outputs[0] is not view


def test_cached_initializer_that_becomes_overridable_is_not_reused() -> None:
    ctx = IRContext(opset=18)
    source = _source(ctx, np.arange(6, dtype=np.float32).reshape(2, 3))
    first_view = inline_reshape_initializer(ctx, source, (6,), "view")
    ctx.builder.inputs.append(first_view)

    next_view = inline_reshape_initializer(ctx, source, (6,), "view")

    assert next_view is not first_view
    assert next_view.graph is ctx.builder.graph and not next_view.is_graph_input()
    ctx.builder.inputs.append(source)
    assert inline_reshape_initializer(ctx, source, (6,), "view") is source


def test_inferred_constant_on_runtime_node_is_not_snapshotted() -> None:
    ctx = IRContext(opset=18)
    source = _source(ctx, np.arange(6, dtype=np.float32))
    runtime = ctx.builder.op(
        "Identity",
        [source],
        output=ir.val("runtime", dtype=source.dtype, shape=source.shape),
    )
    runtime.const_value = source.const_value

    assert inline_reshape_initializer(ctx, runtime, (2, 3), "view") is runtime


def _run(model: onnx.ModelProto, features: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, {"features": features})[0]


@pytest.mark.parametrize("x64,return_mode", [(False, "proto"), (True, "ir")])
def test_reused_linear_general_exports_one_tagged_kernel_and_runtime_parity(
    x64: bool, return_mode: Literal["proto", "ir"], tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        module = nnx.LinearGeneral(
            512, 512, dtype=dtype, param_dtype=dtype, rngs=nnx.Rngs(0)
        )
        kernel = module.kernel.get_value()
        tags = {"trainable": "true", "source_path": "shared.kernel"}

        def apply(features: jax.Array) -> jax.Array:
            return module(module(features))

        def metadata(value: object) -> dict[str, str] | None:
            return tags if value is kernel else None

        model = to_onnx(
            apply,
            [jax.ShapeDtypeStruct((1, 512), dtype)],
            input_names=["features"],
            initializer_metadata=metadata,
            return_mode=return_mode,
            opset=18,
        )
        proto = ir.to_proto(model) if isinstance(model, ir.Model) else model
        kernels = [t for t in proto.graph.initializer if tuple(t.dims) == (512, 512)]
        assert len(kernels) == 1
        assert {p.key: p.value for p in kernels[0].metadata_props} == tags
        gemms = [node for node in proto.graph.node if node.op_type == "Gemm"]
        assert len(gemms) == 2
        assert gemms[0].input[1] == gemms[1].input[1] == kernels[0].name
        path = tmp_path / "shared.onnx"
        onnx.save_model(proto, path)
        reloaded = onnx.load_model(path)
        onnx.checker.check_model(reloaded, full_check=True)
        tagged = next(
            t for t in reloaded.graph.initializer if t.name == kernels[0].name
        )
        assert {p.key: p.value for p in tagged.metadata_props} == tags
        features = np.linspace(-0.5, 0.5, 512, dtype=dtype).reshape(1, 512)
        expected = np.asarray(apply(jnp.asarray(features)))
        actual = _run(reloaded, features)
        assert actual.dtype == np.dtype(dtype)
        tolerance = 1e-12 if x64 else 1e-5
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_equal_valued_linear_general_kernels_preserve_distinct_parameter_sources() -> (
    None
):
    first = nnx.LinearGeneral(4, 4, rngs=nnx.Rngs(0))
    second = nnx.LinearGeneral(4, 4, rngs=nnx.Rngs(1))
    first.kernel[...] = jnp.asarray(np.ones((4, 4), np.float32))
    second.kernel[...] = jnp.asarray(np.ones((4, 4), np.float32))
    first_kernel, second_kernel = first.kernel.get_value(), second.kernel.get_value()
    assert first_kernel is not second_kernel

    def metadata(value: object) -> dict[str, str] | None:
        if value is first_kernel:
            return {"source_path": "first.kernel"}
        if value is second_kernel:
            return {"source_path": "second.kernel"}
        return None

    model = to_onnx(
        lambda x: first(x) + second(x),
        [jax.ShapeDtypeStruct((1, 4), np.float32)],
        initializer_metadata=metadata,
        opset=18,
    )

    kernels = [t for t in model.graph.initializer if tuple(t.dims) == (4, 4)]
    assert len(kernels) == 2
    assert {
        next(p.value for p in tensor.metadata_props if p.key == "source_path")
        for tensor in kernels
    } == {"first.kernel", "second.kernel"}


@pytest.mark.parametrize("scalar", [False, True])
def test_eqx_bias_inline_preserves_one_to_one_metadata_and_tagged_broadcasts(
    scalar: bool,
) -> None:
    ctx = IRContext(opset=18)
    array = np.asarray(0.5, np.float32) if scalar else np.arange(4, dtype=np.float32)
    source = _source(ctx, array)
    shape = ctx.builder.const_i64(name="shape", values=[4])
    bias = ctx.builder.op(
        "Expand",
        [source, shape],
        output=ir.val("bias", dtype=source.dtype, shape=(4,)),
    )

    folded = _inline_scalar_bias(ctx, bias, 4)

    if scalar:
        assert folded is bias
    else:
        assert folded is not bias and folded is not source
        assert folded.const_value.metadata_props == source.const_value.metadata_props
        assert _inline_scalar_bias(ctx, bias, 4) is folded
