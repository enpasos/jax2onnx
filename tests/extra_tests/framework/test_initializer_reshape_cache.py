# tests/extra_tests/framework/test_initializer_reshape_cache.py

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

from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins._utils import inline_reshape_initializer


@onnx_function
class _RepeatedLinearGeneral(nnx.Module):
    def __init__(self, layer: nnx.LinearGeneral):
        self.layer = layer

    def __call__(self, features: jax.Array) -> jax.Array:
        return self.layer(self.layer(features))


def _source(ctx: IRContext, array: np.ndarray, name: str = "weight") -> ir.Value:
    source = ctx.builder.add_initializer_from_array(name=name, array=array)
    assert source.const_value is not None
    source.const_value.metadata_props["application.path"] = name
    source.const_value.doc_string = "Source tensor"
    source.metadata_props["application.owner"] = "caller"
    source.doc_string = "Source value"
    return source


@pytest.mark.parametrize(
    "array",
    [
        np.array([0, 0x8000, 0x7E11, 0xFE22, 0x3C00, 0xBC00], np.uint16)
        .view(np.float16)
        .reshape(2, 3),
        np.array(
            [0, 0x80000000, 0x7FC00011, 0xFFC00022, 0x3F800000, 0xBF800000],
            np.uint32,
        )
        .view(np.float32)
        .reshape(2, 3),
        np.array(
            [0, 1 << 63, 0x7FF8000000000011, 0xFFF8000000000022, 1, 2],
            np.uint64,
        )
        .view(np.float64)
        .reshape(2, 3),
        np.array([0, 0x8000, 0x7FC1, 0xFFC2, 0x3F80, 0xBF80], np.uint16)
        .view(ml_dtypes.bfloat16)
        .reshape(2, 3),
        np.arange(6, dtype=np.int8).reshape(2, 3),
        np.arange(6, dtype=np.uint8).reshape(2, 3),
        np.arange(6, dtype=np.int32).reshape(2, 3),
        np.arange(6, dtype=np.float32).reshape(3, 2).T,
    ],
    ids=[
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int8",
        "uint8",
        "int32",
        "noncontiguous",
    ],
)
def test_cached_views_preserve_dtype_bits_and_independent_metadata(
    array: np.ndarray, tmp_path: Path
) -> None:
    ctx = IRContext(opset=18)
    source = _source(ctx, array)
    tensor = source.const_value
    assert tensor is not None
    expected_bytes = array.tobytes(order="C")

    view = inline_reshape_initializer(ctx, source, (6,), "view")
    repeated = inline_reshape_initializer(ctx, source, (6,), "another_hint")
    same_shape = inline_reshape_initializer(ctx, source, (2, 3), "same_shape")
    other_shape = inline_reshape_initializer(ctx, source, (3, 2), "other_shape")

    assert repeated is view
    assert len({source, view, same_shape, other_shape}) == 4
    for shaped, shape in [(view, (6,)), (same_shape, (2, 3)), (other_shape, (3, 2))]:
        assert shaped.shape == ir.Shape(shape)
        assert shaped.dtype == source.dtype == tensor.dtype
        assert shaped.const_value is not None and shaped.const_value is not tensor
        assert shaped.const_value.numpy().tobytes(order="C") == expected_bytes
        assert shaped.const_value.metadata_props == tensor.metadata_props
        assert shaped.const_value.doc_string == "Source tensor"
        assert shaped.metadata_props == source.metadata_props
        assert shaped.doc_string == "Source value"

    ctx.builder.outputs.append(view)
    path = tmp_path / "view.onnx"
    onnx.save_model(ir.to_proto(ctx.builder.to_ir_model(name="view")), path)
    restored = onnx.load_model(path)
    onnx.checker.check_model(restored, full_check=True)
    stored = next(t for t in restored.graph.initializer if t.name == view.name)
    assert onnx.numpy_helper.to_array(stored).tobytes(order="C") == expected_bytes
    assert {p.key: p.value for p in stored.metadata_props} == {
        "application.path": "weight"
    }
    assert stored.doc_string == "Source tensor"
    assert restored.graph.output[0].doc_string == "Source value"
    assert {p.key: p.value for p in restored.graph.output[0].metadata_props} == {
        "application.owner": "caller"
    }
    assert view.const_value is not None
    view.const_value.metadata_props["application.path"] = "derived"
    view.metadata_props["application.owner"] = "derived"
    assert tensor.metadata_props["application.path"] == "weight"
    assert source.metadata_props["application.owner"] == "caller"
    assert same_shape.const_value.metadata_props["application.path"] == "weight"
    assert same_shape.metadata_props["application.owner"] == "caller"
    assert source.shape == ir.Shape((2, 3))
    assert tensor.numpy().tobytes(order="C") == expected_bytes


def test_equal_sources_are_independent_and_cache_stays_out_of_graph_clones() -> None:
    ctx = IRContext(opset=18)
    array = np.ones((2, 3), np.float32)
    first = _source(ctx, array, "first")
    second = _source(ctx, array, "second")
    first_view = inline_reshape_initializer(ctx, first, (6,), "view")
    second_view = inline_reshape_initializer(ctx, second, (6,), "view")

    assert first_view is not second_view
    assert first_view.const_value.metadata_props["application.path"] == "first"
    assert second_view.const_value.metadata_props["application.path"] == "second"
    ctx.builder.outputs.extend([first_view, second_view])
    model = ctx.builder.to_ir_model(name="views")
    clone = model.graph.clone()
    assert not model.graph.meta and not clone.meta
    assert clone.outputs[0] is not first_view
    assert len(ctx.builder.reshaped_initializers) == 2


def test_views_are_local_to_graph_and_function_body() -> None:
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
        child.builder.graph.remove(view.producer())
        assert inline_reshape_initializer(child, source, (6,), "again") is not view


@pytest.mark.parametrize("stale", ["removed", "input", "foreign", "runtime"])
def test_stale_or_overridable_cache_entries_are_not_reused(stale: str) -> None:
    ctx = IRContext(opset=18)
    source = _source(ctx, np.arange(6, dtype=np.float32))
    view = inline_reshape_initializer(ctx, source, (2, 3), "view")
    if stale == "removed":
        del ctx.builder.graph.initializers[view.name]
    elif stale == "input":
        ctx.builder.inputs.append(view)
    elif stale == "foreign":
        other = IRContext(opset=18)
        view = inline_reshape_initializer(other, source, (2, 3), "view")
        ctx.builder.reshaped_initializers[(source, (2, 3))] = view
    else:
        view = ctx.builder.op(
            "Identity",
            [source],
            output=ir.val("runtime", dtype=source.dtype, shape=(2, 3)),
        )
        view.const_value = source.const_value
        ctx.builder.reshaped_initializers[(source, (2, 3))] = view

    replacement = inline_reshape_initializer(ctx, source, (2, 3), "view")
    assert replacement is not view
    assert replacement.graph is ctx.builder.graph
    assert not replacement.is_graph_input()


def test_runtime_inputs_and_detached_sources_keep_runtime_reshape_sentinel() -> None:
    ctx = IRContext(opset=18)
    source = _source(ctx, np.arange(6, dtype=np.float32))
    inline_reshape_initializer(ctx, source, (2, 3), "view")
    ctx.builder.inputs.append(source)
    assert inline_reshape_initializer(ctx, source, (2, 3), "view") is source
    ctx.builder.inputs.remove(source)
    runtime = ctx.builder.op(
        "Identity",
        [source],
        output=ir.val("runtime", dtype=source.dtype, shape=source.shape),
    )
    runtime.const_value = source.const_value
    assert inline_reshape_initializer(ctx, runtime, (2, 3), "view") is runtime
    del ctx.builder.graph.initializers[source.name]
    assert inline_reshape_initializer(ctx, source, (2, 3), "view") is source


@pytest.mark.parametrize(
    "x64,return_mode,as_function",
    [
        (False, "proto", False),
        (True, "ir", False),
        (False, "file", False),
        (True, "file", False),
        (False, "file", True),
    ],
)
def test_reused_linear_general_stores_one_kernel_and_preserves_runtime_outputs(
    x64: bool,
    return_mode: Literal["proto", "ir", "file"],
    as_function: bool,
    tmp_path: Path,
) -> None:
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        module = nnx.LinearGeneral(
            512, 512, dtype=dtype, param_dtype=dtype, rngs=nnx.Rngs(0)
        )
        source = np.asarray(module.kernel[...])
        expected_payload = source.tobytes(order="C")

        def apply(features: jax.Array) -> jax.Array:
            return module(module(features))

        path = tmp_path / "shared.onnx"
        model = to_onnx(
            _RepeatedLinearGeneral(module) if as_function else apply,
            [jax.ShapeDtypeStruct((1, 512), dtype)],
            input_names=["features"],
            return_mode=return_mode,
            output_path=path if return_mode == "file" else None,
            opset=18,
        )
        if isinstance(model, ir.Model):
            onnx.save_model(ir.to_proto(model), path)
        elif isinstance(model, onnx.ModelProto):
            onnx.save_model(model, path)
        restored = onnx.load_model(path)
        onnx.checker.check_model(restored, full_check=True)
        tensors = list(restored.graph.initializer)
        nodes = list(restored.graph.node)
        for function in restored.functions:
            nodes.extend(function.node)
            tensors.extend(
                attribute.t
                for node in function.node
                for attribute in node.attribute
                if attribute.type == onnx.AttributeProto.TENSOR
            )
        kernels = [t for t in tensors if tuple(t.dims) == (512, 512)]
        assert len(kernels) == 1
        assert (
            onnx.numpy_helper.to_array(kernels[0]).tobytes(order="C")
            == expected_payload
        )
        gemms = [node for node in nodes if node.op_type == "Gemm"]
        assert len(gemms) == 2
        assert gemms[0].input[1] == gemms[1].input[1]
        if as_function:
            assert restored.functions
            assert not [
                t for t in restored.graph.initializer if tuple(t.dims) == (512, 512)
            ]
            assert any(
                node.op_type == "Constant" and gemms[0].input[1] in node.output
                for node in nodes
            )
        else:
            assert gemms[0].input[1] == kernels[0].name
        assert np.asarray(module.kernel[...]).tobytes(order="C") == expected_payload
        features = np.linspace(-0.5, 0.5, 512, dtype=dtype).reshape(1, 512)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {"features": features})[0]
        expected = np.asarray(apply(jnp.asarray(features)))
        assert actual.dtype == np.dtype(dtype)
        tolerance = 1e-12 if x64 else 1e-5
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
