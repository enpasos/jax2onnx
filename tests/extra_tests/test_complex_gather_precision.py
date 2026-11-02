# tests/extra_tests/test_complex_gather_precision.py

"""Gather preserves logical complex indexing and physical component channels."""

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx
from jax2onnx._compat.jax import ShapedArray, Var, new_jaxpr_eqn
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins.jax.lax import gather as gather_module
from tests.extra_tests.test_precision_control_flow import (
    _assert_exact,
    _check_native_and_ort,
)


def _packed(value):
    value = np.asarray(value)
    return np.stack((value.real, value.imag), axis=-1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["rows", "middle-axis", "multiple-axes"])
def test_complex_gather_preserves_shared_source_and_component_layout(dtype, layout):
    shape = (4, 2) if layout == "rows" else (3, 4, 2)
    real = (np.arange(np.prod(shape)).reshape(shape) + 2**24 + 1).astype(dtype)
    imag = (-np.arange(np.prod(shape)).reshape(shape) - 0.25).astype(dtype)
    real.flat[0], imag.flat[1] = -0.0, -0.0
    if layout == "rows":
        indices = np.asarray([[[3], [0], [5]], [[1], [-1], [2]]], np.int32)
        dimensions = jax.lax.GatherDimensionNumbers(
            offset_dims=(2,), collapsed_slice_dims=(0,), start_index_map=(0,)
        )
        slices, mode = (1, 2), "clip"
        selection = np.clip(indices[..., 0], 0, shape[0] - 1)
        expected = real[selection], imag[selection]
    elif layout == "middle-axis":
        indices = np.asarray([[3], [1]], np.int32)
        dimensions = jax.lax.GatherDimensionNumbers(
            offset_dims=(0, 2), collapsed_slice_dims=(1,), start_index_map=(1,)
        )
        slices, mode = (3, 1, 2), "promise_in_bounds"
        expected = real[:, indices[:, 0], :], imag[:, indices[:, 0], :]
    else:
        indices = np.asarray([[2, 1], [0, 3]], np.int32)
        dimensions = jax.lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0, 1), start_index_map=(0, 1)
        )
        slices, mode = (1, 1, 2), "promise_in_bounds"
        expected = (
            real[indices[:, 0], indices[:, 1]],
            imag[indices[:, 0], indices[:, 1]],
        )

    def source(real, imag, positions):
        original = jax.lax.complex(real, imag)
        selected = jax.lax.gather(original, positions, dimensions, slices, mode=mode)
        return (
            jnp.real(selected),
            jnp.imag(selected),
            jnp.real(original),
            jnp.imag(original),
        )

    specs = [jax.ShapeDtypeStruct(shape, dtype)] * 2
    index_shape = ("B", 3, 1) if layout == "rows" else indices.shape
    specs.append(jax.ShapeDtypeStruct(index_shape, np.int32))
    with jax.enable_x64():
        model = _check_native_and_ort(
            source, [real, imag, indices], (*expected, real, imag), inputs=specs
        )
    base_type = (
        onnx.TensorProto.FLOAT if dtype is np.float32 else onnx.TensorProto.DOUBLE
    )
    assert all(
        output.type.tensor_type.elem_type == base_type for output in model.graph.output
    )
    values = {
        value.name: value
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
    }
    assert any(
        values[node.output[0]].type.tensor_type.elem_type == base_type
        and len(values[node.output[0]].type.tensor_type.shape.dim)
        == expected[0].ndim + 1
        and values[node.output[0]].type.tensor_type.shape.dim[-1].dim_value == 2
        for node in model.graph.node
        if node.op_type in {"Gather", "GatherND"} and node.output[0] in values
    )


def _gather_fill(real, imag, indices, fill):
    original = jax.lax.complex(real, imag)
    selected = jax.lax.gather(
        original,
        indices,
        jax.lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
        ),
        slice_sizes=(1, 2),
        mode="fill",
        fill_value=fill,
    )
    return jnp.real(selected), jnp.imag(selected), real, imag


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_complex_gather_equal_components_use_exact_typed_fill(dtype):
    real = np.asarray([[2**24 + 1, 2**24 + 3], [3, 4]], dtype)
    imag = np.asarray([[0.5, -1], [1.5, -2]], dtype)
    indices = np.asarray([[0], [-1], [3]], np.int32)

    def source(real, imag, positions):
        return _gather_fill(real, imag, positions, 7 + 7j)

    expected = tuple(
        np.stack([value[0], np.full((2,), 7, dtype), np.full((2,), 7, dtype)])
        for value in (real, imag)
    )
    with jax.enable_x64():
        model = _check_native_and_ort(
            source, [real, imag, indices], (*expected, real, imag)
        )
    assert any(node.op_type == "Where" for node in model.graph.node)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("fill_bits", ["negative-zero", "nan-payload"])
def test_complex_gather_adapter_keeps_source_equation_and_fill_bits(
    dtype, fill_bits, monkeypatch
):
    component = np.float32 if dtype is np.complex64 else np.float64
    word_dtype = np.uint32 if dtype is np.complex64 else np.uint64
    sign = 1 << (np.dtype(word_dtype).itemsize * 8 - 1)
    nan = 0x7FC12345 if dtype is np.complex64 else 0x7FF8000000001234
    word = sign if fill_bits == "negative-zero" else nan
    fill = np.asarray([word, word], word_dtype).view(dtype)[0]
    data = np.zeros((4, 2), dtype)
    data_var = Var(ShapedArray(data.shape, dtype))
    indices_var = Var(ShapedArray((2, 1), np.int32))
    out_var = Var(ShapedArray((2, 2), dtype))
    dimensions = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    eqn = new_jaxpr_eqn(
        [data_var, indices_var],
        [out_var],
        jax.lax.gather_p,
        {
            "dimension_numbers": dimensions,
            "slice_sizes": (1, 2),
            "mode": jax.lax.GatherScatterMode.FILL_OR_DROP,
            "fill_value": fill,
            "unique_indices": False,
            "indices_are_sorted": False,
        },
        frozenset(),
    )
    original_params = dict(eqn.params)
    original_avals = data_var.aval, out_var.aval
    ctx = IRContext(opset=23)
    packed_data = ctx.bind_const_for_var(data_var, data)
    ctx.bind_const_for_var(indices_var, np.asarray([[2], [0]], np.int32))
    captured = []
    compile_to_gir = gather_module.compile_to_gir

    def capture(physical_eqn, constant_indices):
        captured.append(physical_eqn)
        return compile_to_gir(physical_eqn, constant_indices)

    monkeypatch.setattr(gather_module, "compile_to_gir", capture)
    gather_module.GatherPlugin().lower(ctx, eqn)
    (physical,) = captured
    assert physical.invars[0] is not data_var
    assert physical.outvars[0] is not out_var
    assert physical.invars[1] is indices_var
    assert physical.invars[0].aval.shape == (4, 2, 2)
    assert physical.outvars[0].aval.shape == (2, 2, 2)
    assert physical.invars[0].aval.dtype == np.dtype(component)
    assert physical.outvars[0].aval.dtype == np.dtype(component)
    assert physical.params["slice_sizes"] == (1, 2, 2)
    assert physical.params["dimension_numbers"].offset_dims == (1, 2)
    assert physical.params["dimension_numbers"].start_index_map == (0,)
    assert physical.params["dimension_numbers"].collapsed_slice_dims == (0,)
    assert np.asarray(physical.params["fill_value"]).dtype == np.dtype(component)
    assert np.asarray(physical.params["fill_value"]).tobytes() == fill.real.tobytes()
    assert all(eqn.params[key] is value for key, value in original_params.items())
    assert data_var.aval is original_avals[0] and out_var.aval is original_avals[1]
    assert ctx.require_value_for_var(data_var) is packed_data
    assert ctx.require_value_for_var(physical.invars[0]) is packed_data
    assert packed_data.dtype == ir.DataType.from_numpy(np.dtype(component))
    assert packed_data.shape == ir.Shape((4, 2, 2))
    assert ctx.try_evaluate_const(data_var) is data
    assert ctx.require_value_for_var(out_var).shape == ir.Shape((2, 2, 2))


@pytest.mark.parametrize("fill_kind", ["finite", "zero-sign", "nan-payload", "default"])
def test_complex_gather_unequal_fill_components_raise_precise_diagnostic(fill_kind):
    if fill_kind == "finite":
        fill = 7 + 8j
    elif fill_kind == "zero-sign":
        fill = np.asarray([0, 1 << 63], np.uint64).view(np.complex128)[0]
    elif fill_kind == "nan-payload":
        fill = np.asarray([0x7FF8000000001234, 0x7FF8000000005678], np.uint64).view(
            np.complex128
        )[0]
    else:
        fill = None

    def source(real, imag, indices):
        return _gather_fill(real, imag, indices, fill)

    inputs = [
        np.asarray([[1, 2], [3, 4]], np.float64),
        np.asarray([[0.5, -1], [1.5, -2]], np.float64),
        np.asarray([[0], [3]], np.int32),
    ]
    with (
        jax.enable_x64(),
        pytest.raises(
            NotImplementedError,
            match="complex gather fill_value with unequal real and imaginary components",
        ),
    ):
        to_onnx(source, inputs)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_gather_direct_input_keeps_packed_interface(dtype):
    data = np.empty((3, 2), dtype)
    data.real = np.asarray([[2**24 + 1, -0.0], [3, 4], [5, 6]], data.real.dtype)
    data.imag = np.asarray([[0.5, -1], [1.5, -0.0], [2.5, -3]], data.real.dtype)
    indices = np.asarray([[2], [0]], np.int32)

    def source(value, positions):
        selected = jax.lax.gather(
            value,
            positions,
            jax.lax.GatherDimensionNumbers(
                offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
            ),
            slice_sizes=(1, 2),
            mode="promise_in_bounds",
        )
        return selected, value

    expected = _packed(data[indices[:, 0]]), _packed(data)
    with jax.enable_x64():
        native_data, native_indices = jnp.asarray(data), jnp.asarray(indices)
        reference = tuple(
            _packed(value) for value in source(native_data, native_indices)
        )
        _assert_exact(reference, expected)
        compiled = jax.jit(source)
        for _ in range(2):
            _assert_exact(
                tuple(
                    _packed(value) for value in compiled(native_data, native_indices)
                ),
                reference,
            )
        model = to_onnx(
            source,
            [
                jax.ShapeDtypeStruct(data.shape, dtype),
                jax.ShapeDtypeStruct(indices.shape, np.int32),
            ],
        )
    onnx.checker.check_model(model, full_check=True)
    base_type = (
        onnx.TensorProto.FLOAT if dtype is np.complex64 else onnx.TensorProto.DOUBLE
    )
    assert model.graph.input[0].type.tensor_type.elem_type == base_type
    assert [
        dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim
    ] == [
        3,
        2,
        2,
    ]
    assert all(
        value.type.tensor_type.elem_type == base_type for value in model.graph.output
    )
    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(
            None,
            {
                session.get_inputs()[0].name: _packed(data),
                session.get_inputs()[1].name: indices,
            },
        )
        _assert_exact(tuple(actual), reference)
