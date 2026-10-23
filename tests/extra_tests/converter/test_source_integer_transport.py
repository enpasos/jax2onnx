# tests/extra_tests/converter/test_source_integer_transport.py

"""Exact SOURCE INT32 ranges, native selection and bounded gather indices."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.plugins.jax.lax.gather import _source_int32_gather_bound
from jax2onnx.plugins.jax.lax.iota import _source_int32_range_extent
from tests.extra_tests.converter.source_integer_structure import (
    check_integer_source_structure,
)
from tests.extra_tests.converter.test_source_mlp_runtime_acceptance import (
    _runtime_configuration,
)
from tests.extra_tests.converter.test_source_native_tile import (
    _sample,
    _check_one_artifact,
)


@pytest.mark.parametrize("x64", (False, True))
@pytest.mark.parametrize(
    "shape,axis", [((5,), 0), ((1, 1024), 1), ((3, 4), 1), ((4, 3), 0), ((0,), 0)]
)
def test_source_int32_iota_has_exact_native_count_values_shape_and_axis(
    x64: bool, shape: tuple[int, ...], axis: int
) -> None:
    with _runtime_configuration(x64):

        def original() -> Any:
            return jax.lax.broadcasted_iota(np.int32, shape, axis)

        native = jax.make_jaxpr(original)()
        sample = _sample(original, ())
        model = ui._to_onnx(
            original, [], precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        check_integer_source_structure(native, model)
        assert all(n.op_type != "Cast" for n in model.graph.node)
        _check_one_artifact(model, [sample])


def test_iota_range_count_guard_retains_other_existing_representation_paths() -> None:
    source = SimpleNamespace(precision_policy=PrecisionPolicy.SOURCE)
    legacy = SimpleNamespace(precision_policy=PrecisionPolicy.LEGACY)
    for extent in (0, 1024, 1 << 24):
        equation = SimpleNamespace(
            params={"dtype": np.dtype(np.int32), "shape": (1, extent), "dimension": 1}
        )
        assert _source_int32_range_extent(source, equation) == extent
        assert _source_int32_range_extent(legacy, equation) is None
    for dtype, shape in (
        (np.int32, (1, (1 << 24) + 1)),
        (np.int64, (1, 1024)),
        (np.uint32, (1, 1024)),
        (np.float32, (1, 1024)),
        (np.int32, (1, jax_export.symbolic_shape("B")[0])),
    ):
        equation = SimpleNamespace(
            params={"dtype": np.dtype(dtype), "shape": shape, "dimension": 1}
        )
        assert _source_int32_range_extent(source, equation) is None


@pytest.mark.parametrize("x64", (False, True))
def test_native_int32_selection_keeps_extreme_bits_and_sibling_outputs(
    x64: bool,
) -> None:
    with _runtime_configuration(x64):

        def original(mask: Any, left: Any, right: Any) -> Any:
            return jax.lax.select(mask, left, right), left, right

        minimum, maximum = np.iinfo(np.int32).min, np.iinfo(np.int32).max
        values = (
            np.asarray([True, False, True, False], np.bool_),
            np.asarray([minimum, maximum, -1, 0], np.int32),
            np.asarray([maximum, minimum, 0, -1], np.int32),
        )
        sample = _sample(original, values)
        specs = [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values]
        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        assert all(n.op_type != "Cast" for n in model.graph.node)
        where = next(n for n in model.graph.node if n.op_type == "Where")
        assert len(where.input) == 3
        assert all(
            v.type.tensor_type.elem_type == onnx.TensorProto.INT32
            for v in model.graph.output
        )
        _check_one_artifact(model, [sample])


@pytest.mark.parametrize("x64", (False, True))
@pytest.mark.parametrize("mode", ("clip", "fill"))
def test_int32_gather_keeps_native_negative_wrap_clip_fill_and_extreme_indices(
    x64: bool, mode: str
) -> None:
    with _runtime_configuration(x64):

        def original(table: Any, indices: Any) -> Any:
            return jnp.take(table, indices, axis=0, mode=mode), indices

        values = (
            np.arange(24).reshape(8, 3).astype(np.float32) + np.float32(1),
            np.asarray(
                [np.iinfo(np.int32).min, -9, -8, -1, 0, 7, 8, np.iinfo(np.int32).max],
                np.int32,
            ),
        )
        sample = _sample(original, values)
        specs = [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values]
        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        assert all(n.op_type != "Cast" for n in model.graph.node)
        assert any(n.op_type == "Max" for n in model.graph.node)
        assert any(n.op_type == "Min" for n in model.graph.node)
        if mode == "fill":
            assert any(n.op_type == "Or" for n in model.graph.node)
            assert any(n.op_type == "Where" for n in model.graph.node)
        _check_one_artifact(model, [sample])


def test_int32_gather_bound_proof_never_narrows_dynamic_or_unrepresentable_shapes() -> (
    None
):
    context = SimpleNamespace(precision_policy=PrecisionPolicy.SOURCE)
    indices = ir.Value(
        name="indices", type=ir.TensorType(ir.DataType.INT32), shape=ir.Shape((3,))
    )
    for extent, expected in (
        (1, 0),
        (1024, 1023),
        (1 << 31, (1 << 31) - 1),
        (0, None),
        ((1 << 31) + 1, None),
        ("B", None),
    ):
        data = ir.Value(
            name="table",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((extent, 3)),
        )
        assert _source_int32_gather_bound(context, data, indices, 0) == expected
    data = ir.Value(
        name="table", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1024, 3))
    )
    for dtype in (ir.DataType.INT64, ir.DataType.UINT32, ir.DataType.UINT64):
        other = ir.Value(
            name="indices", type=ir.TensorType(dtype), shape=ir.Shape((3,))
        )
        assert _source_int32_gather_bound(context, data, other, 0) is None
    legacy = SimpleNamespace(precision_policy=PrecisionPolicy.LEGACY)
    assert _source_int32_gather_bound(legacy, data, indices, 0) is None


@onnx_function(name="NativeInt32PositionIndexStructure")
def _position(table: Any) -> Any:
    indices = jax.lax.broadcasted_iota(np.int32, (1, 8), 1)
    return jnp.take(table, indices, axis=0)


def _call_position(table: Any) -> Any:
    return _position(table)


def test_source_integer_counterpart_requires_called_function_and_exact_fill_dataflow() -> (
    None
):
    with _runtime_configuration(False):
        specs = [jax.ShapeDtypeStruct((8, 3), np.float32)]
        native = jax.make_jaxpr(_position)(*specs)
        values = (np.arange(24).reshape(8, 3).astype(np.float32) + np.float32(1),)
        sample = _sample(_call_position, values)
        model = ui._to_onnx(
            _call_position, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        result = check_integer_source_structure(native, model, require_function=True)
        assert result["native_specs"][0]["function"] is True
        assert result["native_specs"][0]["fill_drop_gather"] is True
        _check_one_artifact(model, [sample])
        for mutation in (
            "range",
            "axis",
            "clip",
            "fill",
            "call",
            "overload",
            "dead-output",
            "unused-input",
            "unused-root",
        ):
            corrupt = deepcopy(model)
            function = next(
                f
                for f in corrupt.functions
                if "NativeInt32PositionIndexStructure" in f.name
            )
            if mutation == "call":
                call = next(
                    n
                    for n in corrupt.graph.node
                    if n.domain == function.domain and n.op_type == function.name
                )
                call.op_type = "UncalledIntegerFunction"
            elif mutation == "overload":
                call = next(
                    n
                    for n in corrupt.graph.node
                    if n.domain == function.domain and n.op_type == function.name
                )
                call.overload = "unrelated-overload"
            elif mutation == "dead-output":
                name = "unrelated_constant_result"
                function.node.append(
                    onnx.helper.make_node(
                        "Constant",
                        [],
                        [name],
                        value=onnx.numpy_helper.from_array(
                            np.zeros((1, 8, 3), np.float32)
                        ),
                    )
                )
                function.value_info.append(
                    onnx.helper.make_tensor_value_info(
                        name, onnx.TensorProto.FLOAT, [1, 8, 3]
                    )
                )
                function.output[0] = name
            elif mutation == "unused-input":
                call = next(
                    n
                    for n in corrupt.graph.node
                    if n.domain == function.domain and n.op_type == function.name
                )
                call.input.append(call.input[0])
                function.input.append("unused_function_input")
                function.value_info.append(
                    onnx.helper.make_tensor_value_info(
                        "unused_function_input", onnx.TensorProto.FLOAT, [8, 3]
                    )
                )
            elif mutation == "unused-root":
                corrupt.graph.input.append(
                    onnx.helper.make_tensor_value_info(
                        "unused_model_input", onnx.TensorProto.INT32, [1]
                    )
                )
            elif mutation == "range":
                node = next(n for n in function.node if n.op_type == "Range")
                node.input[1] = node.input[2]
            elif mutation == "axis":
                node = next(n for n in function.node if n.op_type == "Unsqueeze")
                constant = next(n for n in function.node if node.input[1] in n.output)
                attribute = next(a for a in constant.attribute if a.name == "value")
                attribute.t.CopyFrom(
                    onnx.numpy_helper.from_array(np.asarray([1], np.int64))
                )
            elif mutation == "clip":
                node = next(n for n in function.node if n.op_type == "Min")
                lower = next(n for n in function.node if n.output[0] == node.input[0])
                node.input[1] = lower.input[1]
            else:
                node = next(n for n in reversed(function.node) if n.op_type == "Where")
                node.input[2] = node.input[1]
            with pytest.raises(ValueError):
                check_integer_source_structure(native, corrupt, require_function=True)


def test_integer_counterpart_binds_exact_native_iota_not_an_unrelated_same_shape_gather() -> (
    None
):
    with _runtime_configuration(False):

        def original(table: Any, independent_indices: Any) -> Any:
            positions = jax.lax.broadcasted_iota(np.int32, (1, 8), 1)
            return positions, jnp.take(table, independent_indices, axis=0)

        specs = [
            jax.ShapeDtypeStruct((8, 3), np.float32),
            jax.ShapeDtypeStruct((1, 8), np.int32),
        ]
        native = jax.make_jaxpr(original)(*specs)
        values = (
            np.arange(24).reshape(8, 3).astype(np.float32) + np.float32(1),
            np.asarray([[7, 6, 5, 4, 3, 2, 1, 0]], np.int32),
        )
        sample = _sample(original, values)
        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        evidence = check_integer_source_structure(native, model)
        assert evidence["native_specs"][0]["fill_drop_gather"] is False
        _check_one_artifact(model, [sample])
        corrupt = deepcopy(model)
        gather = next(
            n
            for n in corrupt.graph.node
            if n.op_type == "Gather" and n.input[0] == corrupt.graph.input[0].name
        )
        gather.input[1] = corrupt.graph.output[0].name
        with pytest.raises(ValueError, match="unrelated to its exact native iota"):
            check_integer_source_structure(native, corrupt)


def test_integer_structural_amendment_keeps_unrelated_original_callback_obligations() -> (
    None
):
    from jax2onnx.plugins._post_check_onnx_graph import expect_graph
    from tests.extra_tests.converter.source_integer_structure import (
        check_integer_callback_counterpart,
    )

    with _runtime_configuration(False):

        def original() -> Any:
            return jax.lax.iota(np.int32, 5)

        native = jax.make_jaxpr(original)()
        model = ui._to_onnx(
            original, [], precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        check_integer_source_structure(native, model)
        callback = expect_graph(["Range:5 -> Cast:5"], no_unused_inputs=True)
        assert callback(model) is False  # Retained raw LEGACY structural result.
        evidence = check_integer_callback_counterpart(callback, model)
        assert evidence["other_callback_obligations"] == "preserved and passed"
        for incompatible in (
            expect_graph(["Range:5 -> Cast:5", "UnrelatedMissingOperation"]),
            expect_graph(["Range:5 -> UnrelatedMissingOperation -> Gather"]),
            expect_graph(["Range:5 -> UnrelatedMissingOperation -> Cast:5"]),
            expect_graph(["Range:5 -> Cast:5"], must_absent=["Range"]),
            expect_graph(
                [{"path": "Range:5 -> Cast:5", "graph": "unrelated-function"}],
                search_functions=True,
            ),
        ):
            with pytest.raises(
                ValueError, match="preserved original callback obligation"
            ):
                check_integer_callback_counterpart(incompatible, model)
