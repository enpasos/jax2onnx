# tests/extra_tests/converter/test_source_tensor_indexing.py

"""Common tensor/index families enter the product SOURCE route with native data.

Original raw rows retain their factories, criteria, opsets and callbacks. This
file adds no diagnostic validator or alternative exporter. Original loop/trig
compositions and broader symbolic declarations remain owned by their batches.
"""

from collections.abc import Iterator
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.precision import CheckStatus, SampleCriteria, get_precision_report
from scripts.source_precision_inventory import canonical_contract
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter.test_source_profile import _graph as _typed_graph
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _accept,
    _artifact,
    _body,
    _globals,
    _settings,
    _specs,
)


# This is the finite original family selector, frozen before execution. The
# aggregate binds every raw ordinal, name, opset and complete original contract.
_FAMILY_ROWS = {
    "concatenate": 5,
    "dynamic_slice": 4,
    "dynamic_update_slice": 6,
    "gather": 7,
    "iota": 7,
    "pad": 5,
    "reshape": 9,
    "rev": 2,
    "slice": 3,
    "squeeze": 8,
    "transpose": 7,
}
_PENDING_ORIGINAL = {
    ("gather", 0): "original trigonometric composition",
    ("gather", 1): "original trigonometric composition",
    ("pad", 3): "original scan composition",
    ("pad", 4): "original nested scan composition",
    ("slice", 2): "original scan composition",
    ("reshape", 5): "multiple symbolic dimensions",
    ("reshape", 8): "derived symbolic B*4 extent",
}
_ORIGINAL = tuple(
    (name, ordinal) for name, count in _FAMILY_ROWS.items() for ordinal in range(count)
)
_PRODUCT_ORIGINAL = tuple(cell for cell in _ORIGINAL if cell not in _PENDING_ORIGINAL)
_ORIGINAL_SHA256 = "34f51a48c15520e0d39c2c9c8396c33b3b8fb7e82c549dcf1b4b4776be171525"
_REAL_DTYPES = (
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    jnp.bfloat16,
    np.float32,
    np.float64,
)


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    hidden, effective = _globals(), _settings()
    policy = active_precision_policy()
    try:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in hidden.items():
            jax.config.update(name, value)
        assert (_globals(), _settings()) == (hidden, effective)
        assert active_precision_policy() is policy


def test_frozen_original_tensor_index_selector_retains_all_63_rows() -> None:
    inventory = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "internal_docs/precision/source_migration_inventory.json"
        ).read_text()
    )
    registrations = inventory["registrations"]
    rows = []
    for row in inventory["cases"]:
        origin = registrations[row[0]][0]
        if origin in {f"plugin::{name}" for name in _FAMILY_ROWS}:
            rows.append(
                [origin, row[1], row[2], row[4], inventory["contracts"][row[5]]]
            )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload).hexdigest() == _ORIGINAL_SHA256
    assert {(row[0].removeprefix("plugin::"), row[1]) for row in rows} == set(_ORIGINAL)
    assert len(rows) == 63 and len(_PRODUCT_ORIGINAL) == 56


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("family,ordinal", _PRODUCT_ORIGINAL)
def test_original_tensor_index_product_admission(
    family: str, ordinal: int, x64: bool
) -> None:
    origin = f"plugin::{family}"
    metadata = source.raw_source_case(origin, ordinal)
    provenance = source.SourceProvenance(origin, ordinal)
    if (family, ordinal) == ("concatenate", 4):
        # Declare the unchanged zero-argument signature before invoking any
        # helper that requires raw metadata to declare source inputs.
        scenario = source.SourceScenario(
            "raw-x64-on" if x64 else "raw-x64-off",
            x64,
            input_specs=(),
            input_values=(),
            provenance=provenance,
            require_reviewed_criteria=True,
        )
    else:
        scenario = source.raw_source_scenarios(metadata, provenance)[int(x64)]
    # Keep the actual symbolic declaration and the provenance-seeded source
    # sampling recipe. Only supply its required concrete B=2/5 sample extents.
    shapes = metadata.get("input_shapes", ())
    symbolic = any(dim == "B" for shape in shapes for dim in shape)
    batches = (2, 5) if symbolic else (None,)
    for batch in batches:
        sampled = (
            replace(
                scenario,
                concrete_shapes=tuple(
                    tuple(batch if dim == "B" else dim for dim in shape)
                    for shape in shapes
                ),
            )
            if batch is not None
            else scenario
        )
        assert sampled.model_dtype is source.PRESERVE_FACTORY_DTYPE
        assert sampled.criteria is scenario.criteria
        result = source.run_source_case(metadata, sampled, optimization=None)
        if not x64 and family == "gather" and ordinal in (2, 3):
            assert result.model is None and result.stage == "source-preparation"
            assert "64" in result.audit.detail
            return
        if (family, ordinal, x64) == ("dynamic_update_slice", 5, True):
            # The fixed raw source uses Python zero indices beside a strong
            # I32 input. Caller x64 makes those zeros I64 before export exists.
            detail = (
                "index arguments to dynamic_update_slice must be integers of "
                "the same type, got int64, int32, int64"
            )
            assert result.model is None and result.prepared is None
            assert result.stage == "source-preparation"
            assert result.audit.category == "original-source"
            assert result.audit.status is CheckStatus.FAILED
            assert result.audit.detail == detail
            assert sampled.provenance == provenance
            with source.source_configuration(sampled):
                native_specs = tuple(
                    jax.ShapeDtypeStruct(shape, dtype)
                    for shape, dtype in zip(
                        metadata["input_shapes"], metadata["input_dtypes"], strict=True
                    )
                )
                with pytest.raises(TypeError) as native:
                    jax.make_jaxpr(metadata["callable"])(*native_specs)
                assert str(native.value) == detail
            return
        assert result.model is not None, result.audit.detail
        assert result.audit.status is CheckStatus.VALIDATED
        assert result.schema is CheckStatus.VALIDATED
        assert result.raw_callback_status == "passed"
        assert result.numerical_status is CheckStatus.UNVERIFIED
        assert result.prepared is not None and result.prepared.scenario == sampled
        report = get_precision_report(result.model)
        assert report is not None and report.profile == "source-static-v1"
        assert all(
            check.status is CheckStatus.VALIDATED
            for check in report.checks
            if check.category in {"lowering", "representation"}
        )


def _layout(x: Any) -> Any:
    reshaped = jax.lax.reshape(x, (2, 3), dimensions=(1, 0))
    transposed = jax.lax.transpose(x, (1, 0))
    squeezed = jax.lax.squeeze(jax.lax.reshape(x, (1, 3, 2)), (0,))
    joined = jax.lax.concatenate((x, x), dimension=0)
    padded = jax.lax.pad(x, jnp.asarray(1, dtype=x.dtype), ((1, 0, 0), (0, 1, 0)))
    sliced = jax.lax.slice(padded, (1, 0), (4, 2), (2, 1))
    return (
        x,
        reshaped,
        transposed,
        squeezed,
        joined,
        padded,
        sliced,
        jax.lax.rev(x, (0, 1)),
    )


@pytest.mark.parametrize("dtype", _REAL_DTYPES)
@pytest.mark.parametrize("weak", [False, True])
def test_real_tensor_family_has_native_dtype_weakness_and_dataflow(
    dtype: Any, weak: bool
) -> None:
    values = (np.asarray([0, 1, 2, 3, 4, 5], dtype=dtype).reshape(3, 2),)
    _artifact(_layout, _specs([(3, 2)], dtype, weak=weak), feeds=(values,))


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int32, np.uint64, np.float16, np.float32, np.float64]
)
def test_tensor_composition_has_exact_native_cpu_values(dtype: Any) -> None:
    values = np.asarray([0, 1, 2, 3, 4, 5], dtype=dtype).reshape(3, 2)
    if np.dtype(dtype).kind == "f":
        values[0, 0] = -0.0
        if dtype is np.float64:
            values[0, 1] = 1 + 2**-40
    feeds = ((values,),)
    _, model, _, references = _artifact(_layout, _specs([(3, 2)], dtype), feeds=feeds)
    _accept(model, feeds, references, SampleCriteria("bitwise"))


@pytest.mark.parametrize("opset", [16, 23])
@pytest.mark.parametrize(
    "shape,target,dimensions",
    [((2, 0, 3), (0, 6), None), ((0, 2, 3), (3, 0, 2), (2, 0, 1))],
)
def test_empty_reshape_uses_literal_zero_extents(
    shape: Any, target: Any, dimensions: Any, opset: int
) -> None:
    def fn(x: Any) -> Any:
        return jax.lax.reshape(x, target, dimensions=dimensions)

    feeds = ((np.empty(shape, np.float32),),)
    _, model, _, references = _artifact(fn, _specs([shape]), feeds=feeds, opset=opset)
    _accept(model, feeds, references, SampleCriteria("bitwise"))


# These cases distinguish negative wrapping in the public JAX wrapper from
# primitive clipping, signed limits, unsigned high values and consumed indices.
_INDEX_CASES = (
    (np.int8, -8, True),
    (np.int8, -1, True),
    (np.int32, -100, True),
    (np.int32, -1, True),
    (np.int32, -1, False),
    (np.int32, 4, False),
    (np.int32, 100, False),
    (np.int64, -(2**63), True),
    (np.int64, 2**63 - 1, True),
    (np.uint32, 2**32 - 1, False),
    (np.uint64, 2**63, False),
    (np.uint64, 2**64 - 1, False),
)


@pytest.mark.parametrize("index_dtype,start,allow_negative", _INDEX_CASES)
def test_dynamic_index_clipping_preserves_native_values_and_direct_indices(
    index_dtype: Any, start: int, allow_negative: bool
) -> None:
    def fn(x: Any, update: Any, index: Any) -> Any:
        sliced = jax.lax.dynamic_slice(
            x, (index,), (2,), allow_negative_indices=allow_negative
        )
        written = jax.lax.dynamic_update_slice(
            x, update, (index,), allow_negative_indices=allow_negative
        )
        return x, sliced, written, index

    values = (
        np.asarray([10, 20, 30, 40, 50], np.int32),
        np.asarray([70, 80], np.int32),
        np.asarray(start, index_dtype),
    )
    specs = [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values]
    _, model, _, references = _artifact(fn, specs, feeds=(values,))
    _accept(model, (values,), references, SampleCriteria("bitwise"))


@pytest.mark.parametrize("opset", [23, 24, 26])
def test_multiaxis_update_including_tensorscatter_keeps_native_clipping(
    opset: int,
) -> None:
    def fn(x: Any, update: Any, index: Any) -> Any:
        zero = jnp.asarray(0, index.dtype)
        return jax.lax.dynamic_update_slice(
            x, update, (zero, index, zero), allow_negative_indices=False
        )

    values = (
        np.arange(30, dtype=np.float32).reshape(2, 5, 3),
        np.full((2, 2, 3), 99, np.float32),
        np.asarray(-1, np.int32),
    )
    _, model, _, references = _artifact(
        fn,
        [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values],
        feeds=(values,),
        opset=opset,
    )
    if opset == 23:
        _accept(model, (values,), references, SampleCriteria("bitwise"))
    else:
        from onnx.reference import ReferenceEvaluator

        scatter = next(
            node for node in model.graph.node if node.op_type == "TensorScatter"
        )
        assert (
            next(
                attribute.s
                for attribute in scatter.attribute
                if attribute.name == "mode"
            )
            == b"linear"
        )
        # Execute the actual standard operator semantics at both original
        # TensorScatter opsets. This is reference-backend evidence; it does not
        # substitute for separately recorded ORT kernel acceptance.
        feeds = {
            value.name: array
            for value, array in zip(model.graph.input, values, strict=True)
        }
        actual = ReferenceEvaluator(model).run(None, feeds)
        for result, expected in zip(actual, references[0][0], strict=True):
            np.testing.assert_array_equal(result, expected, strict=True)


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.float32, np.float64])
def test_scalar_dynamic_slice_and_update_are_exact(dtype: Any) -> None:
    def fn(x: Any, y: Any) -> Any:
        return jax.lax.dynamic_slice(x, (), ()), jax.lax.dynamic_update_slice(x, y, ())

    values = (np.asarray(1, dtype), np.asarray(0, dtype))
    _, model, _, references = _artifact(fn, _specs([(), ()], dtype), feeds=(values,))
    _accept(model, (values,), references, SampleCriteria("bitwise"))


@pytest.mark.parametrize(
    "mode", [jax.lax.GatherScatterMode.CLIP, jax.lax.GatherScatterMode.FILL_OR_DROP]
)
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64, np.uint64])
def test_point_gather_has_native_signed_unsigned_clip_and_fill(
    mode: Any, index_dtype: Any
) -> None:
    def fn(data: Any, indices: Any) -> Any:
        gathered = jax.lax.gather(
            data,
            indices,
            jax.lax.GatherDimensionNumbers(
                offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
            ),
            slice_sizes=(1, 2),
            mode=mode,
            fill_value=-9 if mode is jax.lax.GatherScatterMode.FILL_OR_DROP else None,
        )
        return data, gathered, indices

    indices = [0, 2, 3, 2**64 - 1] if index_dtype is np.uint64 else [-5, -1, 0, 2, 3]
    values = (
        np.asarray([[10, 11], [20, 21], [30, 31]], np.int32),
        np.asarray(indices, index_dtype).reshape(-1, 1),
    )
    _, model, _, references = _artifact(
        fn, [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values], feeds=(values,)
    )
    _accept(model, (values,), references, SampleCriteria("bitwise"))


def test_native_index_normalization_and_multiple_collapsed_coordinates() -> None:
    def fn(data: Any, indices: Any, coordinates: Any) -> Any:
        return (
            data[indices],
            jax.lax.gather(
                data,
                coordinates,
                jax.lax.GatherDimensionNumbers(
                    offset_dims=(), collapsed_slice_dims=(0, 1), start_index_map=(0, 1)
                ),
                slice_sizes=(1, 1),
            ),
            indices,
            coordinates,
        )

    values = (
        np.asarray([[1, 2], [3, 4], [5, 6]], np.float64),
        np.asarray([-3, -1, 0, 2], np.int64),
        np.asarray([[0, 1], [2, 0]], np.int32),
    )
    _, model, _, references = _artifact(
        fn, [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values], feeds=(values,)
    )
    _accept(model, (values,), references, SampleCriteria("bitwise"))


@pytest.mark.parametrize(
    "source_dtype,narrow_dtype", [(np.float64, np.float32), (np.float32, np.float16)]
)
def test_consumed_narrowing_through_tensor_views_keeps_original_direct_output(
    source_dtype: Any, narrow_dtype: Any
) -> None:
    def fn(x: Any) -> Any:
        narrowed = jax.lax.reshape(x.astype(narrow_dtype), (2, 3))
        return x, narrowed, jax.lax.transpose(narrowed, (1, 0)).astype(source_dtype)

    epsilon = 2**-40 if source_dtype is np.float64 else 2**-20
    values = (np.asarray([1 + epsilon, -0.0, 2, 3, 4, 5], source_dtype),)
    _, model, _, references = _artifact(
        fn, _specs([(6,)], source_dtype), feeds=(values,)
    )
    _accept(model, (values,), references, SampleCriteria("bitwise"))


@pytest.mark.parametrize(
    "dtype,length,opset",
    [
        (np.uint8, 260, 23),
        (np.float16, 2051, 23),
        (np.float16, 2051, 27),
        (jnp.bfloat16, 259, 23),
        (jnp.bfloat16, 259, 27),
    ],
)
def test_iota_preserves_native_cast_rounding_and_extents(
    dtype: Any, length: int, opset: int
) -> None:
    def fn() -> Any:
        return jax.lax.iota(dtype, length)

    _, model, _, references = _artifact(fn, [], feeds=((),), opset=opset)
    # Existing BF16 and opset27 rows explicitly skip numerics. Preserve that
    # target boundary while testing their actual product schema and source trace.
    if np.dtype(dtype).name != "bfloat16" and opset == 23:
        _accept(model, ((),), references, SampleCriteria("bitwise"))


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int32, np.uint32, np.int64, np.uint64]
)
def test_integer_index_arithmetic_preserves_native_modulo_overflow(dtype: Any) -> None:
    limit = np.iinfo(dtype)

    def fn(x: Any, y: Any) -> Any:
        return jax.lax.add(x, y), jax.lax.sub(x, y), x

    values = (np.asarray([limit.max, limit.min], dtype), np.asarray([1, 1], dtype))
    _, model, _, references = _artifact(
        fn, _specs([(2,), (2,)], dtype), feeds=(values,)
    )
    _accept(model, (values,), references, SampleCriteria("bitwise"))


@onnx_function(name="SourceTensorIndexComposition")
def _named_tensor(x: Any) -> Any:
    return jax.lax.rev(jax.lax.reshape(x, (2, 3)), (1,))


def test_named_and_jit_scopes_share_tensor_product_admission() -> None:
    def fn(x: Any) -> Any:
        return _named_tensor(x), jax.jit(lambda y: jax.lax.transpose(y, (1, 0)))(
            jax.lax.reshape(x, (2, 3))
        )

    values = (np.arange(6, dtype=np.float64) + np.float64(2**-40),)
    _, model, _, references = _artifact(
        fn, _specs([(6,)], np.float64), feeds=(values,), named=True
    )
    _accept(model, (values,), references, SampleCriteria("bitwise"))


def test_existing_atomic_leading_batch_tensor_contract_runs_multiple_batches() -> None:
    def fn(x: Any, index: Any) -> Any:
        batch = x.shape[0]
        zero = jnp.asarray(0, index.dtype)
        window = jax.lax.dynamic_slice(x, (zero, index, zero), (batch, 1, 2))
        update = jax.lax.dynamic_update_slice(x, window, (zero, index, zero))
        padded = jax.lax.pad(
            x, jnp.asarray(0, x.dtype), ((0, 0, 0), (1, 1, 0), (0, 0, 0))
        )
        return (
            jax.lax.reshape(x, (batch, 6)),
            jax.lax.transpose(x, (0, 2, 1)),
            jax.lax.concatenate((x, x), 1),
            jax.lax.slice(padded, (0, 1, 0), (batch, 4, 2)),
            jax.lax.squeeze(window, (1,)),
            update,
            jax.lax.rev(x, (0, 2)),
            jax.lax.broadcasted_iota(np.int32, (batch, 3), 1),
        )

    feeds = tuple(
        (
            np.arange(batch * 6, dtype=np.float32).reshape(batch, 3, 2),
            np.asarray(9, np.int32),
        )
        for batch in (2, 5)
    )
    specs = [*_specs([("B", 3, 2)]), jax.ShapeDtypeStruct((), np.int32)]
    _, model, _, references = _artifact(fn, specs, feeds=feeds)
    _accept(model, feeds, references, SampleCriteria("bitwise"))


@pytest.mark.parametrize(
    "primitive",
    [
        "reshape",
        "transpose",
        "squeeze",
        "concatenate",
        "pad",
        "slice",
        "dynamic_slice",
        "dynamic_update_slice",
        "iota",
        "rev",
        "gather",
    ],
)
def test_product_rejects_corrupted_actual_native_tensor_parameters(
    primitive: str,
) -> None:
    functions = {
        "reshape": (lambda x: jax.lax.reshape(x, (6,)), [(3, 2)]),
        "transpose": (lambda x: jax.lax.transpose(x, (1, 0)), [(3, 2)]),
        "squeeze": (lambda x: jax.lax.squeeze(x, (0,)), [(1, 3)]),
        "concatenate": (lambda x: jax.lax.concatenate((x, x), 0), [(3, 2)]),
        "pad": (
            lambda x: jax.lax.pad(x, np.float32(0), ((1, 0, 0), (0, 0, 0))),
            [(3, 2)],
        ),
        "slice": (lambda x: jax.lax.slice(x, (0, 0), (2, 2)), [(3, 2)]),
        "dynamic_slice": (lambda x, i: jax.lax.dynamic_slice(x, (i, i), (1, 1)), None),
        "dynamic_update_slice": (
            lambda x, y, i: jax.lax.dynamic_update_slice(x, y, (i, i)),
            None,
        ),
        "iota": (lambda: jax.lax.iota(np.int32, 4), []),
        "rev": (lambda x: jax.lax.rev(x, (0,)), [(3, 2)]),
        "gather": (
            lambda x: jax.lax.gather(
                x,
                np.asarray([[0]], np.int32),
                jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
                ),
                (1, 2),
            ),
            [(3, 2)],
        ),
    }
    fn, shapes = functions[primitive]
    if primitive == "dynamic_slice":
        specs = [
            jax.ShapeDtypeStruct((3, 2), np.float32),
            jax.ShapeDtypeStruct((), np.int32),
        ]
    elif primitive == "dynamic_update_slice":
        specs = [
            jax.ShapeDtypeStruct((3, 2), np.float32),
            jax.ShapeDtypeStruct((1, 1), np.float32),
            jax.ShapeDtypeStruct((), np.int32),
        ]
    else:
        specs = _specs(shapes)
    body = _body(jax.make_jaxpr(fn)(*specs))
    validate_source_jaxpr(body, 23)
    equations = list(body.eqns)
    index = next(
        i for i, eqn in enumerate(equations) if eqn.primitive.name == primitive
    )
    eqn = equations[index]
    defects = {
        "reshape": {"new_sizes": (7,)},
        "transpose": {"permutation": (0, 0)},
        "squeeze": {"dimensions": (1,)},
        "concatenate": {"dimension": 2},
        "pad": {"padding_config": ((1, 0, -1), (0, 0, 0))},
        "slice": {"strides": (0, 1)},
        "dynamic_slice": {"slice_sizes": (4, 2)},
        "iota": {"dtype": np.dtype(np.float64)},
        "rev": {"dimensions": (0, 0)},
    }
    if primitive == "dynamic_update_slice":
        equations[index] = eqn.replace(
            invars=(*eqn.invars[:2], eqn.invars[0], *eqn.invars[3:])
        )
    elif primitive == "gather":
        numbers = eqn.params["dimension_numbers"]._replace(start_index_map=(2,))
        equations[index] = eqn.replace(
            params={**eqn.params, "dimension_numbers": numbers}
        )
    else:
        equations[index] = eqn.replace(params={**eqn.params, **defects[primitive]})
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(body.replace(eqns=equations), 23)
    assert caught.value.primitive == primitive
    assert caught.value.category == "lowering-defect"


@pytest.mark.parametrize(
    "mode", [jax.lax.GatherScatterMode.CLIP, jax.lax.GatherScatterMode.FILL_OR_DROP]
)
def test_captured_uint64_point_indices_preserve_canonical_constant_consumers(
    mode: Any,
) -> None:
    indices = np.asarray([[2**64 - 1], [1]], np.uint64)
    threshold = np.asarray(2**63, np.uint64)

    def fn(data: Any) -> Any:
        gathered = jax.lax.gather(
            data,
            indices,
            jax.lax.GatherDimensionNumbers(
                offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
            ),
            slice_sizes=(1, 2),
            mode=mode,
            fill_value=-9 if mode is jax.lax.GatherScatterMode.FILL_OR_DROP else None,
        )
        # These consumers observe the original U64_MAX, while only the gather
        # index adaptation may saturate. Both gathered rows carry nonzero data.
        return data, gathered, indices, jax.lax.lt(indices, threshold)

    values = (np.asarray([[11, 12], [23, 24], [37, 38]], np.int32),)
    original_bytes = indices.tobytes()
    _, model, _, references = _artifact(fn, _specs([(3, 2)], np.int32), feeds=(values,))
    expected = references[0][0]
    np.testing.assert_array_equal(
        expected[1][0],
        np.asarray(
            [37, 38] if mode is jax.lax.GatherScatterMode.CLIP else [-9, -9], np.int32
        ),
        strict=True,
    )
    np.testing.assert_array_equal(expected[1][1], values[0][1], strict=True)
    assert expected[2].dtype == np.uint64 and expected[2].tobytes() == original_bytes
    np.testing.assert_array_equal(
        expected[3], np.asarray([[False], [True]]), strict=True
    )
    _accept(model, (values,), references, SampleCriteria("bitwise"))
    assert indices.dtype == np.uint64 and indices.tobytes() == original_bytes


def test_permuted_reshape_retains_original_native_batch_origin_for_all_consumers() -> (
    None
):
    def fn(x: Any) -> Any:
        reshaped = jax.lax.reshape(x, (x.shape[0], 6), dimensions=(1, 0, 2))
        return reshaped, jax.lax.rev(reshaped, (1,)), x

    feeds = tuple(
        (np.arange(batch * 6, dtype=np.float32).reshape(batch, 2, 3),)
        for batch in (2, 5)
    )
    _, model, report, references = _artifact(fn, _specs([("B", 2, 3)]), feeds=feeds)
    assert report.traced_outputs[0].shape == ("B", 6)
    assert report.exported_outputs[0].shape == ("B", 6)
    _accept(model, feeds, references, SampleCriteria("bitwise"))


@pytest.mark.parametrize(
    "update_shape,opset", [((2, 0), 23), ((2, 0), 26), ((3, 0), 26)]
)
def test_nonleading_empty_update_keeps_native_identity_and_original_values(
    update_shape: Any, opset: int
) -> None:
    def fn(x: Any, update: Any, index: Any) -> Any:
        return x, jax.lax.dynamic_update_slice(x, update, (index, index)), update, index

    values = (
        np.arange(12, dtype=np.int32).reshape(3, 4),
        np.empty(update_shape, np.int32),
        np.asarray(-1, np.int32),
    )
    _, model, _, references = _artifact(
        fn,
        [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values],
        feeds=(values,),
        opset=opset,
    )
    assert not any(
        node.op_type in {"ScatterND", "TensorScatter"} for node in model.graph.node
    )
    _accept(model, (values,), references, SampleCriteria("bitwise"))


@pytest.mark.parametrize("opset", [16, 23])
def test_boolean_padding_preserves_native_values_at_original_opsets(opset: int) -> None:
    def fn(x: Any) -> Any:
        padded = jax.lax.pad(x, np.bool_(True), ((1, -1, 0), (0, 1, 0)))
        return x, padded, jax.lax.bitwise_not(padded)

    values = (np.asarray([[False, True], [True, False], [False, False]], np.bool_),)
    _, model, _, references = _artifact(
        fn, _specs([(3, 2)], np.bool_), feeds=(values,), opset=opset
    )
    _accept(model, (values,), references, SampleCriteria("bitwise"))


@pytest.mark.parametrize("opset", [9, 13])
def test_legacy_reshape_keeps_pre_allowzero_schema_and_existing_zero_shape(
    opset: int,
) -> None:
    def fn(x: Any) -> Any:
        return jax.lax.reshape(x, (0, 6))

    artifact = to_onnx(fn, [jax.ShapeDtypeStruct((0, 2, 3), np.float32)], opset=opset)
    onnx.checker.check_model(artifact, full_check=True)
    assert (
        next(item.version for item in artifact.opset_import if item.domain == "")
        == opset
    )
    reshape = next(node for node in artifact.graph.node if node.op_type == "Reshape")
    assert all(attribute.name != "allowzero" for attribute in reshape.attribute)
    assert [
        dim.dim_value for dim in artifact.graph.output[0].type.tensor_type.shape.dim
    ] == [0, 6]


def test_zero_coordinate_gather_retains_specific_preexisting_lowerer_boundary() -> None:
    def fn(x: Any) -> Any:
        return jax.lax.gather(
            x,
            np.empty((4, 0), np.int32),
            jax.lax.GatherDimensionNumbers(
                offset_dims=(1, 2), collapsed_slice_dims=(), start_index_map=()
            ),
            slice_sizes=(3, 2),
        )

    native = _body(jax.make_jaxpr(fn)(*_specs([(3, 2)])))
    with pytest.raises(SourcePrecisionError, match="zero-coordinate gather") as caught:
        validate_source_jaxpr(native, 23)
    assert caught.value.category == "profile-exclusion"
    assert caught.value.primitive == "gather"


_COPY_CONTRACTS = (
    "388efa46557273913d53fefed73f56d125e9b3e57092e0f39b8da0bfb125feaf",
    "dc8aee5484bc787cb54c8d50fb53332c2737ca50d444df2a2ea53d82c83dcb42",
)


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("x64", [False, True])
def test_original_copy_rows_keep_raw_contract_and_product_admission(
    ordinal: int, x64: bool
) -> None:
    metadata = source.raw_source_case("plugin::copy", ordinal)
    unresolved: list[str] = []
    contract = canonical_contract(
        {
            key: value
            for key, value in metadata.items()
            if key not in {"testcase", "description", "context", "component"}
        },
        unresolved=unresolved,
    )
    assert not unresolved
    assert (
        hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == _COPY_CONTRACTS[ordinal]
    )
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::copy", ordinal)
    )[int(x64)]
    assert scenario.model_dtype is source.PRESERVE_FACTORY_DTYPE
    result = source.run_source_case(metadata, scenario, optimization=None)
    if ordinal == 1 and not x64:
        assert result.model is None and result.stage == "source-preparation"
        assert "64" in result.audit.detail
        return
    assert result.model is not None, result.audit.detail
    assert result.audit.status is CheckStatus.VALIDATED
    assert result.schema is CheckStatus.VALIDATED
    assert result.raw_callback_status == "passed"
    assert result.numerical_status is CheckStatus.UNVERIFIED


def _copy_observable(x: Any) -> Any:
    copied = jax.lax.copy_p.bind(x)
    return x, copied, jax.lax.eq(copied, jnp.asarray(1, x.dtype))


@pytest.mark.parametrize("dtype", _REAL_DTYPES)
@pytest.mark.parametrize("shape", ["scalar", "batch"])
def test_copy_product_preserves_real_scalar_and_shared_batch_contracts(
    dtype: Any, shape: str
) -> None:
    if shape == "scalar":
        specs = _specs([()], dtype, weak=True)
        feeds = ((np.asarray(1, dtype),),)
    else:
        specs = _specs([("B", 3)], dtype)
        feeds = tuple(
            (np.resize(np.asarray([0, 1, 2], dtype), (batch, 3)),) for batch in (2, 5)
        )
    _, _, report, _ = _artifact(_copy_observable, specs, feeds=feeds)
    original, copied = report.traced_outputs[:2]
    assert original == copied
    assert original.dtype == np.dtype(dtype).name
    assert original.weak_type is (shape == "scalar")


@pytest.mark.parametrize(
    "dtype,opset",
    [
        (np.float16, 23),
        (np.float32, 16),
        (np.float32, 23),
        (np.float32, 26),
        (np.float64, 23),
        (np.bool_, 16),
        (np.int32, 23),
        (np.uint64, 23),
    ],
)
def test_copy_native_consumer_and_direct_outputs_have_exact_cpu_values(
    dtype: Any,
    opset: int,
) -> None:
    if dtype is np.uint64:
        payload = np.asarray([0, 1, 2**64 - 1], dtype)
    elif dtype is np.int32:
        payload = np.asarray([np.iinfo(dtype).min, 1, np.iinfo(dtype).max], dtype)
    elif dtype is np.bool_:
        payload = np.asarray([False, True, False], dtype)
    else:
        epsilon = (
            2**-40 if dtype is np.float64 else 2**-20 if dtype is np.float32 else 2**-9
        )
        payload = np.asarray([-0.0, 1, 1 + epsilon], dtype)
    feeds = tuple((np.resize(payload, (batch, 3)),) for batch in (2, 5))
    _, model, _, references = _artifact(
        _copy_observable, _specs([("B", 3)], dtype), feeds=feeds, opset=opset
    )
    _accept(model, feeds, references, SampleCriteria("bitwise"))


def test_bfloat16_copy_consumer_preserves_native_rounding_and_direct_source() -> None:
    def fn(x: Any) -> Any:
        native = x.astype(jnp.bfloat16)
        copied = jax.lax.copy_p.bind(native)
        return x, copied.astype(np.float32), native.astype(np.float32)

    payload = np.asarray([-0.0, 1 + 2**-9, 1 + 2**-6], np.float32)
    feeds = tuple((np.resize(payload, (batch, 3)),) for batch in (2, 5))
    _, model, _, references = _artifact(fn, _specs([("B", 3)]), feeds=feeds)
    assert references[0][0][0][0, 1] != references[0][0][1][0, 1]
    _accept(model, feeds, references, SampleCriteria("bitwise"))


@pytest.mark.parametrize(
    "defect",
    [
        "arity",
        "outputs",
        "dtype",
        "shape",
        "weak",
        "weak-metadata",
        "scope",
        "unknown-parameter",
        "placement-none",
        "equation-effects",
        "complex",
    ],
)
def test_copy_rejects_malformed_native_contract_before_identity_lowering(
    defect: str,
) -> None:
    from jax import export

    (batch,) = export.symbolic_shape("B")
    shape = (batch, 3)
    spec = jax.ShapeDtypeStruct(shape, np.float32)
    output = spec
    params: dict[str, Any] = {}
    if defect == "dtype":
        output = jax.ShapeDtypeStruct(shape, np.float16)
    elif defect == "shape":
        output = jax.ShapeDtypeStruct((batch, 4), np.float32)
    elif defect == "weak":
        output = jax.ShapeDtypeStruct(shape, np.float32, weak_type=True)
    elif defect == "scope":
        (foreign,) = export.symbolic_shape("B")
        output = jax.ShapeDtypeStruct((foreign, 3), np.float32)
    elif defect == "unknown-parameter":
        params["layout"] = None
    elif defect == "placement-none":
        params["out_sharding"] = None
    elif defect == "complex":
        spec = output = jax.ShapeDtypeStruct((2, 3), np.complex64)
    graph = _typed_graph(
        "copy", [spec], [output, output] if defect == "outputs" else [output], params
    )
    if defect == "arity":
        graph.eqns[0].invars = []
    elif defect == "weak-metadata":
        graph.eqns[0].outvars[0].aval = SimpleNamespace(
            shape=shape, dtype=np.dtype(np.float32), weak_type=1
        )
    elif defect == "equation-effects":
        graph.eqns[0].effects = ("unreviewed effect",)
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.category == (
        "profile-exclusion"
        if defect in {"scope", "equation-effects", "complex"}
        else "lowering-defect"
    )
    assert caught.value.primitive == "copy"


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("opset", [16, 23])
def test_atomic_dimension_value_keeps_native_origin_scalar_and_consumers(
    x64: bool, opset: int
) -> None:
    def fn(x: Any) -> Any:
        batch = jnp.asarray(x.shape[0])
        consumed = batch.astype(np.float32) + np.float32(0.5)
        return x, batch, consumed

    specs = _specs([("B", 3)])
    feeds = tuple(
        (np.arange(batch * 3, dtype=np.float32).reshape(batch, 3),) for batch in (2, 5)
    )
    with jax.enable_x64(x64):
        native = _body(jax.make_jaxpr(fn)(*specs))
        dimension = next(e for e in native.eqns if e.primitive.name == "dim_as_value")
        assert dimension.invars == [] and set(dimension.params) == {"dim"}
        assert dimension.params["dim"] is specs[0].shape[0]
        assert dimension.outvars[0].aval.shape == ()
        assert dimension.outvars[0].aval.weak_type is True
        _, model, report, references = _artifact(fn, specs, feeds=feeds, opset=opset)
        assert report.traced_outputs[1].dtype == ("int64" if x64 else "int32")
        assert report.traced_outputs[1].shape == ()
        assert report.traced_outputs[1].weak_type is True
        for batch, (eager, compiled) in zip((2, 5), references, strict=True):
            for actual in (eager, compiled):
                assert actual[1].item() == batch
                assert actual[2].item() == batch + 0.5
        _accept(model, feeds, references, SampleCriteria("bitwise"))


@pytest.mark.parametrize(
    "defect",
    [
        "arity",
        "outputs",
        "dtype",
        "nonnative-integer",
        "shape",
        "weak",
        "weak-metadata",
        "no-origin",
        "foreign-scope",
        "foreign-label",
        "printed-label",
        "constant",
        "derived",
        "constrained",
        "missing-parameter",
        "unknown-parameter",
        "placement-none",
        "equation-effects",
        "scope-effects",
    ],
)
def test_dimension_value_rejects_unproved_origin_and_malformed_native_contract(
    defect: str,
) -> None:
    from jax import export

    (batch,) = export.symbolic_shape("B")
    formal = jax.ShapeDtypeStruct((batch, 3), np.float32)
    output = jax.ShapeDtypeStruct((), np.int64, weak_type=True)
    params: dict[str, Any] = {"dim": batch}
    if defect in {"dtype", "nonnative-integer"}:
        output = jax.ShapeDtypeStruct(
            (), np.float32 if defect == "dtype" else np.int16, weak_type=True
        )
    elif defect == "shape":
        output = jax.ShapeDtypeStruct((1,), np.int64, weak_type=True)
    elif defect == "weak":
        output = jax.ShapeDtypeStruct((), np.int64, weak_type=False)
    elif defect == "no-origin":
        formal = jax.ShapeDtypeStruct((2, 3), np.float32)
    elif defect == "foreign-scope":
        (params["dim"],) = export.symbolic_shape("B")
    elif defect == "foreign-label":
        (params["dim"],) = export.symbolic_shape("C", scope=batch.scope)
    elif defect == "printed-label":
        params["dim"] = "B"
    elif defect == "constant":
        params["dim"] = 2
    elif defect == "derived":
        params["dim"] = batch * 2
    elif defect == "constrained":
        (params["dim"],) = export.symbolic_shape("B", constraints=("B >= 2",))
    elif defect == "missing-parameter":
        params = {}
    elif defect == "unknown-parameter":
        params["axis"] = 0
    elif defect == "placement-none":
        params["out_sharding"] = None
    graph = _typed_graph(
        "dim_as_value",
        [formal],
        [output, output] if defect == "outputs" else [output],
        params,
    )
    if defect != "arity":
        graph.eqns[0].invars = []
    if defect == "weak-metadata":
        graph.eqns[0].outvars[0].aval = SimpleNamespace(
            shape=(), dtype=np.dtype(np.int64), weak_type=1
        )
    elif defect == "equation-effects":
        graph.eqns[0].effects = ("unreviewed effect",)
    elif defect == "scope-effects":
        graph.effects = ("unreviewed effect",)
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.primitive == (
        None if defect == "scope-effects" else "dim_as_value"
    )
    assert caught.value.category == (
        "profile-exclusion"
        if defect
        in {
            "no-origin",
            "foreign-scope",
            "foreign-label",
            "printed-label",
            "constant",
            "derived",
            "constrained",
            "equation-effects",
            "scope-effects",
        }
        else "lowering-defect"
    )


@pytest.mark.parametrize(
    "shape,axis,dtype,weak",
    [
        ((), None, np.float16, True),
        ((1, 3, 1), None, np.float64, False),
        ((1, 3, 1), (0, -1), np.int32, True),
        (("B", 1, 3), 1, np.float32, False),
        (("B", 1, 3), (1,), np.bool_, False),
        (("B", 1, 3), -2, np.float16, True),
    ],
)
def test_jnp_squeeze_preserves_native_trace_direct_values_and_consumers(
    shape: Any, axis: Any, dtype: Any, weak: bool
) -> None:
    def fn(x: Any) -> Any:
        squeezed = jnp.squeeze(x, axis=axis)
        return x, squeezed, squeezed.astype(np.float32)

    payload = np.asarray([0, 1, 1 + 2**-40], dtype)
    batches = (2, 5) if "B" in shape else (None,)
    feeds = tuple(
        (np.resize(payload, tuple(batch if dim == "B" else dim for dim in shape)),)
        for batch in batches
    )
    _, model, report, references = _artifact(
        fn, _specs([shape], dtype, weak=weak), feeds=feeds
    )
    assert report.traced_outputs[1].dtype == np.dtype(dtype).name
    assert report.traced_outputs[1].weak_type is weak
    if dtype is np.float64:
        assert (
            references[0][0][1].reshape(-1)[-1] != references[0][0][2].reshape(-1)[-1]
        )
    _accept(model, feeds, references, SampleCriteria("bitwise"))


def test_jnp_squeeze_retains_native_errors_and_existing_legacy_binding() -> None:
    spec = jax.ShapeDtypeStruct((1, 3, 1), np.float32)
    for axis in ((0, 0), 1, 3):

        def fn(x: Any) -> Any:
            return jnp.squeeze(x, axis=axis)

        with pytest.raises((ValueError, IndexError)) as native:
            jax.make_jaxpr(fn)(spec)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            with pytest.raises(type(native.value)) as activated:
                jax.make_jaxpr(fn)(spec)
            assert str(activated.value) == str(native.value)
    with precision_policy_scope(PrecisionPolicy.LEGACY), _activate_plugin_worlds():
        legacy = _body(jax.make_jaxpr(lambda x: jnp.squeeze(x))(spec))
        assert [eqn.primitive.name for eqn in legacy.eqns] == ["jax.numpy.squeeze"]


@onnx_function(name="SourceLocalDimensionX64Off")
def _named_dimension_local_x64_off(x: Any) -> Any:
    with jax.enable_x64(False):
        batch = jnp.asarray(x.shape[0])
        consumed = batch.astype(np.float32) + np.float32(0.5)
        return x, batch, consumed


@onnx_function(name="SourceLocalDimensionX64On")
def _named_dimension_local_x64_on(x: Any) -> Any:
    with jax.enable_x64(True):
        batch = jnp.asarray(x.shape[0])
        consumed = batch.astype(np.float32) + np.float32(0.5)
        return x, batch, consumed


@pytest.mark.parametrize("local_x64", [False, True])
@pytest.mark.parametrize("named", [False, True])
def test_dimension_value_keeps_opposite_local_x64_trace_and_restores_caller(
    local_x64: bool, named: bool
) -> None:
    native_fn = (
        _named_dimension_local_x64_on if local_x64 else _named_dimension_local_x64_off
    )

    def fn(x: Any) -> Any:
        if named:
            return (
                _named_dimension_local_x64_on(x)
                if local_x64
                else _named_dimension_local_x64_off(x)
            )
        return native_fn(x)

    specs = _specs([("B", 3)])
    feeds = tuple(
        (np.arange(batch * 3, dtype=np.float32).reshape(batch, 3),) for batch in (2, 5)
    )
    before = _globals(), _settings()
    with jax.enable_x64(not local_x64):
        caller = _globals(), _settings()
        native = _body(jax.make_jaxpr(fn)(*specs))
        dimension = next(e for e in native.eqns if e.primitive.name == "dim_as_value")
        expected_dtype = np.dtype(np.int64 if local_x64 else np.int32)
        assert np.dtype(dimension.outvars[0].aval.dtype) == expected_dtype
        assert dimension.outvars[0].aval.shape == ()
        assert dimension.outvars[0].aval.weak_type is True
        assert dimension.params["dim"] is specs[0].shape[0]
        assert (_globals(), _settings()) == caller
        _, model, report, references = _artifact(fn, specs, feeds=feeds, named=named)
        scalar = report.traced_outputs[1]
        assert scalar.dtype == expected_dtype.name
        assert scalar.shape == () and scalar.weak_type is True
        for batch, (eager, compiled) in zip((2, 5), references, strict=True):
            for actual in (eager, compiled):
                assert actual[1].dtype == expected_dtype
                assert actual[1].item() == batch
                assert actual[2].item() == batch + 0.5
        _accept(model, feeds, references, SampleCriteria("bitwise"))
        assert (_globals(), _settings()) == caller
    assert (_globals(), _settings()) == before
