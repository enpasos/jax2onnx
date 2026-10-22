# tests/extra_tests/converter/test_source_positive_symbolic_cast.py

"""Positive symbolic casts retain explicit rounding and each original consumer."""

from collections.abc import Callable
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
from jax import export
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import Primitive, Var
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    _Site,
    _Tensor,
    _validate_conversion,
    validate_source_jaxpr,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_model_native_bindings import (
    _contexts,
    _fresh,
    _payload,
)
from tests.extra_tests.converter.test_source_mlp_runtime_acceptance import (
    _runtime_configuration,
)
from tests.extra_tests.converter.test_source_symbolic_contraction import _isolated_graph


def _roundtrip(value: Any) -> Any:
    narrowed = jax.lax.convert_element_type(value, np.float32)
    rounded = jax.lax.convert_element_type(narrowed, np.float64)
    sibling = jax.lax.transpose(value, tuple(reversed(range(value.ndim))))
    return value, sibling, narrowed, rounded


@onnx_function(name="PositiveSymbolicRoundTrip")
def _named_roundtrip(value: Any) -> Any:
    return _roundtrip(value)


def _call_named_roundtrip(value: Any) -> Any:
    return _named_roundtrip(value)


def _to_integer(value: Any) -> Any:
    return jax.lax.convert_element_type(value, np.int32), value


def _to_double(value: Any) -> Any:
    return jax.lax.convert_element_type(value, np.float64), value


def _shape(layout: tuple[Any, ...], batch: Any) -> tuple[Any, ...]:
    return tuple(batch if dimension == "B" else dimension for dimension in layout)


def _public_shape(layout: tuple[Any, ...], batch: Any) -> tuple[Any, ...]:
    return (batch, *(dimension for dimension in layout if dimension != "B"))


def _layout_function(
    function: Callable[..., Any], layout: tuple[Any, ...], *, roundtrip: bool
) -> Callable[..., Any]:
    axis = layout.index("B")
    permutation = list(range(len(layout)))
    permutation.insert(axis, permutation.pop(0))
    restore = tuple(permutation.index(index) for index in range(len(layout)))
    sibling_axis = len(layout) - 1 - axis
    restore_sibling = (
        sibling_axis,
        *(index for index in range(len(layout)) if index != sibling_axis),
    )

    def call(value: Any) -> Any:
        internal = jax.lax.transpose(value, tuple(permutation))
        returned = function(internal)
        return tuple(
            jax.lax.transpose(
                leaf, restore_sibling if roundtrip and index == 1 else restore
            )
            for index, leaf in enumerate(returned)
        )

    return call


def _equations(closed: Any) -> Any:
    inner = getattr(closed, "jaxpr", None)
    body = inner if inner is not None and inner is not closed else closed
    for equation in body.eqns:
        yield equation
        for value in equation.params.values():
            children = value if isinstance(value, (tuple, list)) else (value,)
            for child in children:
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    yield from _equations(child)


def _require_internal_cast_axis(trace: Any, axis: int) -> None:
    casts = [
        equation
        for equation in _equations(trace)
        if equation.primitive is jax.lax.convert_element_type_p
    ]
    assert len(casts) == 2
    for equation in casts:
        assert [
            index
            for index, dimension in enumerate(equation.invars[0].aval.shape)
            if not isinstance(dimension, (int, np.integer))
        ] == [axis]
        assert equation.invars[0].aval.shape == equation.outvars[0].aval.shape


def _rounding_input(shape: tuple[int, ...]) -> np.ndarray:
    # Exact predeclared FP64 inputs straddle FP32 round-to-nearest boundaries.
    seeds = np.asarray(
        [
            1.0 + 2.0**-24,
            1.0 + 3.0 * 2.0**-24,
            -1.0 - 2.0**-24,
            -1.0 - 3.0 * 2.0**-24,
            2.0**24 + 1.0,
            -(2.0**24 + 1.0),
            0.0,
            -0.0,
        ],
        dtype=np.float64,
    )
    return np.resize(seeds, shape)


def _expected_roundtrip(value: np.ndarray) -> tuple[np.ndarray, ...]:
    narrowed = value.astype(np.float32)
    sibling = value.transpose((0, *reversed(range(1, value.ndim))))
    return value, sibling, narrowed, narrowed.astype(np.float64)


def _same(actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...]) -> None:
    assert len(actual) == len(expected), "native cast output arity differs"
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == reference.dtype, "native cast output dtype differs"
        assert value.shape == reference.shape, "native cast output shape differs"
        assert value.tobytes(order="C") == reference.tobytes(
            order="C"
        ), "native cast output bytes differ"


def _retain(returned: Any, path: Path, stage: str) -> tuple[np.ndarray, ...]:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = path.with_suffix(".json")
    event.write_text(json.dumps(dict(stage=stage, status="returned")))
    leaves = jax.tree.leaves(jax.device_get(returned))
    np.savez(path, **{f"leaf_{index}": leaf for index, leaf in enumerate(leaves)})
    event.write_text(
        json.dumps(dict(stage=stage, status="archived", leaves=len(leaves)))
    )
    return tuple(np.asarray(value) for value in leaves)


def _native(
    function: Callable[..., Any],
    value: np.ndarray,
    expected: tuple[np.ndarray, ...],
    directory: Path,
) -> tuple[tuple[np.ndarray, ...], ...]:
    compiled = jax.jit(_fresh(function))
    references = []
    before = value.tobytes(order="C")
    for stage, call in (("fresh", compiled), ("warmed", compiled), ("eager", function)):
        try:
            returned = call(value)
        except BaseException as exc:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"native-{stage}-exception.json").write_text(
                json.dumps(
                    dict(status="raised", type=type(exc).__qualname__, message=str(exc))
                )
            )
            raise
        actual = _retain(returned, directory / f"native-{stage}.npz", stage)
        _same(actual, expected)
        references.append(actual)
    assert value.tobytes(order="C") == before
    return tuple(references)


def _cpu(model: Any, samples: list[Any], directory: Path) -> None:
    blob = model.SerializeToString()
    digest = hashlib.sha256(blob).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source.onnx").write_bytes(blob)
    onnx.checker.check_model(model, full_check=True)
    for stage, profile in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        phase = "session"
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = profile
            session = ort.InferenceSession(
                blob, options, providers=["CPUExecutionProvider"]
            )
            session.disable_fallback()
            assert len(session.get_inputs()) == 1
            for index, (value, expected, references) in enumerate(samples):
                phase = f"run-{index}"
                returned = session.run(None, {session.get_inputs()[0].name: value})
                actual = _retain(
                    returned, directory / f"cpu-{stage}-{index}.npz", stage
                )
                phase = f"comparison-{index}"
                _same(actual, expected)
                assert len(references) == 3
                for reference in references:
                    _same(actual, reference)
                assert session.get_providers() == ["CPUExecutionProvider"]
            assert session.get_session_options().graph_optimization_level == profile
            assert hashlib.sha256(model.SerializeToString()).hexdigest() == digest
        except BaseException as exc:
            (directory / f"cpu-{stage}-exception.json").write_text(
                json.dumps(
                    dict(
                        status="raised",
                        stage=phase,
                        type=type(exc).__qualname__,
                        message=str(exc),
                    )
                )
            )
            raise


def _cast_chain(owner: Any, layout: tuple[Any, ...]) -> tuple[Any, Any]:
    assert len(owner.input) == 1
    producers = {name: node for node in owner.node for name in node.output}
    input_name = (
        owner.input[0] if isinstance(owner.input[0], str) else owner.input[0].name
    )
    outputs = tuple(
        value if isinstance(value, str) else value.name for value in owner.output
    )
    rank = len(layout)
    identity = tuple(range(rank))

    def target(node: Any) -> int | None:
        return next(
            (attribute.i for attribute in node.attribute if attribute.name == "to"),
            None,
        )

    narrowed = [
        node
        for node in owner.node
        if node.op_type == "Cast" and target(node) == onnx.TensorProto.FLOAT
    ]
    widened = [
        node
        for node in owner.node
        if node.op_type == "Cast" and target(node) == onnx.TensorProto.DOUBLE
    ]
    assert len(narrowed) == len(widened) == 1
    first, second = narrowed[0], widened[0]

    def provenance(
        name: str, ancestors: frozenset[str] = frozenset()
    ) -> tuple[Any, ...]:
        assert name not in ancestors, "cyclic cast provenance"
        if name == input_name:
            return identity, ()
        assert name in producers, "cast provenance is not rooted at the actual formal"
        node = producers[name]
        assert len(node.input) == len(node.output) == 1
        permutation, casts = provenance(node.input[0], ancestors | {name})
        if node.op_type == "Identity":
            assert not node.attribute
            return permutation, casts
        if node.op_type == "Transpose":
            assert all(attribute.name == "perm" for attribute in node.attribute)
            assert len(node.attribute) <= 1
            axes = (
                tuple(node.attribute[0].ints)
                if node.attribute
                else tuple(reversed(identity))
            )
            assert len(axes) == rank and sorted(axes) == list(
                identity
            ), "invalid transpose permutation"
            return tuple(permutation[axis] for axis in axes), casts
        assert node.op_type == "Cast" and node in (
            first,
            second,
        ), "unaudited cast provenance producer"
        return permutation, (*casts, node.output[0])

    # The same native pointwise casts may commute with layout transposes. Trace
    # exact formal identities and composed permutations, never equal-shape names.
    assert provenance(second.input[0])[1] == (first.output[0],)
    sibling = (
        tuple(reversed(identity))
        if isinstance(owner, onnx.FunctionProto)
        else (0, *reversed(range(1, rank)))
    )
    expected = (
        (identity, ()),
        (sibling, ()),
        (identity, (first.output[0],)),
        (identity, (first.output[0], second.output[0])),
    )
    assert len(outputs) == 4
    assert (
        tuple(provenance(name) for name in outputs) == expected
    ), "cast output provenance or layout differs"
    return first, second


def _bypass_narrowing(owner: Any, first: Any, second: Any) -> None:
    producers = {name: node for node in owner.node for name in node.output}
    layout_chain = []
    name = second.input[0]
    while name != first.output[0]:
        node = producers[name]
        assert node.op_type in {"Identity", "Transpose"}
        layout_chain.append(node)
        name = node.input[0]
    source = first.input[0]
    copies = []
    for node in reversed(layout_chain):
        cloned = copy.deepcopy(node)
        cloned.input[0] = source
        cloned.output[0] += "_without_narrow"
        assert cloned.output[0] not in producers
        source = cloned.output[0]
        copies.append(cloned)
    second.input[0] = source
    if copies:
        nodes = list(owner.node)
        position = next(index for index, node in enumerate(nodes) if node == second)
        nodes[position:position] = copies
        del owner.node[:]
        owner.node.extend(nodes)


def _roundtrip_owner(model: Any, scope: str) -> Any:
    if scope == "inline":
        return model.graph
    definitions = [
        function
        for function in model.functions
        if "PositiveSymbolicRoundTrip" in function.name
    ]
    assert len(definitions) == 1
    owner = definitions[0]
    assert len(owner.input) == 1 and len(owner.output) == 4
    assert any(
        node.domain == owner.domain and node.op_type == owner.name
        for node in model.graph.node
    )
    return owner


@pytest.mark.parametrize("layout", ((2, "B", 4), (2, 4, "B"), (1, "B", 2, 4, 4)))
@pytest.mark.parametrize("scope", ("inline", "named"))
def test_positive_symbolic_cast_rounding_direct_siblings_and_function_ports(
    layout: tuple[Any, ...], scope: str, tmp_path: Path
) -> None:
    with _runtime_configuration(True):
        batch = export.symbolic_shape("B")[0]
        spec = jax.ShapeDtypeStruct(_public_shape(layout, batch), np.float64)
        selected = _roundtrip if scope == "inline" else _call_named_roundtrip
        function = _layout_function(selected, layout, roundtrip=True)
        inline_trace_function = _layout_function(_roundtrip, layout, roundtrip=True)
        native = jax.make_jaxpr(_fresh(inline_trace_function))(spec)
        before = typed_trace_manifest(native)
        validate_source_jaxpr(native, 23)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(_fresh(inline_trace_function))(spec)
        assert typed_trace_manifest(activated) == before
        _require_internal_cast_axis(native, layout.index("B"))
        _require_internal_cast_axis(activated, layout.index("B"))
        assert _contexts(activated) == _contexts(native)
        assert [_payload(value) for value in activated.consts] == [
            _payload(value) for value in native.consts
        ]
        samples = []
        for size in (2, 5):
            value = _rounding_input(_public_shape(layout, size))
            expected = _expected_roundtrip(value)
            assert expected[0].tobytes() != expected[3].tobytes()
            references = _native(function, value, expected, tmp_path / f"B{size}")
            samples.append((value, expected, references))
        model = ui._to_onnx(
            function, [spec], precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        owner = _roundtrip_owner(model, scope)
        _cast_chain(owner, layout)
        if scope == "named":
            infos = {info.name: info for info in owner.value_info}
            shapes = [layout, layout, tuple(reversed(layout)), layout, layout]
            dtypes = [
                onnx.TensorProto.DOUBLE,
                onnx.TensorProto.DOUBLE,
                onnx.TensorProto.DOUBLE,
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.DOUBLE,
            ]
            for name, shape, dtype in zip(
                (*owner.input, *owner.output), shapes, dtypes, strict=True
            ):
                tensor = infos[name].type.tensor_type
                assert tensor.elem_type == dtype
                assert (
                    tuple(
                        dimension.dim_param or dimension.dim_value
                        for dimension in tensor.shape.dim
                    )
                    == shape
                )
        _cpu(model, samples, tmp_path)
        assert typed_trace_manifest(native) == before


@pytest.mark.parametrize("layout", ((2, "B", 4), (2, 4, "B")))
@pytest.mark.parametrize("conversion", ("float-to-int", "int-to-float"))
def test_positive_symbolic_existing_discrete_casts_and_saturation(
    layout: tuple[Any, ...], conversion: str, tmp_path: Path
) -> None:
    with _runtime_configuration(True):
        batch = export.symbolic_shape("B")[0]
        samples = []
        if conversion == "float-to-int":
            function = _to_integer
            dtype = np.float64
            seeds = np.asarray(
                [
                    -np.inf,
                    -2147483649.0,
                    -3.9,
                    -0.0,
                    0.0,
                    3.9,
                    2147483648.0,
                    np.inf,
                    np.nan,
                ],
                dtype=dtype,
            )
            answers = np.asarray(
                [-2147483648, -2147483648, -3, 0, 0, 3, 2147483647, 2147483647, 0],
                dtype=np.int32,
            )
        else:
            function = _to_double
            dtype = np.int32
            seeds = np.asarray(
                [-2147483648, -16777217, -1, 0, 1, 16777217, 2147483647], dtype=dtype
            )
            answers = np.asarray(
                [-2147483648, -16777217, -1, 0, 1, 16777217, 2147483647],
                dtype=np.float64,
            )
        function = _layout_function(function, layout, roundtrip=False)
        spec = jax.ShapeDtypeStruct(_public_shape(layout, batch), dtype)
        native = jax.make_jaxpr(_fresh(function))(spec)
        validate_source_jaxpr(native, 23)
        for size in (2, 5):
            shape = _public_shape(layout, size)
            value = np.resize(seeds, shape)
            expected = np.resize(answers, shape), value
            references = _native(function, value, expected, tmp_path / f"B{size}")
            samples.append((value, expected, references))
        model = ui._to_onnx(
            function, [spec], precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        assert any(node.op_type == "Cast" for node in model.graph.node)
        if conversion == "float-to-int":
            assert any(node.op_type == "Where" for node in model.graph.node)
        _cpu(model, samples, tmp_path)


@pytest.mark.parametrize("scope", ("inline", "named"))
def test_bypassing_narrowing_boundary_fails_same_exact_criterion(
    scope: str, tmp_path: Path
) -> None:
    with _runtime_configuration(True):
        batch = export.symbolic_shape("B")[0]
        selected = _roundtrip if scope == "inline" else _call_named_roundtrip
        function = _layout_function(selected, (2, "B", 4), roundtrip=True)
        spec = jax.ShapeDtypeStruct((batch, 2, 4), np.float64)
        value = _rounding_input((2, 2, 4))
        expected = _expected_roundtrip(value)
        references = _native(function, value, expected, tmp_path / "native")
        original = ui._to_onnx(
            function, [spec], precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        corrupted = copy.deepcopy(original)
        owner = _roundtrip_owner(corrupted, scope)
        first, second = _cast_chain(owner, (2, "B", 4))
        _bypass_narrowing(owner, first, second)
        onnx.checker.check_model(corrupted, full_check=True)
        with pytest.raises(AssertionError, match="output bytes differ"):
            _cpu(corrupted, [(value, expected, references)], tmp_path / "corrupted")


@pytest.mark.parametrize(
    "fault",
    (
        "foreign-identity",
        "effects",
        "unknown",
        "sharding",
        "out-sharding",
        "missing-dtype",
        "dtype-string",
        "wrong-dtype",
        "nonboolean-weakness",
        "wrong-weakness",
        "shape",
        "axis",
        "scope",
        "extra-input",
        "missing-output",
    ),
)
def test_positive_symbolic_cast_rejects_hostile_native_metadata(fault: str) -> None:
    with _runtime_configuration(True):
        batch = export.symbolic_shape("B")[0]
        spec = jax.ShapeDtypeStruct((2, batch, 4), np.float64)
        native = jax.make_jaxpr(_roundtrip)(spec)
        before = typed_trace_manifest(native)
        original_contexts = _contexts(native)
        original_constants = [_payload(value) for value in native.consts]
        original_ssa = tuple(
            (
                id(equation),
                tuple(map(id, equation.invars)),
                tuple(map(id, equation.outvars)),
                id(equation.params),
            )
            for equation in _equations(native)
        )
        graph = _isolated_graph(native)
        eqn = next(
            e for e in graph.eqns if e.primitive is jax.lax.convert_element_type_p
        )
        if fault == "foreign-identity":
            eqn.primitive = Primitive("convert_element_type")
        elif fault == "effects":
            eqn.effects = {object()}
        elif fault == "unknown":
            eqn.params["unowned"] = None
        elif fault in {"sharding", "out-sharding"}:
            eqn.params[fault.replace("-", "_")] = object()
        elif fault == "missing-dtype":
            del eqn.params["new_dtype"]
        elif fault == "dtype-string":
            eqn.params["new_dtype"] = "float32"
        elif fault == "wrong-dtype":
            eqn.params["new_dtype"] = np.dtype(np.float64)
        elif fault == "nonboolean-weakness":
            eqn.params["weak_type"] = 0
        elif fault == "wrong-weakness":
            eqn.params["weak_type"] = True
        elif fault in {"shape", "axis", "scope"}:
            shape = (2, batch, 3) if fault == "shape" else (batch, 2, 4)
            if fault == "scope":
                foreign = export.symbolic_shape("B")[0]
                shape = (2, foreign, 4)
            replacement = Var(eqn.outvars[0].aval.update(shape=shape))
            eqn.outvars = [replacement]
        elif fault == "extra-input":
            eqn.invars = [*eqn.invars, *eqn.invars]
        else:
            eqn.outvars = []
        with pytest.raises(SourcePrecisionError):
            validate_source_jaxpr(graph, 23)
        assert typed_trace_manifest(native) == before
        assert _contexts(native) == original_contexts
        assert [_payload(value) for value in native.consts] == original_constants
        assert (
            tuple(
                (
                    id(equation),
                    tuple(map(id, equation.invars)),
                    tuple(map(id, equation.outvars)),
                    id(equation.params),
                )
                for equation in _equations(native)
            )
            == original_ssa
        )


def test_conversion_validator_rejects_inconsistent_symbol_axis() -> None:
    with _runtime_configuration(True):
        batch = export.symbolic_shape("B")[0]
        trace = jax.make_jaxpr(_roundtrip)(
            jax.ShapeDtypeStruct((2, batch, 4), np.float64)
        )
        eqn = next(
            e for e in trace.eqns if e.primitive is jax.lax.convert_element_type_p
        )
        symbol = ("B", object())
        source = _Tensor("float64", (2, "B", 4), symbol, 1)
        target = _Tensor("float32", (2, "B", 4), symbol, 0)
        with pytest.raises(
            SourcePrecisionError, match="cast attribute, scope or weakness"
        ):
            _validate_conversion(
                eqn, [source], [target], _Site("cast-axis", 23, "convert_element_type")
            )


@pytest.mark.parametrize("layout", ((2, "B", 0), (0, "B", 4), (2, 0, "B")))
def test_zero_tail_symbolic_cast_remains_explicit_unfinished_work(
    layout: tuple[Any, ...],
) -> None:
    with _runtime_configuration(True):
        batch = export.symbolic_shape("B")[0]
        native = jax.make_jaxpr(_to_double)(
            jax.ShapeDtypeStruct(_shape(layout, batch), np.float32)
        )
        with pytest.raises(SourcePrecisionError, match="no audited owner"):
            validate_source_jaxpr(native, 23)
