# tests/extra_tests/converter/test_source_native_top_k.py

"""Native TopK total-order words and indices, without floating tie approximations."""

from collections.abc import Callable
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
from jax import export as jax_export
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, precision, user_interface as ui
from jax2onnx._compat.jax import (
    Primitive,
    Var,
    native_top_k_parameter_names,
    native_top_k_primitive,
)
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    _Site,
    _tensor,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_top_k import validate_native_top_k
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


def _top_k(x: Any) -> Any:
    return jax.lax.top_k(x, 2)


@onnx_function(name="NativeTotalOrderTopK")
def _named_top_k(x: Any) -> Any:
    return _top_k(x)


def _call_named_top_k(x: Any) -> Any:
    return _named_top_k(x)


def _jit_top_k(x: Any) -> Any:
    return jax.jit(_call_named_top_k)(x)


def _if_top_k(x: Any, other: Any, predicate: Any) -> Any:
    return jax.lax.cond(
        predicate,
        lambda pair: _call_named_top_k(pair[0]),
        lambda pair: _call_named_top_k(pair[1]),
        (x, other),
    )


def _internal_layout(
    selected: Callable[..., Any], symbol_axis: int
) -> Callable[..., Any]:
    if symbol_axis == 0:
        return selected
    if selected is _if_top_k:

        def branch(data: Any) -> Any:
            internal = jax.lax.transpose(data, (1, 0, 2))
            returned = _call_named_top_k(internal)
            return tuple(jax.lax.transpose(value, (1, 0, 2)) for value in returned)

        def conditional(x: Any, other: Any, predicate: Any) -> Any:
            return jax.lax.cond(
                predicate,
                lambda pair: branch(pair[0]),
                lambda pair: branch(pair[1]),
                (x, other),
            )

        return conditional

    def call(*inputs: Any) -> Any:
        data = jax.lax.transpose(inputs[0], (1, 0, 2))
        if len(inputs) == 1:
            returned = selected(data)
        else:
            other = jax.lax.transpose(inputs[1], (1, 0, 2))
            returned = selected(data, other, inputs[2])
        return tuple(jax.lax.transpose(value, (1, 0, 2)) for value in returned)

    return call


def _require_internal_native_axis(trace: Any, symbol_axis: int) -> None:
    body = trace.jaxpr
    equations = [
        equation
        for equation in body.eqns
        if equation.primitive is native_top_k_primitive()
    ]
    assert len(equations) == 1
    equation = equations[0]
    assert equation.params["axis"] == 2 and equation.params["k"] == 2
    assert [
        axis
        for axis, dimension in enumerate(equation.invars[0].aval.shape)
        if not isinstance(dimension, (int, np.integer))
    ] == [symbol_axis]
    assert equation.invars[0].aval.dtype == equation.outvars[0].aval.dtype
    assert equation.outvars[1].aval.dtype == np.dtype(np.int32)


def _word_input(dtype: str) -> np.ndarray:
    # Predeclared inputs, not native/runtime-derived output observations.
    if dtype == "float32":
        words = [
            [0x80000000, 0, 0x80000000, 0],
            [0x7F800000, 0x7FC00001, 0x7FC00002, 0x7F800001],
            [0xFF800000, 0xFFC00001, 0xFFC00002, 0xFF800001],
            [0xFFC00002, 0xFFC00001, 0xFFC00002, 0xFFC00003],
            [1, 0x80000001, 0x3F800000, 0x3F800000],
            [0x7F7FFFFF, 0x80000000, 0x3F000000, 0xFF7FFFFF],
        ]
        unsigned = np.uint32
    else:
        assert dtype == "float64"
        words = [
            [0x8000000000000000, 0, 0x8000000000000000, 0],
            [
                0x7FF0000000000000,
                0x7FF8000000000001,
                0x7FF8000000000002,
                0x7FF0000000000001,
            ],
            [
                0xFFF0000000000000,
                0xFFF8000000000001,
                0xFFF8000000000002,
                0xFFF0000000000001,
            ],
            [
                0xFFF8000000000002,
                0xFFF8000000000001,
                0xFFF8000000000002,
                0xFFF8000000000003,
            ],
            [1, 0x8000000000000001, 0x3FF0000000000000, 0x3FF0000000000000],
            [
                0x7FEFFFFFFFFFFFFF,
                0x8000000000000000,
                0x3FE0000000000000,
                0xFFEFFFFFFFFFFFFF,
            ],
        ]
        unsigned = np.uint64
    return np.asarray(words, dtype=unsigned).view(np.dtype(dtype))


def _expected(
    data: np.ndarray, k: int, axis: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    # Independent host oracle: sort original integer words, never float values.
    moved = np.moveaxis(data, axis, -1)
    width = data.dtype.itemsize * 8
    unsigned = np.uint32 if width == 32 else np.uint64
    words = moved.view(unsigned)
    indices = np.empty((*moved.shape[:-1], k), dtype=np.int32)
    sign = 1 << (width - 1)

    def order(word: int) -> int:
        return ((1 << width) - 1 - word) if word & sign else word + sign

    for position in np.ndindex(moved.shape[:-1]):
        ranked = sorted(
            range(moved.shape[-1]),
            key=lambda index: (-order(int(words[position][index])), index),
        )
        indices[position] = ranked[:k]
    values = np.take_along_axis(moved, indices, axis=-1)
    return np.moveaxis(values, -1, axis), np.moveaxis(indices, -1, axis)


def _retain(returned: Any, path: Path, label: str) -> tuple[np.ndarray, ...]:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the actual return event before arity, dtype, shape or value checks.
    event = path.with_suffix(".json")
    event.write_text(json.dumps(dict(status="returned", stage=label)))
    leaves = jax.tree.leaves(jax.device_get(returned))
    np.savez(path, **{f"leaf_{i}": leaf for i, leaf in enumerate(leaves)})
    event.write_text(
        json.dumps(dict(status="archived", stage=label, leaves=len(leaves)))
    )
    return tuple(np.asarray(leaf) for leaf in leaves)


def _same(actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...]) -> None:
    assert len(actual) == len(expected) == 2
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == reference.dtype and value.shape == reference.shape
        assert value.tobytes(order="C") == reference.tobytes(order="C")
    assert actual[1].dtype == np.int32


def _native(
    fn: Callable[..., Any],
    inputs: tuple[Any, ...],
    expected: tuple[np.ndarray, ...],
    directory: Path,
) -> tuple[tuple[np.ndarray, ...], ...]:
    compiled = jax.jit(_fresh(fn))
    references = []
    for label, operation in (("fresh", compiled), ("warmed", compiled), ("eager", fn)):
        try:
            returned = operation(*inputs)
        except BaseException as exc:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{label}-exception.json").write_text(
                json.dumps(
                    dict(status="raised", type=type(exc).__qualname__, message=str(exc))
                )
            )
            raise
        actual = _retain(returned, directory / f"{label}.npz", label)
        _same(actual, expected)
        references.append(actual)
    return tuple(references)


def _bitcast_schema() -> bool:
    try:
        schema = onnx.defs.get_schema("BitCast", 26, "")
    except onnx.defs.SchemaError:
        return False
    assert schema.since_version == 26
    return True


def _export(fn: Any, specs: list[Any], directory: Path, opset: int) -> Any:
    if opset >= 26 and not _bitcast_schema():
        with pytest.raises(onnx.checker.ValidationError) as caught:
            ui._to_onnx(fn, specs, precision_policy=PrecisionPolicy.SOURCE, opset=opset)
        assert "No Op registered for BitCast" in str(caught.value)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "representation-unavailable.json").write_text(
            json.dumps(
                dict(
                    status="REPRESENTATION_UNAVAILABLE",
                    requested_opset=opset,
                    reason=str(caught.value),
                    runtime="UNVERIFIED",
                    numerics="UNVERIFIED",
                )
            )
        )
        return None
    model = ui._to_onnx(fn, specs, precision_policy=PrecisionPolicy.SOURCE, opset=opset)
    onnx.checker.check_model(model, full_check=True)
    assert next(i.version for i in model.opset_import if not i.domain) == opset
    return model


def _cpu(model: Any, samples: list[Any], directory: Path) -> None:
    blob = model.SerializeToString()
    digest = hashlib.sha256(blob).hexdigest()
    for label, profile in (
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
            for index, (inputs, expected, references) in enumerate(samples):
                phase = f"run-{index}"
                feed = {
                    port.name: value
                    for port, value in zip(session.get_inputs(), inputs, strict=True)
                }
                returned = session.run(None, feed)
                actual = _retain(
                    returned, directory / f"cpu-{label}-{index}.npz", label
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
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"cpu-{label}-exception.json").write_text(
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


def _nodes(owner: Any) -> Any:
    for node in owner.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _nodes(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph in attribute.graphs:
                    yield from _nodes(graph)


@pytest.mark.parametrize(
    "dtype,x64", [("float32", False), ("float32", True), ("float64", True)]
)
@pytest.mark.parametrize("scope", ("inline", "named", "jit", "if-false", "if-true"))
@pytest.mark.parametrize("symbol_axis", (0, 1))
def test_native_top_k_totalorder_words_indices_and_real_scopes(
    dtype: str, x64: bool, scope: str, symbol_axis: int, tmp_path: Path
) -> None:
    with _runtime_configuration(x64):
        batch = jax_export.symbolic_shape("B")[0]
        shape = (batch, 6, 4)
        spec = jax.ShapeDtypeStruct(shape, np.dtype(dtype))
        fn = {"inline": _top_k, "named": _call_named_top_k, "jit": _jit_top_k}.get(
            scope, _if_top_k
        )
        fn = _internal_layout(fn, symbol_axis)
        inline_trace = _internal_layout(_top_k, symbol_axis)
        specs = (
            [spec, spec, jax.ShapeDtypeStruct((), np.bool_)]
            if scope.startswith("if-")
            else [spec]
        )
        native = jax.make_jaxpr(_fresh(inline_trace))(spec)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(_fresh(inline_trace))(spec)
        assert typed_trace_manifest(native) == typed_trace_manifest(activated)
        _require_internal_native_axis(native, symbol_axis)
        _require_internal_native_axis(activated, symbol_axis)
        assert _contexts(native) == _contexts(activated)
        assert [_payload(v) for v in native.consts] == [
            _payload(v) for v in activated.consts
        ]
        validate_source_jaxpr(native, 26)
        samples = []
        for size in (2, 5):
            data = np.repeat(np.expand_dims(_word_input(dtype), 0), size, axis=0)
            other = np.flip(data, axis=-1).copy()
            if scope.startswith("if-"):
                predicate = np.asarray(scope == "if-true", dtype=np.bool_)
                inputs = (data, other, predicate)
                expected = _expected(data if bool(predicate) else other, 2)
            else:
                inputs, expected = (data,), _expected(data, 2)
            input_bytes = tuple(value.tobytes(order="C") for value in inputs)
            references = _native(fn, inputs, expected, tmp_path / f"B{size}")
            assert tuple(value.tobytes(order="C") for value in inputs) == input_bytes
            samples.append((inputs, expected, references))
        model = _export(fn, specs, tmp_path, 26)
        if model is None:
            return
        nodes = [n for owner in (model.graph, *model.functions) for n in _nodes(owner)]
        assert any(n.op_type == "BitCast" for n in nodes)
        assert any(n.op_type == "BitwiseXor" for n in nodes)
        assert any(n.op_type == "GatherElements" for n in nodes)
        assert all(n.attribute for n in nodes if n.op_type == "TopK")
        if scope != "inline":
            functions = [f for f in model.functions if "NativeTotalOrderTopK" in f.name]
            assert len(functions) == 1
            function = functions[0]
            assert any(
                n.domain == function.domain and n.op_type == function.name
                for n in nodes
            )
            assert any(n.op_type == "BitCast" for n in function.node)
            infos = {value.name: value for value in function.value_info}
            input_shape = ("B", 6, 4) if symbol_axis == 0 else (6, "B", 4)
            output_shape = (*input_shape[:-1], 2)
            for name, shape in zip(
                (*function.input, *function.output),
                (input_shape, output_shape, output_shape),
                strict=True,
            ):
                tensor = infos[name].type.tensor_type
                assert (
                    tuple(
                        dimension.dim_param or dimension.dim_value
                        for dimension in tensor.shape.dim
                    )
                    == shape
                )
            assert (
                infos[function.output[1]].type.tensor_type.elem_type
                == onnx.TensorProto.INT32
            )
        _cpu(model, samples, tmp_path)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("extent,k", ((0, 0), (4, 0), (4, 1), (4, 4)))
def test_native_top_k_boundary_k_nonlast_axis_and_typed_empty(
    dtype: str, extent: int, k: int, tmp_path: Path
) -> None:
    with _runtime_configuration(True):

        def original(x: Any) -> Any:
            return jax.lax.top_k(x, k, axis=0)

        data = _word_input(dtype).T[:extent].copy()
        expected = _expected(data, k, axis=0)
        references = _native(original, (data,), expected, tmp_path)
        opset = 23 if k == 0 else 26
        model = _export(
            original, [jax.ShapeDtypeStruct(data.shape, data.dtype)], tmp_path, opset
        )
        if model is None:
            return
        nodes = list(_nodes(model.graph))
        if k == 0:
            assert not any(n.op_type in {"TopK", "BitCast"} for n in nodes)
            assert any(n.op_type == "Slice" for n in nodes)
        _cpu(model, [((data,), expected, references)], tmp_path)


@pytest.mark.parametrize(
    "dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64")
)
def test_native_integer_top_k_lossless_int64_keys_on_original_opset23(
    dtype: str, tmp_path: Path
) -> None:
    with _runtime_configuration(True):
        info = np.iinfo(dtype)
        data = np.asarray([[info.min, info.max, info.max, 0]], dtype=dtype)
        indices = np.asarray([[1, 2]], dtype=np.int32)
        expected = np.take_along_axis(data, indices, axis=-1), indices
        references = _native(_top_k, (data,), expected, tmp_path)
        model = _export(
            _top_k, [jax.ShapeDtypeStruct(data.shape, data.dtype)], tmp_path, 23
        )
        assert not any(n.op_type == "BitCast" for n in model.graph.node)
        _cpu(model, [((data,), expected, references)], tmp_path)


def _equation() -> Any:
    batch = jax_export.symbolic_shape("B")[0]
    trace = jax.make_jaxpr(_top_k)(jax.ShapeDtypeStruct((6, batch, 4), np.float32))
    return trace, next(e for e in trace.eqns if e.primitive is native_top_k_primitive())


def _validate(equation: Any, formals: set[Any], opset: int = 26) -> None:
    site = _Site("native TopK test", opset, equation.primitive.name)
    validate_native_top_k(
        equation,
        [_tensor(v, site) for v in equation.invars],
        [_tensor(v, site) for v in equation.outvars],
        formals,
        site,
    )


def test_native_top_k_original23_is_actionable_and_does_not_change_legacy() -> None:
    with _runtime_configuration(False):
        _, equation = _equation()
        site = _Site("native TopK test", 23, "top_k")
        source = _tensor(equation.invars[0], site)
        assert source.symbol is not None
        with pytest.raises(SourcePrecisionError) as caught:
            _validate(equation, {source.symbol}, 23)
        error = caught.value
        assert error.category == "representation-limit" and error.opset == 23
        assert error.primitive == "top_k" and "opset=26" in error.action
        assert "runtime" in error.action and "BitCast" in error.requirement + str(error)
        # A prior cached named26 body cannot upgrade a later explicit23 request.
        specs = [jax.ShapeDtypeStruct((3, 4), np.float32)]
        if _bitcast_schema():
            for _ in range(2):
                exported = ui._to_onnx(
                    _call_named_top_k,
                    specs,
                    precision_policy=PrecisionPolicy.SOURCE,
                    opset=26,
                )
                assert (
                    next(i.version for i in exported.opset_import if not i.domain) == 26
                )
                with pytest.raises(SourcePrecisionError) as cached_error:
                    ui._to_onnx(
                        _call_named_top_k,
                        specs,
                        precision_policy=PrecisionPolicy.SOURCE,
                        opset=23,
                    )
                assert cached_error.value.category == "representation-limit"
        legacy = ui.to_onnx(_top_k, [jax.ShapeDtypeStruct((3, 4), np.float32)])
        assert any(n.op_type == "TopK" for n in legacy.graph.node)
        assert not any(n.op_type == "BitCast" for n in legacy.graph.node)


def test_native_top_k_schema_identity_scope_shape_type_weakness_and_index_bounds() -> (
    None
):
    with _runtime_configuration(True):
        _, equation = _equation()
        site = _Site("native TopK test", 26, "top_k")
        source = _tensor(equation.invars[0], site)
        assert source.symbol is not None
        _validate(equation, {source.symbol})
        value, index = equation.outvars
        batch = equation.invars[0].aval.shape[1]
        foreign = jax_export.symbolic_shape("B")[0]
        corruptions = {
            "foreign placement": equation.replace(
                invars=[
                    SimpleNamespace(
                        aval=SimpleNamespace(
                            shape=equation.invars[0].aval.shape,
                            dtype=equation.invars[0].aval.dtype,
                            weak_type=equation.invars[0].aval.weak_type,
                            sharding=object(),
                        )
                    )
                ]
            ),
            "foreign identity": equation.replace(primitive=Primitive("top_k")),
            "second input": equation.replace(
                invars=[*equation.invars, *equation.invars]
            ),
            "missing output": equation.replace(outvars=[value]),
            "effects": equation.replace(effects={object()}),
            "extra parameter": equation.replace(
                params={**equation.params, "placement": None}
            ),
            "missing parameter": equation.replace(
                params={k: v for k, v in equation.params.items() if k != "axis"}
            ),
            "boolean k": equation.replace(params={**equation.params, "k": True}),
            "negative k": equation.replace(params={**equation.params, "k": -1}),
            "large k": equation.replace(params={**equation.params, "k": 5}),
            "noncanonical axis": equation.replace(
                params={**equation.params, "axis": -1}
            ),
            "symbolic axis": equation.replace(params={**equation.params, "axis": 1}),
            "wrong values": equation.replace(
                outvars=[Var(value.aval.update(dtype=np.float64)), index]
            ),
            "wrong indices": equation.replace(
                outvars=[value, Var(index.aval.update(dtype=np.int64))]
            ),
            "weak indices": equation.replace(
                outvars=[value, Var(index.aval.update(weak_type=True))]
            ),
            "wrong shape": equation.replace(
                outvars=[Var(value.aval.update(shape=(6, batch, 1))), index]
            ),
            "foreign scope": equation.replace(
                outvars=[Var(value.aval.update(shape=(6, foreign, 2))), index]
            ),
            "wrong axis": equation.replace(
                outvars=[Var(value.aval.update(shape=(batch, 6, 2))), index]
            ),
            "overflow extent": equation.replace(
                invars=[
                    Var(equation.invars[0].aval.update(shape=(6, batch, (1 << 31) + 1)))
                ]
            ),
        }
        if "is_stable" in native_top_k_parameter_names():
            # Native unstable ties permit the same deterministic lower-index
            # choice. Its real Boolean parameter remains part of the equation.
            _validate(
                equation.replace(params={**equation.params, "is_stable": False}),
                {source.symbol},
            )
            corruptions["nonboolean stability"] = equation.replace(
                params={**equation.params, "is_stable": 1}
            )
        for name, corrupted in corruptions.items():
            with pytest.raises(
                (SourcePrecisionError, TypeError),
                match="TopK|primitive|result|shape|symbolic|batch",
            ) as caught:
                _validate(corrupted, {source.symbol})
            assert caught.value is not None, name
        with pytest.raises(SourcePrecisionError, match="batch scope"):
            _validate(equation, set())


def test_native_top_k_defining_aliases_and_parameter_owner_restore_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    defining = importlib.import_module("jax._src.lax.lax")
    original = native_top_k_primitive()
    parameters = native_top_k_parameter_names()
    function = defining._top_k_abstract_eval

    def foreign(operand: Any, *, k: Any, axis: Any) -> Any:
        return operand

    for module in (jax.lax, defining):
        with monkeypatch.context() as patch:
            patch.setattr(module, "top_k_p", Primitive("top_k"))
            with pytest.raises(TypeError, match="TopK primitive bindings"):
                native_top_k_primitive()
        assert native_top_k_primitive() is original
    with monkeypatch.context() as patch:
        patch.setattr(defining, "_top_k_abstract_eval", foreign)
        with pytest.raises(TypeError, match="TopK parameter schema"):
            native_top_k_parameter_names()
    assert native_top_k_parameter_names() == parameters
    for field, value in (
        ("__code__", foreign.__code__),
        ("__defaults__", (1,)),
        ("__kwdefaults__", {"k": 1}),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(function, field, value)
            with pytest.raises(TypeError, match="TopK parameter schema"):
                native_top_k_parameter_names()
        assert native_top_k_parameter_names() == parameters


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "uint64"))
def test_native_top_k_other_word_widths_have_explicit_admission(
    dtype: str,
) -> None:
    with _runtime_configuration(True):
        trace = jax.make_jaxpr(_top_k)(jax.ShapeDtypeStruct((3, 4), np.dtype(dtype)))
        validate_source_jaxpr(trace, 26)
        assert trace.outvars[0].aval.dtype == np.dtype(dtype)
        assert trace.outvars[1].aval.dtype == np.dtype(np.int32)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_top_k_word_oracle_detects_zero_index_and_nan_payload_corruption(
    dtype: str,
) -> None:
    data = _word_input(dtype)
    expected = _expected(data, 2)
    np.testing.assert_array_equal(expected[1][0], np.asarray([1, 3], np.int32))
    wrong_indices = expected[1].copy()
    wrong_indices[0] = [0, 1]
    wrong_values = np.take_along_axis(data, wrong_indices, axis=-1)
    with pytest.raises(AssertionError):
        _same((wrong_values, wrong_indices), expected)
    words = np.uint32 if dtype == "float32" else np.uint64
    changed_payload = expected[0].copy()
    changed_payload.view(words)[1, 0] ^= 1
    assert np.isnan(changed_payload[1, 0]) and np.isnan(expected[0][1, 0])
    with pytest.raises(AssertionError):
        _same((changed_payload, expected[1]), expected)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_native_top_k_symbolic_typed_empty_keeps_actual_batch_origin(
    dtype: str, tmp_path: Path
) -> None:
    with _runtime_configuration(True):

        def original(x: Any) -> Any:
            return jax.lax.top_k(x, 0)

        batch = jax_export.symbolic_shape("B")[0]
        specs = [jax.ShapeDtypeStruct((batch, 6, 4), np.dtype(dtype))]
        samples = []
        for size in (2, 5):
            data = np.repeat(_word_input(dtype)[None, ...], size, axis=0)
            expected = _expected(data, 0)
            references = _native(original, (data,), expected, tmp_path / f"B{size}")
            samples.append(((data,), expected, references))
        model = _export(original, specs, tmp_path, 23)
        assert not any(n.op_type in {"TopK", "BitCast"} for n in model.graph.node)
        _cpu(model, samples, tmp_path)


def _portable_word_input(dtype: str) -> np.ndarray:
    if dtype == "float16":
        words = [
            [0x8000, 0, 0x8000, 0],
            [0x7C00, 0x7E01, 0x7E02, 0x7C01],
            [0xFC00, 0xFE01, 0xFE02, 0xFC01],
            [0xFE02, 0xFE01, 0xFE02, 0xFE03],
            [1, 0x8001, 0x3C00, 0x3C00],
            [0x7BFF, 0x8000, 0x3800, 0xFBFF],
        ]
        return np.asarray(words, dtype=np.uint16).view(np.float16)
    assert dtype == "uint64"
    return np.asarray(
        [
            [0, 0, 1, 1],
            [(1 << 63) - 1, 1 << 63, (1 << 63) + 1, (1 << 64) - 1],
            [(1 << 64) - 2, (1 << 64) - 1, (1 << 64) - 2, (1 << 64) - 3],
            [(1 << 53) + 1, (1 << 53) + 2, (1 << 53) + 3, (1 << 53) + 1],
            [(1 << 63) + 17, (1 << 63) + 18, (1 << 63) + 19, (1 << 63) + 17],
            [1, 2, 3, 4],
        ],
        dtype=np.uint64,
    )


def _portable_words(data: np.ndarray) -> np.ndarray:
    return data.view(np.dtype(f"u{data.dtype.itemsize}"))


def _portable_key(word: int, dtype: np.dtype[Any]) -> int:
    if dtype.kind == "u":
        return word
    assert dtype.kind == "f" or (dtype.name == "bfloat16" and dtype.itemsize == 2)
    width = dtype.itemsize * 8
    return (
        (1 << width) - 1 - word
        if word & (1 << (width - 1))
        else word + (1 << (width - 1))
    )


def _portable_expected(
    data: np.ndarray, k: int, axis: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    moved = np.moveaxis(data, axis, -1)
    words = _portable_words(moved)
    indices = np.empty((*moved.shape[:-1], k), dtype=np.int32)
    for position in np.ndindex(moved.shape[:-1]):
        selected = sorted(
            range(moved.shape[-1]),
            key=lambda index: (
                -_portable_key(int(words[position][index]), data.dtype),
                index,
            ),
        )
        indices[position] = selected[:k]
    values = np.take_along_axis(moved, indices, axis=-1)
    return np.moveaxis(values, -1, axis), np.moveaxis(indices, -1, axis)


def _portable_relation(
    actual: tuple[np.ndarray, ...], data: np.ndarray, k: int, axis: int, stable: bool
) -> None:
    assert len(actual) == 2
    expected = _portable_expected(data, k, axis)
    assert actual[0].dtype == data.dtype and actual[1].dtype == np.dtype(np.int32)
    assert all(
        value.shape == reference.shape
        for value, reference in zip(actual, expected, strict=True)
    )
    moved = np.moveaxis(data, axis, -1)
    words = _portable_words(moved)
    values = _portable_words(np.moveaxis(actual[0], axis, -1))
    indices = np.moveaxis(actual[1], axis, -1)
    ordered_indices = np.moveaxis(expected[1], axis, -1)
    for position in np.ndindex(moved.shape[:-1]):
        row = [int(word) for word in words[position]]
        keys = [_portable_key(word, data.dtype) for word in row]
        chosen = [int(index) for index in indices[position]]
        assert len(chosen) == len(set(chosen)) == k
        assert all(0 <= index < len(row) for index in chosen)
        selected_keys = [keys[index] for index in chosen]
        assert selected_keys == sorted(keys, reverse=True)[:k]
        assert [int(word) for word in values[position]] == [
            row[index] for index in chosen
        ]
        for index, (selected, key) in enumerate(
            zip(chosen, selected_keys, strict=True)
        ):
            if stable or keys.count(key) == 1:
                assert selected == int(ordered_indices[position][index])


def _portable_native(
    fn: Any,
    inputs: tuple[Any, ...],
    data: np.ndarray,
    k: int,
    axis: int,
    stable: bool,
    directory: Path,
) -> tuple[tuple[np.ndarray, ...], ...]:
    compiled = jax.jit(_fresh(fn))
    references = []
    for label, operation in (("fresh", compiled), ("warmed", compiled), ("eager", fn)):
        try:
            returned = operation(*inputs)
        except BaseException as cause:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{label}-exception.json").write_text(
                json.dumps(
                    dict(
                        status="raised",
                        type=type(cause).__qualname__,
                        message=str(cause),
                    )
                )
            )
            raise
        actual = _retain(returned, directory / f"{label}.npz", label)
        _portable_relation(actual, data, k, axis, stable)
        references.append(actual)
    return tuple(references)


def _portable_cpu(
    model: Any, samples: list[Any], k: int, axis: int, stable: bool, directory: Path
) -> None:
    blob = model.SerializeToString()
    digest = hashlib.sha256(blob).hexdigest()
    for label, profile in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = profile
        session = ort.InferenceSession(
            blob, options, providers=["CPUExecutionProvider"]
        )
        session.disable_fallback()
        for index, (inputs, data, expected, references) in enumerate(samples):
            feed = {
                port.name: value
                for port, value in zip(session.get_inputs(), inputs, strict=True)
            }
            returned = session.run(None, feed)
            actual = _retain(returned, directory / f"cpu-{label}-{index}.npz", label)
            _portable_relation(actual, data, k, axis, stable)
            # ONNX specifies deterministic lower-index ties, including for a
            # native unstable request. Native ties may choose another order.
            _same(actual, expected)
            assert len(references) == 3
            for reference in references:
                _portable_relation(reference, data, k, axis, stable)
                if stable:
                    _same(actual, reference)
            assert session.get_providers() == ["CPUExecutionProvider"]
        assert session.get_session_options().graph_optimization_level == profile
        assert hashlib.sha256(model.SerializeToString()).hexdigest() == digest


def _portable_default_top_k(x: Any) -> Any:
    return jax.lax.top_k(x, 2)


@onnx_function(name="NativePortableWordTopKDefault")
def _portable_named_default(x: Any) -> Any:
    return _portable_default_top_k(x)


def _portable_jit_default(x: Any) -> Any:
    return jax.jit(_portable_named_default)(x)


def _portable_unstable_top_k(x: Any) -> Any:
    return jax.lax.top_k(x, 2, is_stable=False)


@onnx_function(name="NativePortableWordTopKUnstable")
def _portable_named_unstable(x: Any) -> Any:
    return _portable_unstable_top_k(x)


def _portable_jit_unstable(x: Any) -> Any:
    return jax.jit(_portable_named_unstable)(x)


def _portable_call_named_default(x: Any) -> Any:
    return _portable_named_default(x)


def _portable_call_named_unstable(x: Any) -> Any:
    return _portable_named_unstable(x)


def _portable_scope(
    stability: bool | None, scope: str, symbol_axis: int
) -> tuple[Any, Any]:
    if stability is None:
        original, named, jitted = (
            _portable_default_top_k,
            _portable_call_named_default,
            _portable_jit_default,
        )
    else:
        assert stability is False
        original, named, jitted = (
            _portable_unstable_top_k,
            _portable_call_named_unstable,
            _portable_jit_unstable,
        )

    def layout(selected: Any, x: Any) -> Any:
        if symbol_axis == 0:
            return selected(x)
        internal = jax.lax.transpose(x, (1, 0, 2))
        returned = selected(internal)
        return tuple(jax.lax.transpose(value, (1, 0, 2)) for value in returned)

    def inline(x: Any) -> Any:
        return layout(original, x)

    if scope.startswith("if-"):

        def conditional(x: Any, other: Any, predicate: Any) -> Any:
            return jax.lax.cond(
                predicate,
                lambda pair: layout(named, pair[0]),
                lambda pair: layout(named, pair[1]),
                (x, other),
            )

        return conditional, inline
    selected = {"inline": original, "named": named, "jit": jitted}[scope]

    def call(x: Any) -> Any:
        return layout(selected, x)

    return call, inline


def _portable_artifact(model: Any, dtype: str) -> None:
    def owners(owner: Any) -> Any:
        yield owner
        for node in owner.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    yield from owners(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph in attribute.graphs:
                        yield from owners(graph)

    count = 0
    for top in (model.graph, *model.functions):
        for owner in owners(top):
            produced = {name: node for node in owner.node for name in node.output}
            infos = {value.name: value for value in owner.value_info}
            initializers = {}
            if isinstance(owner, onnx.GraphProto):
                infos.update(
                    {value.name: value for value in (*owner.input, *owner.output)}
                )
                initializers = {value.name: value for value in owner.initializer}

            def literal(name: str, expected: int) -> None:
                if name in initializers:
                    tensor = initializers[name]
                else:
                    node = produced[name]
                    assert node.op_type == "Constant"
                    tensor = next(
                        attribute.t
                        for attribute in node.attribute
                        if attribute.name == "value"
                    )
                value = onnx.numpy_helper.to_array(tensor)
                assert value.shape == () and value.dtype == np.dtype(np.int64)
                assert int(value) == expected

            for node in owner.node:
                if node.op_type != "TopK":
                    continue
                count += 1
                assert {
                    attribute.name: attribute.i for attribute in node.attribute
                } == {"axis": 2, "largest": 1, "sorted": 1}
                keys = produced[node.input[0]]
                assert (
                    infos[node.input[0]].type.tensor_type.elem_type
                    == onnx.TensorProto.INT64
                )
                if dtype in ("float16", "bfloat16"):
                    assert keys.op_type == "Where"
                    negative, reverse, words = (produced[name] for name in keys.input)
                    assert (
                        negative.op_type == "Less"
                        and reverse.op_type == "BitwiseXor"
                        and words.op_type == "Cast"
                    )
                    assert negative.input[0] == reverse.input[0] == words.output[0]
                    literal(negative.input[1], 0)
                    literal(reverse.input[1], (1 << 15) - 1)
                    assert (
                        next(
                            attribute.i
                            for attribute in words.attribute
                            if attribute.name == "to"
                        )
                        == onnx.TensorProto.INT64
                    )
                    bitcast = produced[words.input[0]]
                    word_type = onnx.TensorProto.INT16
                else:
                    assert dtype == "uint64" and keys.op_type == "BitwiseXor"
                    bitcast = produced[keys.input[0]]
                    literal(keys.input[1], -(1 << 63))
                    word_type = onnx.TensorProto.INT64
                assert bitcast.op_type == "BitCast"
                assert (
                    next(
                        attribute.i
                        for attribute in bitcast.attribute
                        if attribute.name == "to"
                    )
                    == word_type
                )
                gathered = [
                    selected
                    for selected in owner.node
                    if selected.op_type == "GatherElements"
                    and list(selected.input) == [bitcast.input[0], node.output[1]]
                ]
                assert len(gathered) == 1
                assert (
                    infos[gathered[0].output[0]].type.tensor_type.elem_type
                    == {
                        "float16": onnx.TensorProto.FLOAT16,
                        "bfloat16": onnx.TensorProto.BFLOAT16,
                        "uint64": onnx.TensorProto.UINT64,
                    }[dtype]
                )
                index_casts = [
                    selected
                    for selected in owner.node
                    if selected.op_type == "Cast"
                    and list(selected.input) == [node.output[1]]
                ]
                assert len(index_casts) == 1
                assert (
                    next(
                        attribute.i
                        for attribute in index_casts[0].attribute
                        if attribute.name == "to"
                    )
                    == onnx.TensorProto.INT32
                )
    assert count >= 1


@pytest.mark.parametrize("dtype", ("float16", "uint64"))
@pytest.mark.parametrize("scope", ("inline", "named", "jit", "if-false", "if-true"))
def test_native_portable_top_k_words_scopes_and_available_stabilities(
    dtype: str, scope: str, tmp_path: Path
) -> None:
    x64 = dtype == "uint64" or scope not in {"inline", "jit"}
    symbol_axis = 0 if scope == "inline" else 1
    with _runtime_configuration(x64):
        available = "is_stable" in native_top_k_parameter_names()
        stabilities = (
            (None, False) if available and scope in {"inline", "named"} else (None,)
        )
        for stability in stabilities:
            route = "native-default" if stability is None else "native-unstable"
            directory = tmp_path / route
            batch = jax_export.symbolic_shape("B")[0]
            spec = jax.ShapeDtypeStruct((batch, 6, 4), np.dtype(dtype))
            fn, inline = _portable_scope(stability, scope, symbol_axis)
            native = jax.make_jaxpr(_fresh(inline))(spec)
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                activated = jax.make_jaxpr(_fresh(inline))(spec)
            assert typed_trace_manifest(native) == typed_trace_manifest(activated)
            assert _contexts(native) == _contexts(activated)
            _require_internal_native_axis(native, symbol_axis)
            _require_internal_native_axis(activated, symbol_axis)
            equation = next(
                e for e in native.eqns if e.primitive is native_top_k_primitive()
            )
            assert set(equation.params) == native_top_k_parameter_names()
            if available:
                assert equation.params["is_stable"] is (
                    True if stability is None else stability
                )
            else:
                assert "is_stable" not in equation.params and stability is None
            validate_source_jaxpr(native, 26)
            samples = []
            for size in (2, 5):
                data = np.repeat(_portable_word_input(dtype)[None, ...], size, axis=0)
                other = np.flip(data, axis=-1).copy()
                if scope.startswith("if-"):
                    predicate = np.asarray(scope == "if-true", dtype=np.bool_)
                    inputs = (data, other, predicate)
                    selected = data if bool(predicate) else other
                else:
                    inputs, selected = (data,), data
                original_words = tuple(value.tobytes() for value in inputs)
                expected = _portable_expected(selected, 2)
                references = _portable_native(
                    fn,
                    inputs,
                    selected,
                    2,
                    -1,
                    stability is not False,
                    directory / f"B{size}",
                )
                assert tuple(value.tobytes() for value in inputs) == original_words
                samples.append((inputs, selected, expected, references))
            specs = (
                [spec, spec, jax.ShapeDtypeStruct((), np.bool_)]
                if scope.startswith("if-")
                else [spec]
            )
            model = _export(fn, specs, directory, 26)
            if model is None:
                continue
            _portable_artifact(model, dtype)
            if scope != "inline":
                functions = [
                    function
                    for function in model.functions
                    if "NativePortableWordTopK" in function.name
                ]
                assert len(functions) == 1
                function = functions[0]
                nodes = [
                    node
                    for owner in (model.graph, *model.functions)
                    for node in _nodes(owner)
                ]
                assert any(
                    node.domain == function.domain and node.op_type == function.name
                    for node in nodes
                )
                infos = {value.name: value for value in function.value_info}
                for name, shape in zip(
                    (*function.input, *function.output),
                    ((6, "B", 4), (6, "B", 2), (6, "B", 2)),
                    strict=True,
                ):
                    assert (
                        tuple(
                            dim.dim_param or dim.dim_value
                            for dim in infos[name].type.tensor_type.shape.dim
                        )
                        == shape
                    )
                assert (
                    infos[function.output[1]].type.tensor_type.elem_type
                    == onnx.TensorProto.INT32
                )
            _portable_cpu(model, samples, 2, -1, stability is not False, directory)


@pytest.mark.parametrize("dtype", ("float16", "uint64"))
@pytest.mark.parametrize("extent,k", ((0, 0), (4, 0), (4, 1), (4, 4)))
def test_native_portable_top_k_nonlast_axis_and_empty(
    dtype: str, extent: int, k: int, tmp_path: Path
) -> None:
    with _runtime_configuration(dtype == "uint64"):
        stability = False if "is_stable" in native_top_k_parameter_names() else None

        def original(x: Any) -> Any:
            keywords = {} if stability is None else {"is_stable": stability}
            return jax.lax.top_k(x, k, axis=0, **keywords)

        data = _portable_word_input(dtype).T[:extent].copy()
        expected = _portable_expected(data, k, 0)
        references = _portable_native(
            original, (data,), data, k, 0, stability is not False, tmp_path
        )
        trace = jax.make_jaxpr(original)(jax.ShapeDtypeStruct(data.shape, data.dtype))
        validate_source_jaxpr(trace, 23 if k == 0 else 26)
        model = _export(
            original,
            [jax.ShapeDtypeStruct(data.shape, data.dtype)],
            tmp_path,
            23 if k == 0 else 26,
        )
        if model is None:
            return
        if k == 0:
            assert not any(
                node.op_type in {"BitCast", "TopK"} for node in model.graph.node
            )
            assert any(node.op_type == "Slice" for node in model.graph.node)
        _portable_cpu(
            model,
            [((data,), data, expected, references)],
            k,
            0,
            stability is not False,
            tmp_path,
        )


@pytest.mark.parametrize("dtype", ("float16", "uint64"))
def test_native_portable_top_k_nonempty23_requires_explicit26(dtype: str) -> None:
    with _runtime_configuration(dtype == "uint64"):
        trace = jax.make_jaxpr(_top_k)(jax.ShapeDtypeStruct((3, 4), np.dtype(dtype)))
        with pytest.raises(SourcePrecisionError) as caught:
            validate_source_jaxpr(trace, 23)
        assert (
            caught.value.category == "representation-limit" and caught.value.opset == 23
        )
        assert "opset=26" in caught.value.action and "runtime" in caught.value.action
        validate_source_jaxpr(trace, 26)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_native_top_k_actual_stability_schema_has_portable_tie_contract(
    dtype: str, tmp_path: Path
) -> None:
    with _runtime_configuration(True):
        stability = False if "is_stable" in native_top_k_parameter_names() else None

        def original(x: Any) -> Any:
            keywords = {} if stability is None else {"is_stable": stability}
            return jax.lax.top_k(x, 2, **keywords)

        data = _word_input(dtype)
        trace = jax.make_jaxpr(original)(jax.ShapeDtypeStruct(data.shape, data.dtype))
        equation = next(
            e for e in trace.eqns if e.primitive is native_top_k_primitive()
        )
        assert set(equation.params) == native_top_k_parameter_names()
        if stability is None:
            assert "is_stable" not in equation.params
        else:
            assert equation.params["is_stable"] is False
        validate_source_jaxpr(trace, 26)
        expected = _portable_expected(data, 2)
        references = _portable_native(
            original, (data,), data, 2, -1, stability is not False, tmp_path
        )
        model = _export(
            original, [jax.ShapeDtypeStruct(data.shape, data.dtype)], tmp_path, 26
        )
        if model is not None:
            _portable_cpu(
                model,
                [((data,), data, expected, references)],
                2,
                -1,
                stability is not False,
                tmp_path,
            )


@pytest.mark.parametrize("dtype", ("float16", "uint64"))
def test_portable_top_k_relational_oracle_keeps_words_bounds_and_legal_ties(
    dtype: str,
) -> None:
    data = _portable_word_input(dtype)
    expected = _portable_expected(data, 2)
    _portable_relation(expected, data, 2, -1, False)
    # Exactly equal keys may exchange their two indices for unstable native
    # selection; returned original words must still match each chosen index.
    indices = expected[1].copy()
    if dtype == "float16":
        row = 4
    else:
        row = 0
    assert _portable_key(
        int(_portable_words(data)[row, int(indices[row, 0])]), data.dtype
    ) == _portable_key(
        int(_portable_words(data)[row, int(indices[row, 1])]), data.dtype
    )
    indices[row] = indices[row, ::-1]
    values = np.take_along_axis(data, indices, axis=-1)
    _portable_relation((values, indices), data, 2, -1, False)
    with pytest.raises(AssertionError):
        _portable_relation((values, indices), data, 2, -1, True)
    broken = expected[0].copy()
    _portable_words(broken).flat[0] ^= 1
    with pytest.raises(AssertionError):
        _portable_relation((broken, expected[1]), data, 2, -1, False)
    duplicate = expected[1].copy()
    duplicate[0] = [0, 0]
    with pytest.raises(AssertionError):
        _portable_relation(
            (np.take_along_axis(data, duplicate, axis=-1), duplicate),
            data,
            2,
            -1,
            False,
        )


def _empty_selected_batch(x: Any) -> Any:
    return jax.lax.top_k(x, 0, axis=0)


@onnx_function(name="NativeEmptySelectedBatchTopK")
def _empty_named_batch(x: Any) -> Any:
    return _empty_selected_batch(x)


def _empty_call_named_batch(x: Any) -> Any:
    return _empty_named_batch(x)


def _empty_jit_batch(x: Any) -> Any:
    return jax.jit(_empty_named_batch)(x)


def _empty_if_batch(x: Any, other: Any, predicate: Any) -> Any:
    return jax.lax.cond(
        predicate,
        lambda pair: _empty_named_batch(pair[0]),
        lambda pair: _empty_named_batch(pair[1]),
        (x, other),
    )


def _empty_pair(shape: tuple[int, ...], dtype: Any) -> tuple[np.ndarray, np.ndarray]:
    assert 0 in shape
    return np.empty(shape, dtype=dtype), np.empty(shape, dtype=np.int32)


def _empty_shape(port: Any, expected: tuple[object, ...]) -> None:
    dimensions = port.type.tensor_type.shape.dim
    assert len(dimensions) == len(expected)
    for actual, value in zip(dimensions, expected, strict=True):
        if type(value) is int:
            assert actual.HasField("dim_value") and actual.dim_value == value
            assert not actual.HasField("dim_param")
        else:
            assert actual.HasField("dim_param") and actual.dim_param == value
            assert not actual.HasField("dim_value")


def _empty_artifact(model: Any, shape: tuple[object, ...]) -> list[Any]:
    assert next(item.version for item in model.opset_import if not item.domain) == 23
    nodes = [
        node for owner in (model.graph, *model.functions) for node in _nodes(owner)
    ]
    assert not any(node.op_type in {"TopK", "BitCast"} for node in nodes)
    assert any(node.op_type == "Slice" for node in nodes)
    for output in model.graph.output:
        _empty_shape(output, shape)
    assert model.graph.output[1].type.tensor_type.elem_type == onnx.TensorProto.INT32
    return nodes


@pytest.mark.parametrize(
    "dtype,scope",
    (
        ("float16", "inline"),
        ("float32", "named"),
        ("float64", "jit"),
        ("uint64", "if-false"),
        ("int32", "if-true"),
    ),
)
def test_native_top_k_selected_batch_k0_loses_only_selected_symbol(
    dtype: str, scope: str, tmp_path: Path
) -> None:
    with _runtime_configuration(dtype in {"float64", "uint64"}):
        batch = jax_export.symbolic_shape("B")[0]
        spec = jax.ShapeDtypeStruct((batch, 4), np.dtype(dtype))
        trace = jax.make_jaxpr(_empty_selected_batch)(spec)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(_fresh(_empty_selected_batch))(spec)
        assert typed_trace_manifest(trace) == typed_trace_manifest(activated)
        assert _contexts(trace) == _contexts(activated)
        validate_source_jaxpr(trace, 23)
        equation = next(
            item for item in trace.eqns if item.primitive is native_top_k_primitive()
        )
        source = _tensor(equation.invars[0], _Site("selected empty batch", 23, "top_k"))
        assert source.symbol is not None and source.symbol_axis == 0
        assert equation.params["axis"] == 0 and equation.params["k"] == 0
        for variable in equation.outvars:
            target = _tensor(variable, _Site("selected empty batch", 23, "top_k"))
            assert (
                target.shape == (0, 4)
                and target.symbol is None
                and target.symbol_axis is None
            )
        fn = {
            "inline": _empty_selected_batch,
            "named": _empty_call_named_batch,
            "jit": _empty_jit_batch,
        }.get(scope, _empty_if_batch)
        specs = (
            [spec, spec, jax.ShapeDtypeStruct((), np.bool_)]
            if scope.startswith("if-")
            else [spec]
        )
        samples = []
        for size in (2, 5):
            data = np.arange(size * 4, dtype=np.dtype(dtype)).reshape(size, 4)
            if scope.startswith("if-"):
                inputs = (
                    data,
                    np.flip(data, axis=-1).copy(),
                    np.asarray(scope == "if-true", dtype=np.bool_),
                )
            else:
                inputs = (data,)
            input_bytes = tuple(value.tobytes() for value in inputs)
            expected = _empty_pair((0, 4), np.dtype(dtype))
            references = _native(fn, inputs, expected, tmp_path / f"B{size}")
            assert tuple(value.tobytes() for value in inputs) == input_bytes
            samples.append((inputs, expected, references))
        model = _export(fn, specs, tmp_path, 23)
        nodes = _empty_artifact(model, (0, 4))
        if scope != "inline":
            functions = [
                function
                for function in model.functions
                if "NativeEmptySelectedBatchTopK" in function.name
            ]
            assert len(functions) == 1
            function = functions[0]
            assert any(
                node.domain == function.domain and node.op_type == function.name
                for node in nodes
            )
            infos = {value.name: value for value in function.value_info}
            _empty_shape(infos[function.input[0]], ("B", 4))
            for name in function.output:
                _empty_shape(infos[name], (0, 4))
        if scope.startswith("if-"):
            assert any(node.op_type == "If" for node in nodes)
        _cpu(model, samples, tmp_path)


def _empty_retained_two(x: Any) -> Any:
    empty, _ = jax.lax.top_k(x, 0, axis=2)
    return jax.lax.top_k(empty, 2, axis=1)


@onnx_function(name="NativeEmptyRetainedBatchTopKTwo")
def _empty_named_retained_two(x: Any) -> Any:
    return _empty_retained_two(x)


def _empty_call_named_retained_two(x: Any) -> Any:
    return _empty_named_retained_two(x)


def _empty_jit_retained_two(x: Any) -> Any:
    return jax.jit(_empty_named_retained_two)(x)


def _empty_retained_full(x: Any) -> Any:
    empty, _ = jax.lax.top_k(x, 0, axis=2)
    return jax.lax.top_k(empty, 4, axis=1)


@onnx_function(name="NativeEmptyRetainedBatchTopKFull")
def _empty_named_retained_full(x: Any) -> Any:
    return _empty_retained_full(x)


def _empty_call_named_retained_full(x: Any) -> Any:
    return _empty_named_retained_full(x)


def _empty_jit_retained_full(x: Any) -> Any:
    return jax.jit(_empty_named_retained_full)(x)


def _empty_retained_scope(k: int, scope: str) -> Any:
    if k == 2:
        original, named, jitted = (
            _empty_retained_two,
            _empty_call_named_retained_two,
            _empty_jit_retained_two,
        )
    else:
        assert k == 4
        original, named, jitted = (
            _empty_retained_full,
            _empty_call_named_retained_full,
            _empty_jit_retained_full,
        )
    return {"inline": original, "named": named, "jit": jitted}[scope]


@pytest.mark.parametrize(
    "dtype,scope,k",
    (("float32", "inline", 2), ("float16", "named", 4), ("uint64", "jit", 2)),
)
def test_native_top_k_internal_zero_tail_retains_actual_batch_and_selected_extent(
    dtype: str, scope: str, k: int, tmp_path: Path
) -> None:
    with _runtime_configuration(dtype == "uint64"):
        batch = jax_export.symbolic_shape("B")[0]
        spec = jax.ShapeDtypeStruct((batch, 4, 3), np.dtype(dtype))
        inline = _empty_retained_scope(k, "inline")
        trace = jax.make_jaxpr(inline)(spec)
        equations = [
            item for item in trace.eqns if item.primitive is native_top_k_primitive()
        ]
        assert len(equations) == 2
        first, second = equations
        assert first.params["axis"] == 2 and first.params["k"] == 0
        assert second.params["axis"] == 1 and second.params["k"] == k
        assert second.invars[0] is first.outvars[0]
        first_input = _tensor(
            first.invars[0], _Site("retained empty batch", 23, "top_k")
        )
        for variable in (*first.outvars, *second.outvars):
            target = _tensor(variable, _Site("retained empty batch", 23, "top_k"))
            assert target.symbol == first_input.symbol and target.symbol_axis == 0
        validate_source_jaxpr(trace, 23)
        fn = _empty_retained_scope(k, scope)
        samples = []
        for size in (2, 5):
            data = np.arange(size * 12, dtype=np.dtype(dtype)).reshape(size, 4, 3)
            expected = _empty_pair((size, k, 0), np.dtype(dtype))
            references = _native(fn, (data,), expected, tmp_path / f"B{size}")
            samples.append(((data,), expected, references))
        model = _export(fn, [spec], tmp_path, 23)
        nodes = _empty_artifact(model, ("B", k, 0))
        if scope != "inline":
            functions = [
                function
                for function in model.functions
                if "NativeEmptyRetainedBatchTopK" in function.name
            ]
            assert len(functions) == 1
            function = functions[0]
            assert any(
                node.domain == function.domain and node.op_type == function.name
                for node in nodes
            )
            infos = {value.name: value for value in function.value_info}
            _empty_shape(infos[function.input[0]], ("B", 4, 3))
            for name in function.output:
                _empty_shape(infos[name], ("B", k, 0))
        _cpu(model, samples, tmp_path)


@pytest.mark.parametrize("dtype,k", (("float64", 1), ("int16", 4)))
def test_native_top_k_unrelated_static_zero_uses_selected_slice_extent(
    dtype: str, k: int, tmp_path: Path
) -> None:
    with _runtime_configuration(dtype == "float64"):

        def original(x: Any) -> Any:
            return jax.lax.top_k(x, k, axis=0)

        data = np.empty((4, 0, 3), dtype=np.dtype(dtype))
        expected = _empty_pair((k, 0, 3), data.dtype)
        references = _native(original, (data,), expected, tmp_path)
        trace = jax.make_jaxpr(original)(jax.ShapeDtypeStruct(data.shape, data.dtype))
        validate_source_jaxpr(trace, 23)
        model = _export(
            original, [jax.ShapeDtypeStruct(data.shape, data.dtype)], tmp_path, 23
        )
        _empty_artifact(model, (k, 0, 3))
        _cpu(model, [((data,), expected, references)], tmp_path)


def test_native_top_k_empty_result_preserves_direct_sibling_input(
    tmp_path: Path,
) -> None:
    with _runtime_configuration(False):

        def original(x: Any) -> Any:
            _, indices = jax.lax.top_k(x, 0, axis=0)
            return x, indices

        batch = jax_export.symbolic_shape("B")[0]
        spec = jax.ShapeDtypeStruct((batch, 4), np.int32)
        samples = []
        for size in (2, 5):
            data = np.arange(size * 4, dtype=np.int32).reshape(size, 4)
            frozen = data.tobytes()
            expected = (data.copy(), np.empty((0, 4), dtype=np.int32))
            references = _native(original, (data,), expected, tmp_path / f"B{size}")
            assert data.tobytes() == frozen
            samples.append(((data,), expected, references))
        model = _export(original, [spec], tmp_path, 23)
        nodes = list(_nodes(model.graph))
        assert not any(node.op_type in {"TopK", "BitCast"} for node in nodes)
        _empty_shape(model.graph.output[0], ("B", 4))
        _empty_shape(model.graph.output[1], (0, 4))
        _cpu(model, samples, tmp_path)


def test_native_top_k_empty_output_symbol_shape_and_k_correspondence() -> None:
    with _runtime_configuration(False):
        batch = jax_export.symbolic_shape("B")[0]
        trace = jax.make_jaxpr(_empty_selected_batch)(
            jax.ShapeDtypeStruct((batch, 4), np.float32)
        )
        equation = next(
            item for item in trace.eqns if item.primitive is native_top_k_primitive()
        )
        site = _Site("empty metadata", 23, "top_k")
        source = _tensor(equation.invars[0], site)
        outputs = [_tensor(variable, site) for variable in equation.outvars]
        assert source.symbol is not None
        _validate(equation, {source.symbol}, 23)
        invented = [
            replace(outputs[0], symbol=source.symbol, symbol_axis=0),
            outputs[1],
        ]
        with pytest.raises(SourcePrecisionError, match="batch scope"):
            validate_native_top_k(equation, [source], invented, {source.symbol}, site)
        wrong_shape = [replace(outputs[0], shape=(0, 3)), outputs[1]]
        with pytest.raises(SourcePrecisionError, match="result shape"):
            validate_native_top_k(
                equation, [source], wrong_shape, {source.symbol}, site
            )
        for k in (True, -1, 1):
            with pytest.raises(SourcePrecisionError, match="TopK"):
                _validate(
                    equation.replace(params={**equation.params, "k": k}),
                    {source.symbol},
                    23,
                )
        with pytest.raises(SourcePrecisionError, match="symbolic extent"):
            _validate(equation, set(), 23)
        retained = jax.make_jaxpr(_empty_retained_scope(2, "inline"))(
            jax.ShapeDtypeStruct((batch, 4, 3), np.float32)
        )
        second = [
            item for item in retained.eqns if item.primitive is native_top_k_primitive()
        ][1]
        input_contract = _tensor(second.invars[0], site)
        result_contracts = [_tensor(variable, site) for variable in second.outvars]
        removed = [
            replace(result_contracts[0], symbol=None, symbol_axis=None),
            result_contracts[1],
        ]
        with pytest.raises(SourcePrecisionError, match="batch scope"):
            validate_native_top_k(
                second, [input_contract], removed, {source.symbol}, site
            )


def _bf16_word_input(contains_nan: bool) -> np.ndarray:
    # Construct genuine BF16 storage from declared words; no numerical widening.
    if contains_nan:
        words = [
            [0x7F80, 0x7FC1, 0x7FC2, 0x7F81],
            [0xFF80, 0xFFC1, 0xFFC2, 0xFF81],
            [0xFFC2, 0xFFC1, 0xFFC2, 0xFFC3],
            [0x7FC1, 0x7FC1, 0x7F81, 0x0001],
            [0x7F81, 0x7F82, 0x7FC1, 0x7FC2],
            [0xFFC1, 0xFFC2, 0xFF81, 0xFF82],
        ]
    else:
        words = [
            [0x8000, 0x0000, 0x8000, 0x0000],
            [0x8000, 0x8000, 0x8000, 0x8000],
            [0x7F80, 0x7F7F, 0xFF80, 0xFF7F],
            [0xFF80, 0xFF80, 0xFF80, 0xFF80],
            [0x0001, 0x007F, 0x8001, 0x807F],
            [0x8001, 0x807F, 0xBF80, 0xBF80],
            [0xFF7F, 0xFF7F, 0xFF80, 0xFF80],
            [0x3F80, 0x3F80, 0x3F00, 0xC000],
        ]
    return np.asarray(words, np.uint16).view(np.dtype("bfloat16"))


def _bf16_exact_comparison(
    actual: tuple[np.ndarray, ...], reference: tuple[np.ndarray, ...]
) -> dict[str, Any]:
    """Keep exact native-equivalence outcomes separate from portable semantics."""
    assert len(actual) == len(reference) == 2
    criteria = precision.SampleCriteria("bitwise")
    outputs = []
    for value, expected in zip(actual, reference, strict=True):
        error = precision._compare(value, expected, criteria)
        outputs.append(
            dict(
                error=error,
                actual_dtype=value.dtype.name,
                reference_dtype=expected.dtype.name,
                actual_shape=value.shape,
                reference_shape=expected.shape,
                different_words=(
                    int(
                        np.count_nonzero(
                            _portable_words(value) != _portable_words(expected)
                        )
                    )
                    if value.dtype == expected.dtype and value.shape == expected.shape
                    else None
                ),
            )
        )
    status = (
        precision.CheckStatus.FAILED
        if any(output["error"] is not None for output in outputs)
        else precision.CheckStatus.VALIDATED
    )
    return dict(status=status.value, criteria=asdict(criteria), outputs=outputs)


def _bf16_native(
    fn: Any,
    inputs: tuple[Any, ...],
    data: np.ndarray,
    contains_nan: bool,
    stable: bool,
    directory: Path,
) -> tuple[tuple[np.ndarray, ...], ...]:
    compiled = jax.jit(_fresh(fn))
    expected = _portable_expected(data, 2)
    references = []
    observations = []
    for label, operation in (("fresh", compiled), ("warmed", compiled), ("eager", fn)):
        try:
            returned = operation(*inputs)
        except BaseException as cause:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{label}-exception.json").write_text(
                json.dumps(
                    dict(
                        status="raised",
                        type=type(cause).__qualname__,
                        message=str(cause),
                    )
                )
            )
            raise
        path = directory / f"{label}.npz"
        actual = _retain(returned, path, label)
        observation = dict(
            native_role=label,
            retained_file=path.name,
            retained_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            exact_portable_word_comparison=_bf16_exact_comparison(actual, expected),
        )
        observations.append(observation)
        (directory / "native-word-observations.json").write_text(
            json.dumps(
                dict(contains_nan=contains_nan, native_roles=observations), indent=2
            )
        )
        assert actual[0].dtype == data.dtype and actual[1].dtype == np.dtype(np.int32)
        assert all(
            value.shape == want.shape
            for value, want in zip(actual, expected, strict=True)
        )
        if not contains_nan:
            _portable_relation(actual, data, 2, -1, stable)
        # NaN comparisons above remain their actual FAILED/VALIDATED results.
        # They do not become acceptance by using a portable oracle or masking.
        references.append(actual)
    return tuple(references)


def _bf16_cpu(
    model: Any, samples: list[Any], contains_nan: bool, directory: Path
) -> None:
    blob = model.SerializeToString()
    digest = hashlib.sha256(blob).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "artifact.onnx").write_bytes(blob)
    records = []
    for label, profile in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = profile
            session = ort.InferenceSession(
                blob, options, providers=["CPUExecutionProvider"]
            )
            session.disable_fallback()
            for index, (inputs, expected, references, native_directory) in enumerate(
                samples
            ):
                feeds = {
                    port.name: value
                    for port, value in zip(session.get_inputs(), inputs, strict=True)
                }
                returned = precision._run_cpu_samples(session, feeds, model)
                path = directory / f"cpu-{label}-{index}.npz"
                actual = _retain(returned, path, label)
                portable = _bf16_exact_comparison(actual, expected)
                assert len(references) == 3
                native_equivalence = []
                for role, reference in zip(
                    ("fresh", "warmed", "eager"), references, strict=True
                ):
                    native_file = native_directory / f"{role}.npz"
                    native_equivalence.append(
                        dict(
                            native_role=role,
                            native_file=str(native_file.relative_to(directory)),
                            native_file_sha256=hashlib.sha256(
                                native_file.read_bytes()
                            ).hexdigest(),
                            comparison=_bf16_exact_comparison(actual, reference),
                        )
                    )
                records.append(
                    dict(
                        artifact_sha256=digest,
                        cpu_profile=label,
                        sample=index,
                        actual_file=path.name,
                        actual_file_sha256=hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                        contains_nan=contains_nan,
                        portable_original_word_semantics=portable,
                        native_equivalence=native_equivalence,
                    )
                )
                (directory / "portable-and-native-equivalence.json").write_text(
                    json.dumps(records, indent=2)
                )
                _same(actual, expected)
                if not contains_nan:
                    for reference in references:
                        _same(actual, reference)
                # A validated portable check never rewrites a FAILED native check.
                assert portable["status"] == precision.CheckStatus.VALIDATED.value
                assert session.get_providers() == ["CPUExecutionProvider"]
            assert session.get_session_options().graph_optimization_level == profile
            assert hashlib.sha256(model.SerializeToString()).hexdigest() == digest
        except BaseException as cause:
            (directory / f"cpu-{label}-exception.json").write_text(
                json.dumps(
                    dict(
                        status="raised",
                        type=type(cause).__qualname__,
                        message=str(cause),
                    )
                )
            )
            raise


@pytest.mark.parametrize(
    "scope,contains_nan",
    [
        ("inline", False),
        ("named", False),
        ("jit", False),
        ("if-false", False),
        ("if-true", False),
        ("inline", True),
        ("named", True),
    ],
    ids=[
        "finite-inline",
        "finite-named",
        "finite-jit",
        "finite-if-false",
        "finite-if-true",
        "nan-portable-inline",
        "nan-portable-named",
    ],
)
def test_source_bfloat16_top_k_portable_words_and_native_equivalence(
    scope: str, contains_nan: bool, tmp_path: Path
) -> None:
    """Require portable CPU words; preserve separate strict native NaN verdicts."""
    with _runtime_configuration(scope not in {"inline", "jit"}):
        available = "is_stable" in native_top_k_parameter_names()
        stabilities = (
            (None, False)
            if available and contains_nan and scope == "inline"
            else (None,)
        )
        symbol_axis = 0 if scope == "inline" else 1
        payload = _bf16_word_input(contains_nan)
        rows = payload.shape[0]
        for stability in stabilities:
            directory = tmp_path / (
                "native-default" if stability is None else "native-unstable"
            )
            batch = jax_export.symbolic_shape("B")[0]
            spec = jax.ShapeDtypeStruct((batch, rows, 4), payload.dtype)
            fn, inline = _portable_scope(stability, scope, symbol_axis)
            native = jax.make_jaxpr(_fresh(inline))(spec)
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                activated = jax.make_jaxpr(_fresh(inline))(spec)
            assert typed_trace_manifest(native) == typed_trace_manifest(activated)
            assert _contexts(native) == _contexts(activated)
            _require_internal_native_axis(native, symbol_axis)
            _require_internal_native_axis(activated, symbol_axis)
            equation = next(
                e for e in native.eqns if e.primitive is native_top_k_primitive()
            )
            assert set(equation.params) == native_top_k_parameter_names()
            if available:
                assert equation.params["is_stable"] is (
                    True if stability is None else False
                )
            else:
                assert stability is None and "is_stable" not in equation.params
            validate_source_jaxpr(native, 26)
            samples = []
            for size in (2, 5):
                data = np.repeat(payload[None, ...], size, axis=0)
                other = np.flip(data, axis=-1).copy()
                if scope.startswith("if-"):
                    predicate = np.asarray(scope == "if-true", dtype=np.bool_)
                    inputs = (data, other, predicate)
                    selected = data if bool(predicate) else other
                else:
                    inputs, selected = (data,), data
                original_words = tuple(value.tobytes() for value in inputs)
                expected = _portable_expected(selected, 2)
                native_directory = directory / f"B{size}"
                references = _bf16_native(
                    fn,
                    inputs,
                    selected,
                    contains_nan,
                    stability is not False,
                    native_directory,
                )
                assert tuple(value.tobytes() for value in inputs) == original_words
                samples.append((inputs, expected, references, native_directory))
            specs = (
                [spec, spec, jax.ShapeDtypeStruct((), np.bool_)]
                if scope.startswith("if-")
                else [spec]
            )
            model = _export(fn, specs, directory, 26)
            if model is None:
                continue
            _portable_artifact(model, "bfloat16")
            assert (
                model.graph.output[0].type.tensor_type.elem_type
                == onnx.TensorProto.BFLOAT16
            )
            assert (
                model.graph.output[1].type.tensor_type.elem_type
                == onnx.TensorProto.INT32
            )
            if scope != "inline":
                functions = [
                    function
                    for function in model.functions
                    if "NativePortableWordTopK" in function.name
                ]
                assert len(functions) == 1
                function = functions[0]
                assert any(
                    node.domain == function.domain and node.op_type == function.name
                    for owner in (model.graph, *model.functions)
                    for node in _nodes(owner)
                )
                infos = {value.name: value for value in function.value_info}
                for name, shape, dtype in zip(
                    (*function.input, *function.output),
                    ((rows, "B", 4), (rows, "B", 2), (rows, "B", 2)),
                    (
                        onnx.TensorProto.BFLOAT16,
                        onnx.TensorProto.BFLOAT16,
                        onnx.TensorProto.INT32,
                    ),
                    strict=True,
                ):
                    tensor = infos[name].type.tensor_type
                    assert tensor.elem_type == dtype
                    assert (
                        tuple(
                            dim.dim_param or dim.dim_value for dim in tensor.shape.dim
                        )
                        == shape
                    )
            _bf16_cpu(model, samples, contains_nan, directory)


def test_source_bfloat16_top_k_explicit26_and_typed_empty23(tmp_path: Path) -> None:
    with _runtime_configuration(False):
        spec = jax.ShapeDtypeStruct((3, 4), np.dtype("bfloat16"))
        with pytest.raises(SourcePrecisionError) as caught:
            ui._to_onnx(
                _top_k, [spec], precision_policy=PrecisionPolicy.SOURCE, opset=23
            )
        assert caught.value.category == "representation-limit"
        assert "BitCast" in str(caught.value) and "26" in str(caught.value)

        def empty(x: Any) -> Any:
            return jax.lax.top_k(x, 0)

        data = _bf16_word_input(True)
        expected = _portable_expected(data, 0)
        references = _native(empty, (data,), expected, tmp_path / "native-empty")
        model = _export(
            empty, [jax.ShapeDtypeStruct(data.shape, data.dtype)], tmp_path, 23
        )
        assert not any(node.op_type in {"TopK", "BitCast"} for node in model.graph.node)
        _bf16_cpu(
            model,
            [((data,), expected, references, tmp_path / "native-empty")],
            False,
            tmp_path,
        )
