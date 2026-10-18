# tests/extra_tests/converter/test_source_array_creation_promotion.py

"""Finite SOURCE diagnostics for where/full promotion and canonical ownership.

Original JAX traces and eager references precede converter patch activation.
The local validator checks the exercised structures; it grants no product
admission and never claims an audited lowering or an available runtime kernel.
"""

from collections.abc import Callable, Iterator
from copy import copy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx, user_interface
from jax2onnx._compat.jax import DropVar, Literal, ShapedArray, scan_arity
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins.plugin_system import _SourceFunctionTrace
from jax2onnx.precision import CheckStatus, get_precision_report

_ORIGINAL_WHERE = jnp.where
_ORIGINAL_FULL = jnp.full
_DTYPES = {"bool", "int32", "int64", "float16", "bfloat16", "float32", "float64"}
_PLACEMENT = {
    "sharding",
    "out_sharding",
    "in_shardings",
    "out_shardings",
    "in_layouts",
    "out_layouts",
    "ctx_mesh",
}


@pytest.fixture(autouse=True)
def _restore_jax_settings() -> Iterator[None]:
    previous = {
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "jax_platforms": jax.config.jax_platforms,
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "jax_numpy_dtype_promotion": jax.config.jax_numpy_dtype_promotion,
        "jax_numpy_rank_promotion": jax.config.jax_numpy_rank_promotion,
    }
    try:
        yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)


def _body(value: Any) -> Any:
    inner = getattr(value, "jaxpr", value)
    return inner if inner is not value else value


def _typed(aval: Any) -> tuple[str, tuple[int, ...], bool]:
    dtype = np.dtype(aval.dtype).name
    shape = tuple(aval.shape)
    assert dtype in _DTYPES
    assert all(isinstance(dim, (int, np.integer)) and dim >= 0 for dim in shape)
    assert type(aval.weak_type) is bool
    return dtype, shape, aval.weak_type


def _signature(values: Any) -> tuple[Any, ...]:
    return tuple(_typed(var.aval) for var in values)


def _shape_spec(aval: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)


def _placement(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (tuple, list)):
        return all(_placement(item) for item in value)
    return type(value).__name__ in {"UnspecifiedValue", "SingleDeviceSharding"} or (
        type(value).__name__ in {"Mesh", "AbstractMesh"} and bool(value.empty)
    )


class _ArrayCreationAudit:
    """Validate only these typed primitives and the tested recursive scopes."""

    def __init__(self) -> None:
        self.equations: list[Any] = []
        self.roots: list[Any] = []

    def __call__(self, jaxpr: Any, opset: int) -> None:
        assert opset == 23
        self.roots.append(_body(jaxpr))
        with precision_policy_scope(PrecisionPolicy.SOURCE):
            self._walk(_body(jaxpr), set())

    def _walk(self, jaxpr: Any, ancestors: set[int]) -> None:
        assert id(jaxpr) not in ancestors and len(ancestors) < 12
        assert not jaxpr.effects
        ancestors = ancestors | {id(jaxpr)}
        bound = {id(var) for var in (*jaxpr.constvars, *jaxpr.invars)}
        for var in (*jaxpr.constvars, *jaxpr.invars, *jaxpr.outvars):
            _typed(var.aval)
        for eqn in jaxpr.eqns:
            assert not eqn.effects
            name = eqn.primitive.name
            self.equations.append(eqn)
            for var in eqn.invars:
                _typed(var.aval)
                assert isinstance(var, Literal) or id(var) in bound
            for var in eqn.outvars:
                _typed(var.aval)
                assert isinstance(var, DropVar) or id(var) not in bound
                bound.add(id(var))
            params = eqn.params
            if name in {"jit", "pjit"}:
                allowed = {
                    "jaxpr",
                    "call_jaxpr",
                    "name",
                    "inline",
                    "keep_unused",
                    "donated_invars",
                    "compiler_options_kvs",
                } | _PLACEMENT
                assert set(params) <= allowed
                child = _body(params.get("jaxpr", params.get("call_jaxpr")))
                assert child is not None
                assert _signature(eqn.invars) == _signature(child.invars)
                assert _signature(eqn.outvars) == _signature(child.outvars)
                assert not params.get("compiler_options_kvs", ())
                assert all(
                    _placement(params[key]) for key in _PLACEMENT & params.keys()
                )
                self._walk(child, ancestors)
            elif name == "cond":
                assert set(params) == {"branches"}
                assert _signature(eqn.invars[:1]) == (("int32", (), False),)
                assert len(params["branches"]) == 2
                for branch in params["branches"]:
                    child = _body(branch)
                    assert _signature(eqn.invars[1:]) == _signature(child.invars)
                    assert _signature(eqn.outvars) == _signature(child.outvars)
                    self._walk(child, ancestors)
            elif name == "scan":
                allowed = {
                    "jaxpr",
                    "length",
                    "reverse",
                    "linear",
                    "num_consts",
                    "num_carry",
                    "unroll",
                    "_split_transpose",
                    "ft_in",
                    "ft_out",
                }
                assert set(params) <= allowed
                child = _body(params["jaxpr"])
                nc, nk, nx = scan_arity(params, len(child.invars))
                assert nx == 0 and type(params["length"]) is int
                assert params["length"] >= 0 and params["reverse"] is False
                assert _signature(eqn.invars) == _signature(child.invars)
                assert _signature(eqn.outvars) == _signature(child.outvars)
                assert _signature(eqn.invars[nc : nc + nk]) == _signature(eqn.outvars)
                self._walk(child, ancestors)
            elif name.startswith("onnx_fn::"):
                # The callback also sees the actual retraced body during lowering.
                assert set(params) == {"__jax2onnx_source_trace"}
                trace = params["__jax2onnx_source_trace"]
                assert isinstance(trace, _SourceFunctionTrace)
                assert _signature(eqn.outvars) == tuple(map(_typed, trace.output_avals))
            else:
                self._leaf(eqn)
        for var in jaxpr.outvars:
            assert isinstance(var, Literal) or id(var) in bound

    @staticmethod
    def _leaf(eqn: Any) -> None:
        name, params = eqn.primitive.name, eqn.params
        inputs = tuple(_shape_spec(var.aval) for var in eqn.invars)
        if name == "jax.numpy.where":
            assert len(inputs) == 3 and len(eqn.outvars) == 1
            assert set(params) == {"source_dtype", "source_weak_type"}
            assert isinstance(params["source_dtype"], np.dtype)
            assert type(params["source_weak_type"]) is bool
            native = jax.eval_shape(_ORIGINAL_WHERE, *inputs)
            assert params["source_dtype"] == native.dtype
            assert params["source_weak_type"] is native.weak_type
            assert _signature(eqn.outvars) == (_typed(native),)
        elif name == "jax.numpy.full":
            assert len(inputs) == 1 and len(eqn.outvars) == 1
            assert set(params) == {"shape"}
            assert isinstance(params["shape"], tuple)
            assert all(type(dim) is int and dim >= 0 for dim in params["shape"])
            native = jax.eval_shape(
                lambda fill: _ORIGINAL_FULL(params["shape"], fill), *inputs
            )
            assert _signature(eqn.outvars) == (_typed(native),)
        else:
            rules = {
                "add": (2, set()),
                "sub": (2, set()),
                "mul": (2, set()),
                "neg": (1, set()),
                "lt": (2, set()),
                "gt": (2, set()),
                "ne": (2, set()),
                "select_n": (3, set()),
                "convert_element_type": (1, {"new_dtype", "weak_type", "sharding"}),
                "broadcast_in_dim": (1, {"shape", "broadcast_dimensions", "sharding"}),
                "reshape": (1, {"new_sizes", "dimensions", "sharding"}),
                "slice": (1, {"start_indices", "limit_indices", "strides"}),
                "squeeze": (1, {"dimensions"}),
                "jax.numpy.ones": (0, {"shape", "dtype"}),
                "jax.numpy.zeros": (0, {"shape", "dtype"}),
            }
            assert name in rules, f"Outside array-creation diagnostic: {name}"
            arity, allowed = rules[name]
            assert len(inputs) == arity and len(eqn.outvars) == 1
            assert set(params) <= allowed
            assert all(_placement(params[key]) for key in _PLACEMENT & params.keys())
        inferred, effects = eqn.primitive.abstract_eval(
            *(var.aval for var in eqn.invars), **params
        )
        assert not effects
        inferred_outputs = inferred if eqn.primitive.multiple_results else (inferred,)
        assert tuple(map(_typed, inferred_outputs)) == _signature(eqn.outvars)


def _reference(
    fn: Callable[..., Any], arguments: tuple[Any, ...]
) -> tuple[Any, list[np.ndarray]]:
    assert jnp.where is _ORIGINAL_WHERE and jnp.full is _ORIGINAL_FULL
    arrays = tuple(jnp.asarray(value) for value in arguments)
    specs = [_shape_spec(value) for value in arrays]
    original = jax.make_jaxpr(fn)(*specs).jaxpr
    expected = [np.asarray(value) for value in jax.tree.leaves(fn(*arrays))]
    return original, expected


def _assert_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def _cpu(
    proto: onnx.ModelProto,
    arguments: tuple[Any, ...],
    expected: list[np.ndarray],
    *,
    optimized: bool = False,
) -> None:
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = settings.inter_op_num_threads = 1
    settings.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimized
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        proto.SerializeToString(), settings, providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        None,
        {
            item.name: np.asarray(value)
            for item, value in zip(session.get_inputs(), arguments)
        },
    )
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        _assert_exact(result, reference)


def _export(
    fn: Callable[..., Any], arguments: tuple[Any, ...] = (), *, cpu: bool = True
) -> tuple[onnx.ModelProto, _ArrayCreationAudit, Any, list[np.ndarray]]:
    original, expected = _reference(fn, arguments)
    specs = [_shape_spec(jnp.asarray(value)) for value in arguments]
    audit = _ArrayCreationAudit()

    def observe(jaxpr: Any, opset: int) -> None:
        if not audit.roots:
            assert _signature(jaxpr.invars) == _signature(original.invars)
            assert _signature(jaxpr.outvars) == _signature(original.outvars)
        audit(jaxpr, opset)

    artifact = user_interface._to_onnx(
        fn,
        specs,
        opset=23,
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=observe,
    )
    proto = onnx.load_model_from_string(artifact.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-diagnostic-v1"
    assert (
        next(check.status for check in report.checks if check.category == "lowering")
        is CheckStatus.UNVERIFIED
    )
    assert tuple(
        (item.dtype, item.shape, item.weak_type) for item in report.traced_outputs
    ) == _signature(original.outvars)
    for output, reference in zip(proto.graph.output, expected):
        assert (
            output.type.tensor_type.elem_type
            == onnx.helper.np_dtype_to_tensor_dtype(reference.dtype)
        )
        assert (
            tuple(dim.dim_value for dim in output.type.tensor_type.shape.dim)
            == reference.shape
        )
    if cpu:
        _cpu(proto, arguments, expected)
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    return proto, audit, original, expected


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("promotion", ["standard", "strict"])
@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.float32, np.int32])
def test_where_native_promotion_and_weak_inputs(
    x64: bool, promotion: str, dtype: Any
) -> None:
    with jax.enable_x64(x64), jax.numpy_dtype_promotion(promotion):
        data = np.asarray([1, 2], dtype=dtype)
        weak = 7 if dtype is np.int32 else 0.5

        def fn(mask: Any, value: Any) -> Any:
            return jnp.where(mask, value, weak)

        _, audit, _, _ = _export(
            fn, (np.asarray([True, False]), data), cpu=dtype is not jnp.bfloat16
        )
        eqn = next(eqn for eqn in audit.equations if eqn.primitive.name == "select_n")
        assert np.dtype(eqn.outvars[0].aval.dtype) == np.dtype(dtype)
        assert eqn.outvars[0].aval.weak_type is False
        assert not eqn.params


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("promotion", ["standard", "strict"])
@pytest.mark.parametrize(
    "kind",
    [
        "float",
        "int",
        "bool",
        "numeric_condition",
        "broadcast",
        "mixed_strong",
        "integer_weak_float",
        "float32_weak_int",
    ],
)
def test_where_weak_outputs_broadcast_and_native_strict_errors(
    x64: bool, promotion: str, kind: str
) -> None:
    with jax.enable_x64(x64), jax.numpy_dtype_promotion(promotion):
        if kind == "float":
            fn, arguments = lambda mask: jnp.where(mask, 0.5, 0.25), (
                np.asarray([True, False]),
            )
        elif kind == "int":
            fn, arguments = lambda mask: jnp.where(mask, 7, 11), (
                np.asarray([True, False]),
            )
        elif kind == "bool":
            fn, arguments = lambda mask: jnp.where(mask, True, False), (
                np.asarray([True, False]),
            )
        elif kind == "numeric_condition":
            fn, arguments = lambda mask: jnp.where(mask, 0.5, 0.25), (
                np.asarray([0, -1], np.int32),
            )
        elif kind == "broadcast":

            def fn(mask: Any, value: Any) -> Any:
                return jnp.where(mask, value, 0.5)

            arguments = np.asarray([[True], [False]]), np.asarray(
                [[1, 2, 3]], np.float32
            )
        elif kind == "integer_weak_float":

            def fn(mask: Any, value: Any) -> Any:
                return jnp.where(mask, value, 0.5)

            arguments = np.asarray([True, False]), np.asarray([7, 11], np.int32)
        elif kind == "float32_weak_int":

            def fn(mask: Any, value: Any) -> Any:
                return jnp.where(mask, value, 7)

            arguments = np.asarray([True, False]), np.asarray([0.125, 0.25], np.float32)
        else:

            def fn(mask: Any, value: Any) -> Any:
                return jnp.where(mask, value, np.asarray([7, 11], np.int32))

            arguments = np.asarray([True, False]), np.asarray([0.125, 0.25], np.float32)
        if kind in {"mixed_strong", "integer_weak_float"} and promotion == "strict":
            with pytest.raises(jax.dtypes.TypePromotionError):
                _reference(fn, arguments)
            with pytest.raises(jax.dtypes.TypePromotionError):
                user_interface._to_onnx(
                    fn,
                    [_shape_spec(jnp.asarray(value)) for value in arguments],
                    precision_policy=PrecisionPolicy.SOURCE,
                    diagnostic_jaxpr_validator=_ArrayCreationAudit(),
                )
            assert active_precision_policy() is PrecisionPolicy.LEGACY
        else:
            _export(fn, arguments)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("promotion", ["standard", "strict"])
@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.float32, np.int32])
def test_full_preserves_weak_fill_for_downstream_promotion(
    x64: bool, promotion: str, dtype: Any
) -> None:
    with jax.enable_x64(x64), jax.numpy_dtype_promotion(promotion):
        fill = 7 if dtype is np.int32 else 0.5

        def fn(value: Any) -> Any:
            return jnp.full((2,), fill), jnp.full((2,), fill) + value

        _, audit, _, _ = _export(
            fn, (np.asarray([1, 2], dtype=dtype),), cpu=dtype is not jnp.bfloat16
        )
        assert all(
            eqn.outvars[0].aval.weak_type
            for eqn in audit.equations
            if eqn.primitive.name == "jax.numpy.full"
        )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "typed_float16",
        "typed_bfloat16",
        "typed_float32",
        "typed_int32",
        "explicit_float16",
        "explicit_bfloat16",
        "explicit_float32",
        "explicit_int32",
        "row",
        "dynamic_scalar",
        "ones_zeros",
    ],
)
def test_full_strong_and_existing_creation_controls(x64: bool, kind: str) -> None:
    with jax.enable_x64(x64):
        if kind.startswith(("typed_", "explicit_")):
            dtype = np.dtype(kind.split("_", 1)[1])
            if kind.startswith("typed_"):
                fn, arguments = lambda: jnp.full((2,), dtype.type(0.5)), ()
            else:
                fn, arguments = lambda: jnp.full((2,), 0.5, dtype=dtype), ()
        elif kind == "row":
            fn, arguments = lambda value: jnp.full((2, 3), value), (
                np.asarray([[1, 2, 3]], np.float32),
            )
        elif kind == "dynamic_scalar":
            fn, arguments = lambda value: jnp.full((2,), value), (
                np.asarray(0.5, np.float32),
            )
        else:
            fn, arguments = (
                lambda: (jnp.ones((2,)), jnp.zeros((2,), dtype=np.int32)),
                (),
            )
        _export(fn, arguments, cpu=not kind.endswith("bfloat16"))


_CAPTURED = np.asarray([16777217, 2147483647], np.int32)


def _shared(mask: Any, floating: Any, integer: Any) -> Any:
    before = _CAPTURED + integer
    selected = jnp.where(mask, floating, _CAPTURED)
    after = _CAPTURED + integer
    return _CAPTURED, before, selected, after


@onnx_function
def _named_shared(mask: Any, floating: Any, integer: Any) -> Any:
    return _shared(mask, floating, integer)


@pytest.mark.parametrize("scope", ["inline", "function", "if", "loop"])
@pytest.mark.parametrize("direct", [False, True])
def test_where_never_retypes_shared_captured_int32(scope: str, direct: bool) -> None:
    with jax.enable_x64(True), jax.numpy_dtype_promotion("standard"):

        def target(mask: Any, floating: Any, integer: Any) -> Any:
            if scope == "function":
                result = _named_shared(mask, floating, integer)
            elif scope == "if":
                result = jax.lax.cond(
                    mask[0],
                    lambda _: _shared(mask, floating, integer),
                    lambda _: _shared(mask, floating, integer),
                    None,
                )
            elif scope == "loop":
                initial = (_CAPTURED, _CAPTURED, floating, _CAPTURED)
                result = jax.lax.fori_loop(
                    0, 2, lambda _, state: _shared(mask, floating, integer), initial
                )
            else:
                result = _shared(mask, floating, integer)
            return result if direct else result[1:]

        args = (
            np.asarray([True, False]),
            np.asarray([0.125, 0.25], np.float64),
            np.asarray([0, 0], np.int32),
        )
        proto, _, _, expected = _export(target, args)
        _cpu(proto, args, expected, optimized=True)
        tensors = list(proto.graph.initializer)
        nodes = list(proto.graph.node)
        for function in proto.functions:
            nodes.extend(function.node)
        for node in nodes:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    tensors.extend(attribute.g.initializer)
                    nodes.extend(attribute.g.node)
                elif attribute.type == onnx.AttributeProto.TENSOR:
                    tensors.append(attribute.t)
        captured = [
            tensor
            for tensor in tensors
            if tensor.data_type == onnx.TensorProto.INT32
            and tuple(tensor.dims) == (2,)
            and onnx.numpy_helper.to_array(tensor).tobytes() == _CAPTURED.tobytes()
        ]
        assert (
            captured
        ), "The canonical captured INT32 payload must survive serialization"
        assert any(
            node.op_type == "Cast"
            and any(
                attribute.name == "to" and attribute.i == onnx.TensorProto.DOUBLE
                for attribute in node.attribute
            )
            for node in nodes
        )


@pytest.mark.parametrize("transform", ["vmap", "jvp"])
def test_source_where_transforms_preserve_native_types_and_values(
    transform: str,
) -> None:
    with jax.enable_x64(True):

        def select(value: Any) -> Any:
            return jnp.where(value > 0, value, 0.5)

        if transform == "vmap":
            fn = jax.vmap(select)
        else:

            def fn(value: Any) -> Any:
                return jax.jvp(select, (value,), (jnp.ones_like(value),))

        _export(fn, (np.asarray([-1, 0.25], np.float32),))


def _cache_operation(mask: Any, value: Any) -> Any:
    return jnp.where(mask, value, 0.5) + jnp.full((2,), 0.25)


@onnx_function
def _cache_named(mask: Any, value: Any) -> Any:
    return _cache_operation(mask, value)


@onnx_function
def _named_boolean_where(condition: Any, x: Any, y: Any) -> Any:
    return jnp.where(condition, x, y)


def _boolean_case(kind: str) -> tuple[np.ndarray, ...]:
    if kind == "truth":
        indices = np.arange(8, dtype=np.int32)
        return tuple((indices & bit) != 0 for bit in (4, 2, 1))
    if kind == "scalar":
        return np.asarray(True), np.asarray(False), np.asarray(True)
    if kind == "broadcast":
        return (
            np.asarray([[False], [True]]),
            np.asarray([[False, True, False]]),
            np.asarray(True),
        )
    if kind == "numeric":
        return (
            np.asarray([[0], [-1]], np.int32),
            np.asarray([[False, True, False]]),
            np.asarray(True),
        )
    return (
        np.empty((0, 1), np.bool_),
        np.asarray([[False, True, False]]),
        np.asarray(True),
    )


@pytest.mark.parametrize("kind", ["truth", "scalar", "broadcast", "numeric", "empty"])
@pytest.mark.parametrize("scope", ["inline", "function", "if", "loop"])
def test_source_boolean_selection_has_exact_logical_cpu_lowering(
    kind: str, scope: str
) -> None:
    arguments = (*_boolean_case(kind), np.asarray(True))

    def target(condition: Any, x: Any, y: Any, gate: Any) -> Any:
        if scope == "function":
            return _named_boolean_where(condition, x, y)
        if scope == "if":
            return jax.lax.cond(
                gate,
                lambda: jnp.where(condition, x, y),
                lambda: jnp.where(condition, y, x),
            )
        if scope == "loop":
            shape = jnp.broadcast_shapes(condition.shape, x.shape, y.shape)
            initial = jnp.broadcast_to(y, shape)
            return jax.lax.fori_loop(
                0, 2, lambda _, carry: jnp.where(condition, x, carry), initial
            )
        return jnp.where(condition, x, y)

    proto, audit, _, expected = _export(target, arguments)
    _cpu(proto, arguments, expected, optimized=True)
    nodes = list(proto.graph.node)
    for function in proto.functions:
        nodes.extend(function.node)
    for node in nodes:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                nodes.extend(attribute.g.node)
    assert "Where" not in {node.op_type for node in nodes}
    if kind == "empty":
        assert all(eqn.primitive.name != "select_n" for eqn in audit.equations)
    else:
        assert {"Not", "And", "Or"} <= {node.op_type for node in nodes}


def test_source_boolean_intermediates_keep_their_own_broadcast_shapes() -> None:
    arguments = _boolean_case("broadcast")
    fn = _ORIGINAL_WHERE

    def target(condition: Any, x: Any, y: Any) -> Any:
        return jnp.where(condition, x, y)

    original, _ = _reference(fn, arguments)
    raw = conversion_api._convert_to_ir(
        fn=target,
        inputs=[_shape_spec(jnp.asarray(value)) for value in arguments],
        input_params=None,
        model_name="boolean_pairwise_shapes",
        opset=23,
        enable_double_precision=False,
        record_primitive_calls_file=None,
        precision_policy=PrecisionPolicy.SOURCE,
        jaxpr_validator=_ArrayCreationAudit(),
    )
    logical = [node for node in raw.graph if node.op_type in {"Not", "And", "Or"}]
    assert [node.op_type for node in logical] == ["Not", "And", "And", "Or"]
    assert [tuple(node.outputs[0].shape) for node in logical] == [
        (2, 3),
        (2, 3),
        (2, 3),
        (2, 3),
    ]
    assert raw.graph.outputs[0].dtype.numpy() == original.outvars[0].aval.dtype


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scope", ["inline", "function", "if"])
def test_native_legacy_source_trace_caches_are_isolated_without_clearing(
    reverse: bool, scope: str
) -> None:
    with jax.enable_x64(True):

        def target(mask: Any, value: Any) -> Any:
            if scope == "function":
                return _cache_named(mask, value)
            if scope == "if":
                return jax.lax.cond(
                    mask[0],
                    lambda _: _cache_operation(mask, value),
                    lambda _: -_cache_operation(mask, value),
                    None,
                )
            return _cache_operation(mask, value)

        fn = jax.jit(target)
        args = np.asarray([True, False]), np.asarray([0.125, 0.25], np.float32)
        original, expected = _reference(fn, args)
        order = ("native", "legacy", "source", "native")
        if reverse:
            order = ("native", "source", "legacy", "native")
        for kind in order:
            if kind == "native":
                native, values = _reference(fn, args)
                assert _signature(native.outvars) == _signature(original.outvars)
                for actual, reference in zip(values, expected):
                    _assert_exact(actual, reference)
            elif kind == "source":
                _export(fn, args)
            else:
                to_onnx(fn, [_shape_spec(jnp.asarray(value)) for value in args])
            assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_dtype",
        "unknown_param",
        "dtype_string",
        "wrong_dtype",
        "weak_int",
        "wrong_weak",
        "wrong_output",
    ],
)
def test_local_diagnostic_rejects_inconsistent_resolved_where_metadata(
    corruption: str,
) -> None:
    # The retained custom primitive still requires all seven schema guards;
    # native SOURCE Where now forwards JAX instead of constructing this schema.
    from jax2onnx.plugins.jax.numpy.where import JnpWherePlugin

    with precision_policy_scope(PrecisionPolicy.SOURCE):
        trace = jax.make_jaxpr(
            lambda mask, value: JnpWherePlugin._PRIM.bind(
                mask,
                value,
                0.5,
                source_dtype=np.dtype(np.float32),
                source_weak_type=False,
            )
        )(jax.ShapeDtypeStruct((2,), np.bool_), jax.ShapeDtypeStruct((2,), np.float32))
    eqn = copy(
        next(
            eqn for eqn in _body(trace).eqns if eqn.primitive.name == "jax.numpy.where"
        )
    )
    eqn.params = dict(eqn.params)
    if corruption == "missing_dtype":
        eqn.params.pop("source_dtype")
    elif corruption == "unknown_param":
        eqn.params["extra"] = True
    elif corruption == "dtype_string":
        eqn.params["source_dtype"] = "float32"
    elif corruption == "wrong_dtype":
        eqn.params["source_dtype"] = np.dtype(np.float16)
    elif corruption == "weak_int":
        eqn.params["source_weak_type"] = 0
    elif corruption == "wrong_weak":
        eqn.params["source_weak_type"] = True
    else:
        eqn.outvars = [copy(eqn.outvars[0])]
        eqn.outvars[0].aval = ShapedArray((2,), np.float16)
    with precision_policy_scope(PrecisionPolicy.SOURCE), pytest.raises(AssertionError):
        _ArrayCreationAudit._leaf(eqn)


@pytest.mark.parametrize("dtype", [np.bool_, jnp.bfloat16])
def test_runtime_evidence_separates_kernel_availability(
    dtype: Any,
    record_property: Callable[..., Any],
) -> None:
    def target(mask: Any, value: Any) -> Any:
        if dtype is np.bool_:
            return jnp.where(mask, True, False)
        low = value.astype(jnp.bfloat16)
        return (jnp.where(mask, low, 0.5) + jnp.full((2,), 0.25)).astype(jnp.float32)

    args = np.asarray([True, False]), np.asarray([0.125, 0.25], np.float32)
    if dtype is np.bool_:
        _, expected = _reference(target, args)
        proto = to_onnx(
            target, [_shape_spec(jnp.asarray(value)) for value in args], opset=23
        )
        onnx.checker.check_model(proto, full_check=True)
        assert "Where" in {node.op_type for node in proto.graph.node}
    else:
        proto, audit, _, expected = _export(target, args, cpu=False)
        assert any(
            output.aval.dtype == np.dtype(dtype)
            for eqn in audit.equations
            for output in eqn.outvars
        )
    for optimized in (False, True):
        key = f"cpu_{np.dtype(dtype).name}_{'all' if optimized else 'off'}"
        try:
            _cpu(proto, args, expected, optimized=optimized)
        except ort.capi.onnxruntime_pybind11_state.NotImplemented as exc:
            assert "Could not find an implementation" in str(exc)
            record_property(key, f"kernel-unavailable: {exc}")
        else:
            record_property(key, "exact-sample-match")


@pytest.mark.parametrize("component", ["where", "full"])
def test_creation_repairs_do_not_widen_product_admission(component: str) -> None:
    def target(value: Any) -> Any:
        if component == "where":
            return jnp.where(np.asarray([True, False]), value, 0.5)
        return jnp.full((2,), value)

    arguments = (np.asarray(0.25, np.float32),)
    _, expected = _reference(target, arguments)
    if component == "where":
        artifact = user_interface._to_onnx(
            target,
            [_shape_spec(jnp.asarray(arguments[0]))],
            precision_policy=PrecisionPolicy.SOURCE,
        )
        report = get_precision_report(artifact)
        assert report is not None and report.profile == "source-static-v1"
        _cpu(artifact, arguments, expected)
        assert active_precision_policy() is PrecisionPolicy.LEGACY
        return
    with pytest.raises(SourcePrecisionError) as caught:
        user_interface._to_onnx(
            target,
            [_shape_spec(jnp.asarray(arguments[0]))],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert caught.value.category == "profile-exclusion"
    assert caught.value.primitive == f"jax.numpy.{component}"
    assert active_precision_policy() is PrecisionPolicy.LEGACY
