# tests/extra_tests/converter/test_source_array_creation_sequences.py

"""Native sequence bindings with finite diagnostics and honest target evidence.

Typed/eager unpatched JAX references precede export. Correct graph typing does
not turn the retained exact linspace failures or BF16 target failures into
numerical acceptance. Data-dependent arange remains unfinished compatibility
work even though native JAX cannot trace it.
"""

from collections.abc import Callable, Iterator
from copy import copy
from enum import Enum
from types import SimpleNamespace
from typing import Any
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest

from jax2onnx import onnx_function, to_onnx, user_interface
from jax2onnx._compat.jax import DropVar, Literal, ShapedArray, scan_arity
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.plugins.plugin_system import _SourceFunctionTrace
from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _artifact as _product_artifact,
)

_ORIGINAL_ARANGE = jnp.arange
_ORIGINAL_LINSPACE = jnp.linspace
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
        for key, value in previous.items():
            jax.config.update(key, value)


def _body(value: Any) -> Any:
    inner = getattr(value, "jaxpr", value)
    return inner if inner is not value else value


def _typed(aval: Any) -> tuple[str, tuple[int, ...], bool]:
    dtype = np.dtype(aval.dtype).name
    shape = tuple(aval.shape)
    assert dtype in _DTYPES
    assert all(type(dim) is int and dim >= 0 for dim in shape)
    assert type(aval.weak_type) is bool
    return dtype, shape, aval.weak_type


def _signature(values: Any) -> tuple[Any, ...]:
    return tuple(_typed(var.aval) for var in values)


def _spec(value: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        value.shape, value.dtype, weak_type=bool(value.weak_type)
    )


def _placement(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (tuple, list)):
        return all(_placement(item) for item in value)
    if isinstance(value, jax.sharding.SingleDeviceSharding):
        return (
            value.device_set == {jax.devices("cpu")[0]}
            and value.memory_kind == "device"
        )
    return type(value).__name__ == "UnspecifiedValue" or (
        type(value).__name__ in {"Mesh", "AbstractMesh"} and bool(value.empty)
    )


def _parameter(value: Any) -> Any:
    if hasattr(value, "eqns") or hasattr(value, "jaxpr"):
        return _program(_body(value))
    if isinstance(value, (tuple, list)):
        return tuple(_parameter(item) for item in value)
    if isinstance(value, np.dtype):
        return value.name
    if isinstance(value, frozenset):
        assert not value
        return "frozenset", ()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        assert (
            type(value).__name__ == "Inline"
            and value.name == "AUTO"
            and value.value == "auto"
        )
        return "Inline", value.name, value.value
    assert _placement(value), type(value)
    return type(value).__name__


def _program(jaxpr: Any) -> Any:
    def value(var: Any) -> Any:
        payload = np.asarray(var.val).tobytes() if isinstance(var, Literal) else None
        return _typed(var.aval), payload

    return (
        _signature(jaxpr.invars),
        _signature(jaxpr.outvars),
        tuple(
            (
                eqn.primitive.name,
                tuple(map(value, eqn.invars)),
                _signature(eqn.outvars),
                tuple(
                    (key, _parameter(item)) for key, item in sorted(eqn.params.items())
                ),
            )
            for eqn in jaxpr.eqns
        ),
    )


class _SequenceAudit:
    """Only the explicit static native forms exercised in this module."""

    def __init__(self) -> None:
        self.roots: list[Any] = []
        self.equations: list[Any] = []

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
            self.equations.append(eqn)
            for var in eqn.invars:
                _typed(var.aval)
                assert isinstance(var, Literal) or id(var) in bound
            for var in eqn.outvars:
                _typed(var.aval)
                assert isinstance(var, DropVar) or id(var) not in bound
                bound.add(id(var))
            name, params = eqn.primitive.name, eqn.params
            if name in {"jit", "pjit"}:
                assert (
                    set(params)
                    <= {
                        "jaxpr",
                        "call_jaxpr",
                        "name",
                        "inline",
                        "keep_unused",
                        "donated_invars",
                        "compiler_options_kvs",
                    }
                    | _PLACEMENT
                )
                assert not params.get("compiler_options_kvs", ())
                inline = params["inline"]
                assert inline is False or (
                    isinstance(inline, Enum)
                    and _parameter(inline) == ("Inline", "AUTO", "auto")
                )
                assert all(
                    _placement(params[key]) for key in _PLACEMENT & params.keys()
                )
                child = _body(params.get("jaxpr", params.get("call_jaxpr")))
                assert child is not None
                assert _signature(child.invars) == _signature(eqn.invars)
                assert _signature(child.outvars) == _signature(eqn.outvars)
                self._walk(child, ancestors)
            elif name == "cond":
                assert set(params) == {"branches"} and len(params["branches"]) == 2
                assert _signature(eqn.invars[:1]) == (("int32", (), False),)
                for branch in params["branches"]:
                    child = _body(branch)
                    assert _signature(child.invars) == _signature(eqn.invars[1:])
                    assert _signature(child.outvars) == _signature(eqn.outvars)
                    self._walk(child, ancestors)
            elif name == "scan":
                assert set(params) <= {
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
                child = _body(params["jaxpr"])
                nc, nk, nx = scan_arity(params, len(child.invars))
                assert (
                    nx == 0 and type(params["length"]) is int and params["length"] >= 0
                )
                assert params["reverse"] is False
                assert _signature(child.invars) == _signature(eqn.invars)
                assert _signature(child.outvars) == _signature(eqn.outvars)
                assert _signature(eqn.outvars) == _signature(eqn.invars[nc : nc + nk])
                self._walk(child, ancestors)
            elif name.startswith("onnx_fn::"):
                assert set(params) == {"__jax2onnx_source_trace"}
                trace = params["__jax2onnx_source_trace"]
                assert isinstance(trace, _SourceFunctionTrace)
                assert _signature(eqn.outvars) == tuple(map(_typed, trace.output_avals))
                # The same callback checks the actual body during lowering.
            else:
                self._leaf(eqn)
        for var in jaxpr.outvars:
            assert isinstance(var, Literal) or id(var) in bound

    @staticmethod
    def _leaf(eqn: Any) -> None:
        name, params = eqn.primitive.name, eqn.params
        rules = {
            "iota": (0, {"dtype", "shape", "dimension", "sharding"}),
            "empty": (0, {"shape", "dtype", "sharding", "out_sharding"}),
            "convert_element_type": (1, {"new_dtype", "weak_type", "sharding"}),
            "broadcast_in_dim": (1, {"shape", "broadcast_dimensions", "sharding"}),
            "reshape": (1, {"new_sizes", "dimensions", "sharding"}),
            "concatenate": (None, {"dimension"}),
            "add": (2, set()),
            "sub": (2, set()),
            "mul": (2, {"out_dtype"}),
            "div": (2, set()),
            "floor": (1, set()),
            "lt": (2, set()),
            "sharding_constraint": (
                1,
                {"sharding", "layout", "context_mesh", "unconstrained_dims"},
            ),
        }
        assert name in rules, f"Outside native-sequence diagnostic: {name}"
        arity, allowed = rules[name]
        assert arity is None or len(eqn.invars) == arity
        assert len(eqn.outvars) == 1 and set(params) <= allowed
        assert all(
            _placement(params[key])
            for key in {"sharding", "out_sharding"} & params.keys()
        )
        assert params.get("out_dtype") is None
        ins = _signature(eqn.invars)
        out = _signature(eqn.outvars)[0]
        if name == "sharding_constraint":
            assert set(params) == {
                "sharding",
                "layout",
                "context_mesh",
                "unconstrained_dims",
            }
            assert isinstance(params["sharding"], jax.sharding.SingleDeviceSharding)
            assert params["layout"] is None and _placement(params["context_mesh"])
            assert (
                type(params["unconstrained_dims"]) is frozenset
                and not params["unconstrained_dims"]
            )
            assert out == ins[0]
        elif name in {"iota", "empty"}:
            if name == "empty":
                assert params.get("out_sharding") is None
            shape = params["shape"]
            assert isinstance(shape, tuple) and all(
                type(dim) is int and dim >= 0 for dim in shape
            )
            assert isinstance(params["dtype"], np.dtype)
            assert out == (params["dtype"].name, shape, False)
            if name == "iota":
                assert type(params["dimension"]) is int and 0 <= params[
                    "dimension"
                ] < len(shape)
        elif name == "convert_element_type":
            assert (
                isinstance(params["new_dtype"], np.dtype)
                and type(params["weak_type"]) is bool
            )
            assert out == (params["new_dtype"].name, ins[0][1], params["weak_type"])
        elif name == "broadcast_in_dim":
            shape, dimensions = params["shape"], params["broadcast_dimensions"]
            assert isinstance(shape, tuple) and all(
                type(dim) is int and dim >= 0 for dim in shape
            )
            assert isinstance(dimensions, tuple) and len(dimensions) == len(ins[0][1])
            assert tuple(sorted(set(dimensions))) == dimensions
            assert all(type(dim) is int and 0 <= dim < len(shape) for dim in dimensions)
            assert all(
                extent in (1, shape[dim]) for extent, dim in zip(ins[0][1], dimensions)
            )
            assert out == (ins[0][0], shape, ins[0][2])
        elif name == "reshape":
            assert params["dimensions"] is None
            assert isinstance(params["new_sizes"], tuple)
            assert all(type(dim) is int and dim >= 0 for dim in params["new_sizes"])
            assert np.prod(ins[0][1]) == np.prod(params["new_sizes"])
            assert out == (ins[0][0], params["new_sizes"], ins[0][2])
        elif name == "concatenate":
            axis = params["dimension"]
            assert ins and type(axis) is int and 0 <= axis < len(out[1])
            assert all(
                dtype == out[0] and len(shape) == len(out[1]) for dtype, shape, _ in ins
            )
            assert out[1][axis] == sum(shape[axis] for _, shape, _ in ins)
            assert all(
                all(
                    shape[dim] == out[1][dim]
                    for dim in range(len(shape))
                    if dim != axis
                )
                for _, shape, _ in ins
            )
        elif name == "floor":
            assert out == ins[0] and out[0] in {
                "float16",
                "bfloat16",
                "float32",
                "float64",
            }
        else:
            assert ins[0][0] == ins[1][0]
            assert out[0] == ("bool" if name == "lt" else ins[0][0])
            assert out[1] == tuple(np.broadcast_shapes(ins[0][1], ins[1][1]))
        inferred, effects = eqn.primitive.abstract_eval(
            *(var.aval for var in eqn.invars), **params
        )
        assert not effects and _typed(inferred) == out


def _reference(
    fn: Callable[..., Any], arguments: tuple[Any, ...]
) -> tuple[Any, list[np.ndarray]]:
    assert jnp.arange is _ORIGINAL_ARANGE and jnp.linspace is _ORIGINAL_LINSPACE
    values = tuple(jnp.asarray(value) for value in arguments)
    original = jax.make_jaxpr(fn)(*map(_spec, values)).jaxpr
    references = [np.asarray(value) for value in jax.tree.leaves(fn(*values))]
    return original, references


def _export(
    fn: Callable[..., Any],
    arguments: tuple[Any, ...] = (),
    *,
    same_program: bool = True,
) -> tuple[onnx.ModelProto, _SequenceAudit, list[np.ndarray]]:
    original, references = _reference(fn, arguments)
    audit = _SequenceAudit()

    def observe(jaxpr: Any, opset: int) -> None:
        if not audit.roots:
            assert _signature(jaxpr.invars) == _signature(original.invars)
            assert _signature(jaxpr.outvars) == _signature(original.outvars)
            if same_program:
                assert _program(jaxpr) == _program(original)
        audit(jaxpr, opset)

    artifact = user_interface._to_onnx(
        fn,
        [_spec(jnp.asarray(value)) for value in arguments],
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
        (value.dtype, value.shape, value.weak_type) for value in report.traced_outputs
    ) == _signature(original.outvars)
    assert len(proto.graph.output) == len(references)
    for output, reference in zip(proto.graph.output, references):
        assert (
            output.type.tensor_type.elem_type
            == onnx.helper.np_dtype_to_tensor_dtype(reference.dtype)
        )
        assert (
            tuple(dim.dim_value for dim in output.type.tensor_type.shape.dim)
            == reference.shape
        )
    assert not {"jax.numpy.arange", "jax.numpy.linspace"} & {
        eqn.primitive.name for eqn in audit.equations
    }
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    return proto, audit, references


def _samples(
    proto: onnx.ModelProto,
    arguments: tuple[Any, ...],
    references: list[np.ndarray],
    optimization: str = "disabled",
) -> Any:
    return validate_source_samples(
        proto,
        samples=[
            SourceSample(
                tuple(np.asarray(value) for value in arguments),
                references,
                SampleCriteria("bitwise"),
            )
        ],
        optimization=optimization,
    )


def _accepted(
    proto: onnx.ModelProto,
    arguments: tuple[Any, ...],
    references: list[np.ndarray],
    *,
    both: bool = False,
) -> None:
    for optimization in ("disabled", "all") if both else ("disabled",):
        report = _samples(proto, arguments, references, optimization)
        report.require_sample_acceptance()
        assert report.status is CheckStatus.VALIDATED


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "implicit_int",
        "implicit_float",
        "typed_i32",
        "typed_f32",
        "dtype_float16",
        "dtype_bfloat16",
        "dtype_float32",
        "dtype_float64",
        "descending",
        "empty",
    ],
)
def test_arange_preserves_native_dtype_and_arithmetic(x64: bool, kind: str) -> None:
    with jax.enable_x64(x64):
        if kind == "implicit_int":

            def fn():
                return jnp.arange(1, 5)

        elif kind == "implicit_float":

            def fn():
                return jnp.arange(0.125, 1.125, 0.125)

        elif kind == "typed_i32":

            def fn():
                return jnp.arange(np.int32(1), np.int32(5))

        elif kind == "typed_f32":

            def fn():
                return jnp.arange(
                    np.float32(0.125), np.float32(1.125), np.float32(0.125)
                )

        elif kind.startswith("dtype_"):
            dtype = np.dtype(kind.removeprefix("dtype_"))

            def fn():
                return jnp.arange(0.125, 1.125, 0.125, dtype=dtype)

        elif kind == "descending":

            def fn():
                return jnp.arange(5, -1, -2, dtype=np.int32)

        else:

            def fn():
                return jnp.arange(0, dtype=np.float32)

        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            proto, _, references = _export(fn)
        if kind == "dtype_float64" and not x64:
            assert references[0].dtype == np.float32
            assert any("truncated" in str(item.message) for item in emitted)
        if kind != "dtype_bfloat16":
            _accepted(proto, (), references)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("promotion", ["standard", "strict"])
@pytest.mark.parametrize("dtype,start", [(np.float16, 2049), (jnp.bfloat16, 257)])
def test_arange_omitted_step_has_its_own_low_precision_algorithm(
    x64: bool, promotion: str, dtype: Any, start: int
) -> None:
    with jax.enable_x64(x64), jax.numpy_dtype_promotion(promotion):

        def target() -> Any:
            return jnp.arange(start, start + 6, dtype=dtype), jnp.arange(
                start, start + 6, 1, dtype=dtype
            )

        proto, audit, references = _export(target)
        assert references[0].tobytes() != references[1].tobytes()
        iotas = [eqn for eqn in audit.roots[0].eqns if eqn.primitive.name == "iota"]
        assert [eqn.outvars[0].aval.dtype for eqn in iotas] == [
            np.dtype(dtype),
            np.dtype(np.float32),
        ]
        if dtype is np.float16:
            _accepted(proto, (), references, both=True)


@pytest.mark.parametrize(
    "kind",
    [
        "arange_keywords",
        "arange_mixed",
        "arange_none_step",
        "arange_positional_dtype",
        "arange_placement_none",
        "linspace_keywords",
        "linspace_positional_options",
        "linspace_device_none",
        "linspace_cpu_device",
        "linspace_retstep",
        "linspace_vector_axis",
    ],
)
def test_source_dispatch_preserves_original_argument_forms(kind: str) -> None:
    targets = {
        "arange_keywords": lambda: jnp.arange(start=1, stop=6, step=2, dtype=np.int32),
        "arange_mixed": lambda: jnp.arange(1, stop=6, step=2),
        "arange_none_step": lambda: jnp.arange(1, 5, None),
        "arange_positional_dtype": lambda: jnp.arange(1, 5, 1, np.int32),
        "arange_placement_none": lambda: jnp.arange(1, 5, out_sharding=None),
        "linspace_keywords": lambda: jnp.linspace(start=0, stop=1, num=3),
        "linspace_positional_options": lambda: jnp.linspace(
            0, 1, 3, True, False, np.float32, 0
        ),
        "linspace_device_none": lambda: jnp.linspace(0, 1, 3, device=None),
        "linspace_cpu_device": lambda: jnp.linspace(
            0, 1, 3, device=jax.devices("cpu")[0]
        ),
        "linspace_retstep": lambda: jnp.linspace(0, 1, 3, retstep=True),
        "linspace_vector_axis": lambda: jnp.linspace(
            np.asarray([0, 2], np.float32), np.asarray([1, 4], np.float32), 3, axis=-1
        ),
    }
    proto, _, references = _export(targets[kind])
    _accepted(proto, (), references, both=True)


def test_source_dispatch_preserves_native_invalid_placement_error() -> None:
    def target() -> Any:
        return jnp.arange(
            1, 5, out_sharding=jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
        )

    with pytest.raises(TypeError, match="NamedSharding"):
        _reference(target, ())
    with pytest.raises(TypeError, match="NamedSharding"):
        user_interface._to_onnx(
            target,
            [],
            precision_policy=PrecisionPolicy.SOURCE,
            diagnostic_jaxpr_validator=_SequenceAudit(),
        )
    assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("promotion", ["standard", "strict"])
@pytest.mark.parametrize(
    "kind",
    [
        "mixed_i32_f32",
        "weak_float16",
        "weak_bfloat16",
        "weak_float32",
        "all_weak",
        "integer_floor",
        "explicit_float64",
    ],
)
def test_linspace_resolves_native_promotion_before_lowering(
    x64: bool, promotion: str, kind: str
) -> None:
    with jax.enable_x64(x64), jax.numpy_dtype_promotion(promotion):
        if kind == "mixed_i32_f32":

            def fn(start: Any, stop: Any) -> Any:
                return jnp.linspace(start, stop, 5)

            arguments = np.asarray(0, np.int32), np.asarray(1, np.float32)
        elif kind.startswith("weak_"):
            dtype = np.dtype(kind.removeprefix("weak_"))

            def fn(start: Any) -> Any:
                return jnp.linspace(start, 1.125, 5)

            arguments = (np.asarray(0.125, dtype),)
        elif kind == "all_weak":
            fn, arguments = lambda: jnp.linspace(0.125, 1.125, 5), ()
        elif kind == "integer_floor":

            def fn(start: Any, stop: Any) -> Any:
                return jnp.linspace(start, stop, 4, dtype=np.int32)

            arguments = np.asarray(-1, np.int32), np.asarray(1, np.int32)
        else:
            fn, arguments = lambda: jnp.linspace(0, 1, 5, dtype=np.float64), ()
        if kind == "mixed_i32_f32" and promotion == "strict":
            with pytest.raises(jax.dtypes.TypePromotionError):
                _reference(fn, arguments)
            with pytest.raises(jax.dtypes.TypePromotionError):
                user_interface._to_onnx(
                    fn,
                    [_spec(jnp.asarray(value)) for value in arguments],
                    precision_policy=PrecisionPolicy.SOURCE,
                    diagnostic_jaxpr_validator=_SequenceAudit(),
                )
            assert active_precision_policy() is PrecisionPolicy.LEGACY
        else:
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                proto, audit, references = _export(fn, arguments)
            if kind == "integer_floor":
                np.testing.assert_array_equal(
                    references[0], np.asarray([-1, -1, 0, 1], np.int32), strict=True
                )
                assert "floor" in {eqn.primitive.name for eqn in audit.equations}
            if kind == "explicit_float64" and not x64:
                assert references[0].dtype == np.float32
                assert any("truncated" in str(item.message) for item in emitted)
            if kind != "weak_bfloat16":
                _accepted(proto, arguments, references)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(
    "num,stop",
    [(0, 1.0), (0, np.inf), (0, np.nan), (1, 1.0), (1, np.inf), (1, np.nan), (2, 1.0)],
)
@pytest.mark.parametrize("endpoint", [False, True])
def test_linspace_degenerate_counts_keep_native_endpoint_behavior(
    x64: bool, num: int, endpoint: bool, stop: float
) -> None:
    with jax.enable_x64(x64):

        def target(start: Any, end: Any) -> Any:
            return jnp.linspace(start, end, num, endpoint=endpoint)

        arguments = np.asarray(0.125, np.float32), np.asarray(stop, np.float32)
        proto, audit, references = _export(target, arguments)
        if num == 1:
            np.testing.assert_array_equal(
                references[0], np.asarray([0.125], np.float32), strict=True
            )
        if num == 0:
            assert references[0].shape == (0,)
            assert {eqn.primitive.name for eqn in audit.equations} & {
                "empty",
                "broadcast_in_dim",
            }
        _accepted(proto, arguments, references, both=True)


@onnx_function
def _named_sequences(start: Any, stop: Any) -> Any:
    return jnp.linspace(start, stop, 3) + jnp.arange(3, dtype=np.float32)


def _scoped_sequences(scope: str, start: Any, stop: Any, gate: Any) -> Any:
    if scope == "function":
        return _named_sequences(start, stop)
    if scope == "if":
        return jax.lax.cond(
            gate,
            lambda: _named_sequences(start, stop),
            lambda: _named_sequences(stop, start),
        )
    if scope == "loop":
        initial = jnp.broadcast_to(start, (3,))
        return jax.lax.fori_loop(
            0, 2, lambda _, carry: _named_sequences(start, stop), initial
        )
    return jnp.linspace(start, stop, 3) + jnp.arange(3, dtype=np.float32)


@pytest.mark.parametrize("scope", ["inline", "function", "if", "loop"])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_legacy_source_sequence_cache_orders_without_clearing(
    scope: str, reverse: bool
) -> None:
    def target(start: Any, stop: Any, gate: Any) -> Any:
        return _scoped_sequences(scope, start, stop, gate)

    fn = jax.jit(target)
    arguments = np.asarray(0, np.float32), np.asarray(1, np.float32), np.asarray(True)
    original, references = _reference(fn, arguments)
    order = (
        ("native", "source", "legacy", "native")
        if reverse
        else ("native", "legacy", "source", "native")
    )
    for stage in order:
        if stage == "native":
            traced, values = _reference(fn, arguments)
            assert _signature(traced.outvars) == _signature(original.outvars)
            for actual, expected in zip(values, references):
                assert (
                    actual.dtype == expected.dtype
                    and actual.shape == expected.shape
                    and actual.tobytes() == expected.tobytes()
                )
        elif stage == "legacy":
            model = to_onnx(fn, [_spec(jnp.asarray(value)) for value in arguments])
            onnx.checker.check_model(model, full_check=True)
        else:
            proto, _, expected = _export(fn, arguments, same_program=scope == "inline")
            _accepted(proto, arguments, expected, both=True)
        assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize("dtype,endpoint", [(np.float16, True), (np.float32, False)])
@pytest.mark.parametrize("x64", [False, True])
def test_known_exact_linspace_failures_remain_failed_acceptance(
    dtype: Any, endpoint: bool, x64: bool, record_property: Callable[..., Any]
) -> None:
    with jax.enable_x64(x64):
        start, stop, num = (-10, 3, 7) if dtype is np.float16 else (-1, 1, 5)

        def target(a: Any, b: Any) -> Any:
            return jnp.linspace(a, b, num, endpoint=endpoint)

        arguments = np.asarray(start, dtype), np.asarray(stop, dtype)
        proto, _, references = _export(target, arguments)
        compiled = np.asarray(jax.jit(target)(*arguments))
        assert compiled.tobytes() == references[0].tobytes()
        feed = {port.name: value for port, value in zip(proto.graph.input, arguments)}
        independent = ReferenceEvaluator(proto).run(None, feed)[0]
        assert (
            independent.dtype == references[0].dtype
            and independent.shape == references[0].shape
        )
        record_property(
            "onnx_reference_exact",
            (
                "validated"
                if independent.tobytes() == references[0].tobytes()
                else "failed"
            ),
        )
        for optimization in ("disabled", "all"):
            report = _samples(proto, arguments, references, optimization)
            if report.status is CheckStatus.VALIDATED:
                report.require_sample_acceptance()
                record_property(f"ort_{optimization}_exact", "validated")
                continue
            assert report.status is CheckStatus.FAILED
            assert any(
                check.category == "runtime-execution"
                and check.status is CheckStatus.VALIDATED
                for check in report.samples[0].checks
            )
            assert any(
                check.category == "numerical-acceptance"
                and check.status is CheckStatus.FAILED
                for check in report.samples[0].checks
            )
            with pytest.raises(SampleAcceptanceError, match="failed"):
                report.require_sample_acceptance()
            record_property(f"ort_{optimization}_exact", "failed")


@pytest.mark.parametrize("explicit", [False, True])
def test_bfloat16_arange_target_failures_are_not_numerical_acceptance(
    explicit: bool, record_property: Callable[..., Any]
) -> None:
    def target() -> Any:
        return (
            jnp.arange(257, 263, 1, dtype=jnp.bfloat16)
            if explicit
            else jnp.arange(257, 263, dtype=jnp.bfloat16)
        )

    proto, _, references = _export(target)
    for optimization in ("disabled", "all"):
        report = _samples(proto, (), references, optimization)
        if report.status is CheckStatus.VALIDATED:
            report.require_sample_acceptance()
            record_property(f"ort_{optimization}", "exact-sample-match")
            continue
        assert report.status is CheckStatus.FAILED
        failures = [
            check
            for check in report.samples[0].checks
            if check.status is CheckStatus.FAILED
        ]
        expected_category = "runtime-execution" if explicit else "runtime-load"
        assert any(check.category == expected_category for check in failures)
        assert any(
            check.category == "numerical-acceptance"
            and check.status is CheckStatus.UNVERIFIED
            for check in report.samples[0].checks
        )
        if explicit:
            assert any(
                "No corresponding Numpy type" in check.detail for check in failures
            )
        else:
            assert any(
                "Could not find an implementation" in check.detail
                and "Add" in check.detail
                for check in failures
            )
        with pytest.raises(SampleAcceptanceError, match="failed"):
            report.require_sample_acceptance()
        record_property(f"ort_{optimization}", f"failed:{expected_category}")


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
@pytest.mark.parametrize("opset", [23, 27])
def test_dynamic_arange_remains_an_unfinished_legacy_positive_capability(
    dtype: Any, opset: int, record_property: Callable[..., Any]
) -> None:
    def target(stop: Any) -> Any:
        return jnp.arange(stop, dtype=dtype)

    spec = jax.ShapeDtypeStruct((), np.float32)
    references = [np.asarray(target(np.float32(4)))]
    with pytest.raises(jax.errors.ConcretizationTypeError):
        _reference(target, (np.asarray(4, np.float32),))
    with pytest.raises(jax.errors.ConcretizationTypeError):
        user_interface._to_onnx(
            target,
            [spec],
            opset=23,
            precision_policy=PrecisionPolicy.SOURCE,
            diagnostic_jaxpr_validator=_SequenceAudit(),
        )
    model = to_onnx(target, [spec], opset=opset)
    onnx.checker.check_model(model, full_check=True)
    operations = [node.op_type for node in model.graph.node]
    assert "Range" in operations
    if dtype is jnp.bfloat16:
        assert (operations[-1] == "Cast") is (opset == 23)
    if dtype is np.float32 and opset == 23:
        _accepted(model, (np.asarray(4, np.float32),), references, both=True)
    else:
        for optimization in ("disabled", "all"):
            report = _samples(
                model, (np.asarray(4, np.float32),), references, optimization
            )
            if report.status is CheckStatus.VALIDATED:
                report.require_sample_acceptance()
                record_property(f"legacy_{optimization}", "exact-sample-match")
                continue
            assert report.status is CheckStatus.FAILED
            failures = [
                check
                for check in report.samples[0].checks
                if check.status is CheckStatus.FAILED
            ]
            if opset == 27:
                assert any(
                    check.category == "runtime-load"
                    and "Opset 27 is under development" in check.detail
                    for check in failures
                )
            else:
                assert any(
                    check.category == "runtime-execution"
                    and "No corresponding Numpy type" in check.detail
                    for check in failures
                )
            assert any(
                check.category == "numerical-acceptance"
                and check.status is CheckStatus.UNVERIFIED
                for check in report.samples[0].checks
            )
            with pytest.raises(SampleAcceptanceError, match="failed"):
                report.require_sample_acceptance()
            record_property(
                f"legacy_{optimization}",
                "failed:runtime-opset" if opset == 27 else "failed:runtime-interface",
            )
    record_property(
        "source_compatibility", "unfinished:native-data-dependent-arange-untraceable"
    )
    assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize("component", ["arange", "linspace"])
def test_native_sequence_binding_does_not_widen_product_admission(
    component: str,
) -> None:
    def target() -> Any:
        return (
            jnp.arange(3, dtype=np.float32)
            if component == "arange"
            else jnp.linspace(0, 1, 3)
        )

    def consumed() -> Any:
        value = target()
        return value, value + value

    for fn, output_count in ((target, 1), (consumed, 2)):
        _, proto, report, references = _product_artifact(fn, [], feeds=((),))
        dtype = references[0][0][0].dtype
        expected = np.asarray(
            [0, 1, 2] if component == "arange" else [0, 0.5, 1], dtype
        )
        expected_outputs = (
            (expected,) if output_count == 1 else (expected, expected + expected)
        )
        assert len(report.exported_outputs) == output_count
        assert all(
            value.dtype == dtype.name and value.shape == (3,)
            for value in report.traced_outputs
        )
        for outputs in references[0]:
            for actual, reference in zip(outputs, expected_outputs, strict=True):
                assert (
                    actual.dtype == reference.dtype and actual.shape == reference.shape
                )
                assert actual.tobytes() == reference.tobytes()
        _accepted(proto, (), list(expected_outputs), both=True)


@pytest.mark.parametrize(
    "corruption",
    ["out_dtype", "sharding", "dimension", "dtype", "unknown", "weak", "output_shape"],
)
def test_native_sequence_validator_rejects_malformed_forms(corruption: str) -> None:
    def target() -> Any:
        return jnp.arange(1, 5, 1, dtype=np.float16)

    _, audit, _ = _export(target)
    primitive = "mul" if corruption == "out_dtype" else "iota"
    eqn = copy(next(eqn for eqn in audit.equations if eqn.primitive.name == primitive))
    eqn.params = dict(eqn.params)
    if corruption == "out_dtype":
        eqn.params["out_dtype"] = np.dtype(np.float64)
    elif corruption == "sharding":
        eqn.params["sharding"] = object()
    elif corruption == "dimension":
        eqn.params["dimension"] = 4
    elif corruption == "dtype":
        eqn.params["dtype"] = np.dtype(np.float16)
    elif corruption == "unknown":
        eqn.params["unexpected"] = None
    else:
        eqn.outvars = [copy(eqn.outvars[0])]
        shape = (5,) if corruption == "output_shape" else (4,)
        eqn.outvars[0].aval = ShapedArray(
            shape, np.float32, weak_type=corruption == "weak"
        )
    with pytest.raises(AssertionError):
        _SequenceAudit._leaf(eqn)


@pytest.mark.parametrize("parameter", ["out_sharding", "out_dtype"])
def test_empty_diagnostic_does_not_admit_new_placement_or_output_forms(
    parameter: str,
) -> None:
    params = {"shape": (0,), "dtype": np.dtype(np.float32), parameter: object()}
    eqn = SimpleNamespace(
        primitive=SimpleNamespace(name="empty"),
        params=params,
        invars=[],
        outvars=[SimpleNamespace(aval=ShapedArray((0,), np.float32))],
    )
    with pytest.raises(AssertionError):
        _SequenceAudit._leaf(eqn)
