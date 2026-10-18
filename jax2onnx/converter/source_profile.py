# jax2onnx/converter/source_profile.py

"""Pure Jaxpr admission checks for the internal ``source-static-v1`` milestone.

The persisted identifier includes finite rank-two/rank-three leading-batch
Add/Mul/Exp and canonical MatMul, plus static native Where, typed plain-JIT
carriers, the static bias reshape, actual named Function bodies and returned
aliases. Other symbolic rules remain separate work.

These checks describe an export representation, not runtime numerical evidence.
The top-level facade owns typed floating and static Boolean input normalization.
Recursive callbacks may receive Boolean comparisons and uint32 bit representations.
Function bodies are checked when their lowering retraces them with this callback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, NoReturn

import jax
import numpy as np

from jax2onnx._compat.jax import (
    AbstractValue,
    ClosedJaxpr,
    DropVar,
    Literal as JaxLiteral,
    is_empty_jit_mesh,
    is_unspecified_jit_sharding,
    native_jit_boolean_inline_hints,
)
from jax2onnx.converter.source_representation import (
    source_shape_contract,
    validate_source_shapes,
)


SOURCE_STATIC_PROFILE: Final[str] = "source-static-v1"
SourcePrecisionCategory = Literal[
    "profile-exclusion",
    "representation-limit",
    "lowering-defect",
    "configuration-conflict",
]
_FLOATS: Final[frozenset[str]] = frozenset(
    {"float16", "bfloat16", "float32", "float64"}
)
_VALUE_DTYPES: Final[frozenset[str]] = _FLOATS | {"bool", "uint32", "int32"}
_PLACEMENT_PARAMS: Final[frozenset[str]] = frozenset({"sharding", "out_sharding"})


class SourcePrecisionError(ValueError):
    """A source precision failure with inspectable, artifact-independent fields."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        category: SourcePrecisionCategory,
        location: str,
        primitive: str | None = None,
        requested_dtypes: Sequence[str] = (),
        precision_params: Mapping[str, str] | None = None,
        opset: int | None = None,
        requirement: str,
        action: str,
    ) -> None:
        self.detail = message
        self.phase = phase
        self.category = category
        self.location = location
        self.primitive = primitive
        self.requested_dtypes = tuple(requested_dtypes)
        self.precision_params = dict(precision_params or {})
        self.opset = opset
        self.requirement = requirement
        self.action = action
        operation = f" ({primitive})" if primitive else ""
        super().__init__(
            f"{SOURCE_STATIC_PROFILE}: {location}{operation}: {message}. "
            f"Required: {requirement}. Action: {action}"
        )

    def with_context(self, location: str) -> SourcePrecisionError:
        """Prefix a branch/function location without discarding failure details."""
        return SourcePrecisionError(
            self.detail,
            phase=self.phase,
            category=self.category,
            location=f"{location}/{self.location}",
            primitive=self.primitive,
            requested_dtypes=self.requested_dtypes,
            precision_params=self.precision_params,
            opset=self.opset,
            requirement=self.requirement,
            action=self.action,
        )


def validate_source_input_dtype(dtype: Any, *, index: int, opset: int) -> None:
    """Reject declared dtypes that JAX would change under the caller's settings."""
    declared = np.dtype(dtype)
    canonical = np.dtype(jax.dtypes.canonicalize_dtype(declared))
    if canonical != declared:
        configuration = (
            "JAX x64 is disabled"
            if not jax.config.x64_enabled
            else "the caller's JAX configuration changes the dtype"
        )
        raise SourcePrecisionError(
            f"input requests {declared.name}, but {configuration}; JAX would "
            f"canonicalize it to {canonical.name}",
            phase="configuration",
            category="configuration-conflict",
            location=f"input {index}",
            requested_dtypes=(declared.name,),
            opset=opset,
            requirement="the declared input dtype must survive JAX canonicalization unchanged",
            action="configure JAX x64 before preparing 64-bit inputs, or explicitly choose the canonical dtype",
        )


@dataclass(frozen=True)
class _Site:
    location: str
    opset: int
    primitive: str | None = None
    requested_dtypes: tuple[str, ...] = ()
    precision_params: Mapping[str, str] | None = None

    def fail(
        self,
        message: str,
        requirement: str,
        action: str = "use operations within this audited profile or add audited support for the requested operation",
        *,
        category: SourcePrecisionCategory = "profile-exclusion",
    ) -> NoReturn:
        raise SourcePrecisionError(
            message,
            phase=(
                "representation"
                if category == "representation-limit"
                else "profile-validation"
            ),
            category=category,
            location=self.location,
            primitive=self.primitive,
            requested_dtypes=self.requested_dtypes,
            precision_params=self.precision_params,
            opset=self.opset,
            requirement=requirement,
            action=action,
        )

    def require(
        self,
        condition: bool,
        message: str,
        requirement: str,
        *,
        category: SourcePrecisionCategory = "profile-exclusion",
    ) -> None:
        if not condition:
            self.fail(message, requirement, category=category)


@dataclass(frozen=True)
class _Tensor:
    dtype: str
    shape: tuple[int | str, ...]
    symbol: tuple[str, object | None] | None = None


def _tensor(var: Any, site: _Site) -> _Tensor:
    site.require(
        not isinstance(var.aval, AbstractValue)
        or isinstance(var.aval, jax.core.ShapedArray),
        "non-tensor or reference aval is outside the profile",
        "ordinary ShapedArray values without refs, tokens, or high-level types",
    )
    try:
        dtype = np.dtype(var.aval.dtype).name
        shape = tuple(var.aval.shape)
        weak_type = var.aval.weak_type
    except (AttributeError, TypeError, ValueError) as exc:
        site.fail(
            f"invalid traced tensor metadata: {exc}",
            "a tensor aval with dtype and shape",
            "report this tracing/lowering defect with a minimal reproduction",
            category="lowering-defect",
        )
    site.require(
        dtype in _VALUE_DTYPES,
        f"dtype {dtype} is outside the profile",
        "FP16/BF16/FP32/FP64 values, Boolean comparisons, or uint32 bits",
    )
    site.require(
        type(weak_type) is bool,
        "invalid traced weakness metadata",
        "a native Boolean weak_type flag",
        category="lowering-defect",
    )
    try:
        normalized, symbol = source_shape_contract(shape)
    except ValueError as exc:
        site.fail(f"shape {shape} is outside the profile", str(exc))
    site.require(
        symbol is None or symbol[1] is not None,
        "traced symbolic dimension has no verified native scope",
        "an actual native JAX symbolic batch variable",
    )
    site.require(
        symbol is None or dtype in _FLOATS,
        "symbolic nonfloating values are outside this audited batch packet",
        "FP16/BF16/FP32/FP64 symbolic tensors",
    )
    return _Tensor(dtype, normalized, symbol)


def _arity(eqn: Any, inputs: int, outputs: int, site: _Site) -> None:
    site.require(
        len(eqn.invars) == inputs and len(eqn.outvars) == outputs,
        "malformed primitive operand/result count",
        f"{inputs} operands and {outputs} results",
        category="lowering-defect",
    )


def _params(
    eqn: Any, names: set[str], site: _Site, *, required: set[str] | None = None
) -> None:
    missing = (required or set()) - set(eqn.params)
    site.require(
        not missing,
        f"missing primitive parameters {sorted(missing)}",
        "the attributes required by the traced primitive",
        category="lowering-defect",
    )
    unknown = set(eqn.params) - names - _PLACEMENT_PARAMS
    site.require(
        not unknown,
        f"unaudited parameters {sorted(unknown)}",
        "only the profile's audited primitive parameters",
    )


def _same_shape(
    actual: tuple[int | str, ...], expected: tuple[int | str, ...], site: _Site
) -> None:
    site.require(
        actual == expected,
        f"result shape {actual} differs from {expected}",
        "the shape implied by the primitive and its operands",
        category="lowering-defect",
    )


def _float_values(values: Sequence[_Tensor], site: _Site) -> None:
    site.require(
        all(value.dtype in _FLOATS for value in values),
        "nonfloating arithmetic or conversion is outside the profile",
        "floating arithmetic and explicit floating conversions",
    )


def _validate_exp(
    eqn: Any, inputs: list[_Tensor], outputs: list[_Tensor], site: _Site
) -> None:
    _arity(eqn, 1, 1, site)
    _params(eqn, {"accuracy"}, site)
    site.require(
        eqn.params.get("accuracy") is None
        and all(eqn.params.get(name) is None for name in _PLACEMENT_PARAMS),
        "exponential accuracy or placement is outside the profile",
        "native Exp with absent/None accuracy and unspecified placement",
    )
    source, target = inputs[0], outputs[0]
    _float_values((source,), site)
    site.require(
        target.dtype == source.dtype,
        "exponential result dtype differs from its operand",
        "the original floating dtype",
        category="lowering-defect",
    )
    site.require(
        bool(eqn.outvars[0].aval.weak_type) == bool(eqn.invars[0].aval.weak_type),
        "exponential result weakness differs from its native operand",
        "the native Exp weak-type rule without rebuilding avals",
        category="lowering-defect",
    )
    _same_shape(target.shape, source.shape, site)
    site.require(
        target.symbol == source.symbol,
        "exponential result scope differs from its operand",
        "the original verified native leading-batch scope",
        category="lowering-defect",
    )


def _validate_ordinary_dot(
    eqn: Any, inputs: list[_Tensor], outputs: list[_Tensor], site: _Site
) -> None:
    lhs, rhs = inputs
    rank = len(lhs.shape)
    site.require(
        rank in (2, 3, 4)
        and len(rhs.shape) == rank
        and all(
            isinstance(dim, (int, np.integer))
            and not isinstance(dim, (bool, np.bool_))
            and dim > 0
            for dim in (*lhs.shape, *rhs.shape)
        ),
        "ordinary matrix dimensions are outside the profile",
        "equal positive static operand ranks 2, 3, or 4",
    )
    dimensions = eqn.params["dimension_numbers"]
    site.require(
        isinstance(dimensions, (tuple, list))
        and len(dimensions) == 2
        and all(
            isinstance(pair, (tuple, list)) and len(pair) == 2 for pair in dimensions
        )
        and all(
            isinstance(axes, (tuple, list)) for pair in dimensions for axes in pair
        ),
        "malformed matrix dimension_numbers",
        "ordered contracting and batch axis pairs",
        category="lowering-defect",
    )
    (lc, rc), (lb, rb) = (tuple(tuple(axes) for axes in pair) for pair in dimensions)
    for contracting, batch in ((lc, lb), (rc, rb)):
        axes = contracting + batch
        site.require(
            all(
                isinstance(axis, (int, np.integer))
                and not isinstance(axis, (bool, np.bool_))
                and 0 <= axis < rank
                for axis in axes
            )
            and len(set(axes)) == len(axes),
            "invalid or overlapping matrix axes",
            "distinct non-Boolean axes in the operand rank",
            category="lowering-defect",
        )
    site.require(
        len(lc) == len(rc) and len(lb) == len(rb),
        "inconsistent matrix axis pair counts",
        "equal numbers of paired contracting and batch axes",
        category="lowering-defect",
    )
    site.require(
        len(lc) == 1 and len(lb) == rank - 2,
        "ordinary matrix axis cardinality is outside the profile",
        "one contracting and one free axis per operand, with rank minus two paired batch axes",
    )
    site.require(
        lhs.shape[lc[0]] == rhs.shape[rc[0]]
        and all(
            lhs.shape[left] == rhs.shape[right]
            for left, right in zip(lb, rb, strict=True)
        ),
        "inconsistent matrix contracting/batch dimensions",
        "equal paired contracting and batch sizes",
        category="lowering-defect",
    )
    lf = next(axis for axis in range(rank) if axis not in lc + lb)
    rf = next(axis for axis in range(rank) if axis not in rc + rb)
    site.require(
        outputs[0].dtype == lhs.dtype,
        "ordinary matrix result dtype differs from its operands",
        "the common operand dtype as the result dtype",
        category="lowering-defect",
    )
    _same_shape(
        outputs[0].shape,
        tuple(lhs.shape[axis] for axis in lb) + (lhs.shape[lf], rhs.shape[rf]),
        site,
    )


def _validate_dot(
    eqn: Any, inputs: list[_Tensor], outputs: list[_Tensor], site: _Site
) -> None:
    _arity(eqn, 2, 1, site)
    _params(
        eqn,
        {"dimension_numbers", "precision", "preferred_element_type"},
        site,
        required={"dimension_numbers"},
    )
    site.require(
        not set(eqn.params)
        - {"dimension_numbers", "precision", "preferred_element_type", "out_sharding"}
        and eqn.params.get("out_sharding") is None,
        "unaudited matrix parameters or placement",
        "only dimension_numbers, default precision, preferred_element_type and absent/None out_sharding",
    )
    default = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    precision = eqn.params.get("precision")
    site.require(
        precision is None or precision == default,
        "matrix precision/algorithm is outside the profile",
        "default precision without HIGH, HIGHEST, or an explicit algorithm",
    )
    preferred = eqn.params.get("preferred_element_type")
    if inputs[0].dtype == inputs[1].dtype and inputs[0].dtype in {"float32", "float64"}:
        site.require(
            preferred is None or np.dtype(preferred) == np.dtype(inputs[0].dtype),
            "ordinary matrix preferred result type is outside the profile",
            "absent or same-dtype preferred_element_type",
        )
        if any(value.symbol is not None for value in (*inputs, *outputs)):
            _validate_leading_batch_dot(eqn, inputs, outputs, site)
            return
        _validate_ordinary_dot(eqn, inputs, outputs, site)
        return
    site.require(
        tuple(value.dtype for value in (*inputs, *outputs))
        == ("float16", "float16", "float32")
        and preferred is not None
        and np.dtype(preferred) == np.dtype(np.float32),
        "matrix operand/result types are outside the profile",
        "FP16 operands with an explicitly preferred FP32 result",
    )
    lhs, rhs = inputs
    rank = len(lhs.shape)
    batch = tuple(range(rank - 2))
    dimensions = (((rank - 1,), (rank - 2,)), (batch, batch))
    site.require(
        rank in (2, 3)
        and len(rhs.shape) == rank
        and all(isinstance(dim, int) and dim > 0 for dim in (*lhs.shape, *rhs.shape))
        and eqn.params.get("dimension_numbers") == dimensions,
        "matrix dimensions are outside the profile",
        "positive static rank-2 matrices or rank-3 matrices with batch axis 0 and standard contracting axes (-1, -2)",
    )
    site.require(
        lhs.shape[-1] == rhs.shape[-2] and lhs.shape[:-2] == rhs.shape[:-2],
        "inconsistent matrix contracting/batch dimensions",
        "equal contracting and batch sizes",
        category="lowering-defect",
    )
    _same_shape(outputs[0].shape, lhs.shape[:-1] + (rhs.shape[-1],), site)


def _validate_leading_batch_dot(
    eqn: Any, inputs: list[_Tensor], outputs: list[_Tensor], site: _Site
) -> None:
    lhs, rhs = inputs
    result = outputs[0]
    free_batch = (
        len(lhs.shape) == len(rhs.shape) == 2
        and lhs.symbol is not None
        and rhs.symbol is None
    )
    paired_batch = (
        len(lhs.shape) == len(rhs.shape) == 3
        and lhs.symbol is not None
        and rhs.symbol == lhs.symbol
    )
    site.require(
        free_batch or paired_batch,
        "symbolic matrix dimensions are outside the profile",
        "(B,K)@(K,N) or (B,M,K)@(B,K,N) with one shared native batch scope",
    )
    dimensions = (
        (((1,), (0,)), ((), ())) if free_batch else (((2,), (1,)), ((0,), (0,)))
    )
    actual_dimensions = eqn.params["dimension_numbers"]
    site.require(
        isinstance(actual_dimensions, (tuple, list))
        and len(actual_dimensions) == 2
        and all(
            isinstance(pair, (tuple, list))
            and len(pair) == 2
            and all(
                isinstance(axes, (tuple, list))
                and all(
                    isinstance(axis, (int, np.integer))
                    and not isinstance(axis, (bool, np.bool_))
                    for axis in axes
                )
                for axes in pair
            )
            for pair in actual_dimensions
        ),
        "malformed symbolic matrix axes",
        "non-Boolean integer contracting and batch axes",
        category="lowering-defect",
    )
    site.require(
        actual_dimensions == dimensions,
        "symbolic matrix axes are outside the profile",
        "canonical final contracting axes and only the declared leading batch axis",
    )
    site.require(
        all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
            for dim in (*lhs.shape[1:], *(rhs.shape if free_batch else rhs.shape[1:]))
        ),
        "symbolic matrix static dimensions are outside the profile",
        "positive static M/K/N extents",
    )
    site.require(
        lhs.shape[-1] == rhs.shape[-2],
        "inconsistent symbolic matrix contracting dimensions",
        "equal positive static contracting sizes",
        category="lowering-defect",
    )
    site.require(
        result.symbol == lhs.symbol and result.dtype == lhs.dtype,
        "symbolic matrix result scope or dtype differs from its operands",
        "the same native batch scope and common FP32/FP64 dtype",
        category="lowering-defect",
    )
    site.require(
        bool(eqn.outvars[0].aval.weak_type)
        == all(bool(var.aval.weak_type) for var in eqn.invars),
        "symbolic matrix result weakness differs from its native operands",
        "the native dot_general weak-type rule without rebuilding avals",
        category="lowering-defect",
    )
    _same_shape(result.shape, lhs.shape[:-1] + (rhs.shape[-1],), site)


def _validate_bias_reshape(
    eqn: Any, inputs: list[_Tensor], outputs: list[_Tensor], site: _Site
) -> None:
    _arity(eqn, 1, 1, site)
    _params(
        eqn,
        {"new_sizes", "dimensions"},
        site,
        required={"new_sizes", "dimensions"},
    )
    site.require(
        not set(eqn.params) - {"new_sizes", "dimensions", "sharding", "out_sharding"}
        and eqn.params.get("sharding") is None
        and eqn.params.get("out_sharding") is None
        and eqn.params["dimensions"] is None,
        "reshape attributes or placement are outside the profile",
        "only a static unpermuted bias reshape with unspecified placement",
    )
    source, target = inputs[0], outputs[0]
    _float_values((source, target), site)
    site.require(
        len(source.shape) == 1
        and isinstance(source.shape[0], int)
        and source.shape[0] > 0,
        "reshape dimensions are outside the profile",
        "a positive static floating bias [N] reshaped to [1,N]",
    )
    expected = (1, source.shape[0])
    site.require(
        eqn.params["new_sizes"] == expected
        and all(type(dim) is int for dim in eqn.params["new_sizes"]),
        "reshape sizes are outside the profile",
        "resolved new_sizes=(1,N) with positive static N",
    )
    site.require(
        source.dtype == target.dtype
        and bool(eqn.invars[0].aval.weak_type) == bool(eqn.outvars[0].aval.weak_type),
        "bias reshape result dtype or weakness differs from its operand",
        "the original floating dtype and native weakness",
        category="lowering-defect",
    )
    _same_shape(target.shape, expected, site)


def _signature(variables: Sequence[Any], site: _Site) -> tuple[Any, ...]:
    return tuple((_tensor(var, site), var.aval.weak_type) for var in variables)


def _validate_select(
    eqn: Any, inputs: list[_Tensor], outputs: list[_Tensor], site: _Site
) -> None:
    _arity(eqn, 3, 1, site)
    site.require(
        not eqn.params,
        "selection has unaudited parameters",
        "native select_n without parameters or placement",
    )
    condition, false_arm, true_arm = inputs
    result = outputs[0]
    site.require(
        condition.dtype == "bool" and false_arm.dtype in _FLOATS | {"bool"},
        "selection operand types are outside the profile",
        "one Boolean selector and two floating or Boolean cases",
    )
    site.require(
        false_arm.dtype == true_arm.dtype == result.dtype
        and false_arm.shape == true_arm.shape == result.shape
        and false_arm.symbol == true_arm.symbol == result.symbol
        and condition.shape in ((), result.shape)
        and bool(eqn.outvars[0].aval.weak_type)
        == all(bool(var.aval.weak_type) for var in eqn.invars[1:]),
        "selection signatures disagree with the native two-case rule",
        "equal case/result dtypes and shapes, scalar or same-shaped selector, and conjunction of case weakness",
        category="lowering-defect",
    )
    if result.symbol is not None:
        site.require(
            condition.shape == ()
            and condition.symbol is None
            and result.dtype in _FLOATS,
            "symbolic selection is outside the bounded selector family",
            "a static scalar Boolean and equal native leading-batch floating arms",
        )


_JIT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "jaxpr",
        "in_shardings",
        "out_shardings",
        "in_layouts",
        "out_layouts",
        "donated_invars",
        "ctx_mesh",
        "name",
        "keep_unused",
        "inline",
        "compiler_options_kvs",
    }
)


def _validate_jit(eqn: Any, site: _Site, ancestors: frozenset[int]) -> None:
    """Prove a native plain-JIT carrier before the existing lowerer binds it."""
    site.require(
        not set(eqn.params) - _JIT_FIELDS,
        "JIT has unaudited parameters",
        "the native eleven-field plain-JIT schema without lazy or alternative body fields",
    )
    site.require(
        set(eqn.params) == _JIT_FIELDS,
        "JIT is missing native parameters",
        "the complete native plain-JIT schema",
        category="lowering-defect",
    )
    params = eqn.params
    closed = params["jaxpr"]
    site.require(
        isinstance(closed, ClosedJaxpr),
        "JIT body is not a concrete closed Jaxpr",
        "an actual native ClosedJaxpr, never a thunk or an open body",
        category="lowering-defect",
    )
    body = closed.jaxpr  # JAX 0.11 returns self; never repeatedly unwrap it.
    site.require(
        not getattr(body, "is_high", False),
        "high-level JIT bodies are outside the profile",
        "a low-level pure tensor Jaxpr",
    )
    for key, count in (("in", len(eqn.invars)), ("out", len(eqn.outvars))):
        for suffix in ("shardings", "layouts"):
            entries = params[f"{key}_{suffix}"]
            site.require(
                type(entries) is tuple and len(entries) == count,
                f"JIT {key}_{suffix} has malformed type or arity",
                "a native tuple with one entry per port",
                category="lowering-defect",
            )
            site.require(
                all(
                    (
                        is_unspecified_jit_sharding(value)
                        if suffix == "shardings"
                        else value is None
                    )
                    for value in entries
                ),
                f"JIT {key}_{suffix} requests explicit placement",
                "actual UNSPECIFIED sharding sentinels and None layouts",
            )
    donated = params["donated_invars"]
    site.require(
        type(donated) is tuple
        and len(donated) == len(eqn.invars)
        and all(type(value) is bool for value in donated),
        "JIT donation metadata has malformed type or arity",
        "one native Boolean donation flag per input",
        category="lowering-defect",
    )
    site.require(not any(donated), "JIT requests donation", "no donated inputs")
    site.require(
        isinstance(params["name"], str)
        and any(
            params["inline"] is value for value in native_jit_boolean_inline_hints()
        )
        and type(params["keep_unused"]) is bool,
        "JIT compilation hints have malformed types",
        "a string name, native False/True inline representation, and Boolean keep_unused",
        category="lowering-defect",
    )
    site.require(
        type(params["compiler_options_kvs"]) is tuple,
        "JIT compiler options have a malformed type",
        "the native tuple of compiler options",
        category="lowering-defect",
    )
    site.require(
        not params["compiler_options_kvs"] and is_empty_jit_mesh(params["ctx_mesh"]),
        "JIT compiler options or mesh are outside the profile",
        "empty compiler options and an actual empty concrete JAX Mesh",
    )
    site.require(
        all(
            isinstance(var.aval, jax.core.ShapedArray)
            for var in (*body.invars, *body.constvars, *body.outvars)
        ),
        "JIT has non-tensor or reference boundary values",
        "ordinary ShapedArray boundaries without refs/tokens",
    )
    site.require(
        _signature(eqn.invars, site) == _signature(body.invars, site)
        and _signature(eqn.outvars, site) == _signature(body.outvars, site),
        "JIT outer/body signatures disagree",
        "exact input/output counts, dtype, shape, weakness and native symbolic scope",
        category="lowering-defect",
    )
    consts = closed.consts
    site.require(
        len(consts) == len(body.constvars),
        "JIT constant payload count disagrees with its bindings",
        "one unchanged payload per constant variable",
        category="lowering-defect",
    )
    for index, (var, value) in enumerate(zip(body.constvars, consts, strict=True)):
        site.require(
            isinstance(
                value, (np.ndarray, np.generic, jax.Array, bool, int, float, complex)
            ),
            f"JIT constant {index} is not an array or scalar payload",
            "a concrete numeric payload without conversion hooks",
            category="lowering-defect",
        )
        array = np.asarray(
            value
        )  # Never use jnp.asarray: ambient x64 must not recanonicalize it.
        site.require(
            array.dtype == np.dtype(var.aval.dtype)
            and array.shape == tuple(var.aval.shape),
            f"JIT constant {index} payload dtype or shape disagrees with its aval",
            "exact payload dtype/shape while retaining declared weakness and bytes",
            category="lowering-defect",
        )
    try:
        _validate_scope(body, site.opset, ancestors)
    except SourcePrecisionError as exc:
        raise exc.with_context(f"{site.location}/jit body") from exc


def _validate_equation(
    eqn: Any,
    site: _Site,
    consumers: Mapping[int, list[tuple[str, int]]],
    graph_outputs: set[int],
    formal_symbols: set[tuple[str, object | None]],
    ancestors: frozenset[int],
) -> None:
    inputs = [_tensor(var, site) for var in eqn.invars]
    outputs = [_tensor(var, site) for var in eqn.outvars]
    site.require(
        all(
            value.symbol is None or value.symbol in formal_symbols
            for value in (*inputs, *outputs)
        ),
        "symbolic value differs from this scope's formal batch contract",
        "the same atomic batch variable and scope as actual formals",
    )
    name = site.primitive
    symbolic = any(value.symbol is not None for value in (*inputs, *outputs))
    site.require(
        not symbolic
        or name in {"add", "mul", "exp", "dot_general", "jit", "select_n"}
        or bool(name and name.startswith("onnx_fn::")),
        "symbolic operation is outside this audited batch packet",
        "shared leading-batch Add/Mul/Exp/MatMul, named Function signatures, and returned aliases",
    )
    if name == "jit":
        _validate_jit(eqn, site, ancestors)
    elif name == "select_n":
        _validate_select(eqn, inputs, outputs, site)
    elif name == "ne":
        _arity(eqn, 2, 1, site)
        _params(eqn, set(), site)
        _float_values(inputs, site)
        site.require(
            not eqn.params
            and inputs[0].dtype == inputs[1].dtype
            and inputs[1].shape == ()
            and isinstance(eqn.invars[1], JaxLiteral)
            and eqn.invars[1].val == 0,
            "numeric condition is outside the native Where family",
            "a static floating condition compared with a same-dtype scalar zero literal",
        )
        site.require(
            outputs[0].dtype == "bool"
            and bool(eqn.outvars[0].aval.weak_type)
            == all(bool(var.aval.weak_type) for var in eqn.invars),
            "numeric condition result dtype or weakness differs from native Ne",
            "the native Boolean comparison result",
            category="lowering-defect",
        )
        _same_shape(outputs[0].shape, inputs[0].shape, site)
    elif name == "exp":
        _validate_exp(eqn, inputs, outputs, site)
    elif name in {"add", "sub", "mul", "div", "neg", "sqrt", "jax.numpy.sqrt", "gt"}:
        unary = name in {"neg", "sqrt", "jax.numpy.sqrt"}
        _arity(eqn, 1 if unary else 2, 1, site)
        allowed_params = (
            {"out_dtype"}
            if name == "mul"
            else {"accuracy"} if name == "sqrt" else set()
        )
        _params(eqn, allowed_params, site)
        site.require(
            eqn.params.get("out_dtype") is None and eqn.params.get("accuracy") is None,
            "explicit arithmetic output dtype/accuracy is outside the profile",
            "ordinary floating arithmetic; select wider work with explicit conversions",
        )
        if name == "mul":
            site.require(
                all(eqn.params.get(param) is None for param in _PLACEMENT_PARAMS),
                "multiplication placement is outside the profile",
                "native floating Mul with unspecified placement",
            )
        _float_values(inputs, site)
        site.require(
            len({value.dtype for value in inputs}) == 1,
            "inconsistent arithmetic operand types",
            "equal operand dtypes",
            category="lowering-defect",
        )
        expected_dtype = "bool" if name == "gt" else inputs[0].dtype
        site.require(
            outputs[0].dtype == expected_dtype,
            "inconsistent arithmetic result dtype",
            f"a {expected_dtype} result",
            category="lowering-defect",
        )
        if symbolic:
            operation = "Add" if name == "add" else "Mul"
            symbolic_inputs = [value for value in inputs if value.symbol is not None]
            site.require(
                bool(symbolic_inputs),
                f"symbolic {operation} operands do not provide a native batch variable",
                "at least one operand with the declared native B",
                category="lowering-defect",
            )
            batch = symbolic_inputs[0]
            site.require(
                all(
                    value.shape == batch.shape
                    and value.symbol == batch.symbol
                    or value.symbol is None
                    and value.shape == (1, *batch.shape[1:])
                    for value in inputs
                )
                and outputs[0].symbol == batch.symbol,
                f"symbolic {operation} operands do not share the same leading-batch shape",
                "equal rank-two/rank-three tails with the same native B or a literal leading extent 1",
            )
            site.require(
                bool(eqn.outvars[0].aval.weak_type)
                == all(bool(var.aval.weak_type) for var in eqn.invars),
                f"symbolic {operation} result weakness differs from its native operands",
                "the native floating pointwise weak-type rule",
                category="lowering-defect",
            )
            shape = batch.shape
        else:
            try:
                shape = tuple(np.broadcast_shapes(*(value.shape for value in inputs)))
            except ValueError:
                site.fail(
                    "incompatible arithmetic operand shapes",
                    "broadcast-compatible shapes",
                    category="lowering-defect",
                )
        _same_shape(outputs[0].shape, shape, site)
    elif name == "convert_element_type":
        _arity(eqn, 1, 1, site)
        _params(eqn, {"new_dtype", "weak_type"}, site, required={"new_dtype"})
        source, target = inputs[0], outputs[0]
        site.require(
            np.dtype(eqn.params.get("new_dtype")) == np.dtype(target.dtype),
            "cast attribute disagrees with its result dtype",
            "new_dtype matching the traced result",
            category="lowering-defect",
        )
        if "weak_type" in eqn.params:
            site.require(
                type(eqn.params["weak_type"]) is bool
                and bool(eqn.outvars[0].aval.weak_type) == eqn.params["weak_type"],
                "cast weak_type disagrees with its result",
                "a Boolean weak_type matching the traced result",
                category="lowering-defect",
            )
        _same_shape(target.shape, source.shape, site)
        if source.dtype == target.dtype == "bool":
            site.require(
                set(eqn.params)
                <= {"new_dtype", "weak_type", "sharding", "out_sharding"}
                and "weak_type" in eqn.params
                and all(eqn.params.get(key) is None for key in _PLACEMENT_PARAMS),
                "Boolean conversion attributes or placement are outside the profile",
                "an explicit same-Boolean conversion with native weakness and unspecified placement",
            )
        elif source.dtype == "bool" and target.dtype == "int32":
            uses = consumers.get(id(eqn.outvars[0]), [])
            site.require(
                source.shape == ()
                and bool(uses)
                and all(use == ("cond", 0) for use in uses)
                and id(eqn.outvars[0]) not in graph_outputs,
                "Boolean-to-int32 conversion is not a conditional selector",
                "a scalar Boolean converted solely for two-branch cond selection",
            )
        else:
            _float_values((source, target), site)
    elif name == "broadcast_in_dim":
        _arity(eqn, 1, 1, site)
        _params(
            eqn,
            {"shape", "broadcast_dimensions"},
            site,
            required={"shape", "broadcast_dimensions"},
        )
        source, target = inputs[0], outputs[0]
        if source.dtype == target.dtype == "bool":
            site.require(
                all(eqn.params.get(key) is None for key in _PLACEMENT_PARAMS),
                "Boolean broadcast placement is outside the profile",
                "an unplaced static Boolean broadcast",
            )
            site.require(
                bool(eqn.invars[0].aval.weak_type)
                == bool(eqn.outvars[0].aval.weak_type),
                "Boolean broadcast changes native weakness",
                "the original Boolean weakness",
                category="lowering-defect",
            )
        else:
            _float_values((*inputs, *outputs), site)
        dimensions = eqn.params.get("broadcast_dimensions", ())
        site.require(
            len(dimensions) == len(source.shape)
            and all(
                isinstance(dim, (int, np.integer))
                and not isinstance(dim, bool)
                and 0 <= dim < len(target.shape)
                for dim in dimensions
            )
            and tuple(sorted(set(dimensions))) == tuple(dimensions),
            "invalid broadcast dimensions",
            "one distinct, increasing output axis per input axis",
            category="lowering-defect",
        )
        site.require(
            tuple(eqn.params.get("shape", ())) == target.shape
            and source.dtype == target.dtype
            and all(
                size in (1, target.shape[axis])
                for size, axis in zip(source.shape, dimensions, strict=True)
            ),
            "broadcast attributes or result disagree with the operand",
            "a matching target shape/dtype and compatible mapped dimensions",
            category="lowering-defect",
        )
    elif name == "reduce_sum":
        _arity(eqn, 1, 1, site)
        _params(eqn, {"axes"}, site, required={"axes"})
        _float_values((*inputs, *outputs), site)
        axes = eqn.params.get("axes", ())
        source, target = inputs[0], outputs[0]
        site.require(
            all(
                isinstance(axis, (int, np.integer))
                and not isinstance(axis, bool)
                and 0 <= axis < len(source.shape)
                for axis in axes
            )
            and len(set(axes)) == len(axes),
            "invalid reduction axes",
            "distinct axes within the operand rank",
            category="lowering-defect",
        )
        site.require(
            source.dtype == target.dtype,
            "reduction result dtype differs from its operand",
            "explicit conversion before a wider reduction",
            category="lowering-defect",
        )
        _same_shape(
            target.shape,
            tuple(size for axis, size in enumerate(source.shape) if axis not in axes),
            site,
        )
    elif name == "dot_general":
        _validate_dot(eqn, inputs, outputs, site)
    elif name == "reshape":
        _validate_bias_reshape(eqn, inputs, outputs, site)
    elif name == "bitcast_convert_type":
        _arity(eqn, 1, 1, site)
        _params(eqn, {"new_dtype"}, site, required={"new_dtype"})
        source, target = inputs[0], outputs[0]
        site.require(
            (source.dtype, target.dtype) == ("float32", "uint32"),
            "bitcast type pair is outside the profile",
            "float32-to-uint32 bit representation with equal word width and shape",
        )
        _same_shape(target.shape, source.shape, site)
        site.require(
            np.dtype(eqn.params.get("new_dtype")) == np.dtype(np.uint32),
            "bitcast attribute disagrees with its result dtype",
            "new_dtype=uint32",
            category="lowering-defect",
        )
        if site.opset != 26:
            site.fail(
                "ONNX BitCast requires opset 26",
                "opset 26 for float32-to-uint32 BitCast",
                "select opset=26 if the target supports it, or remove the bitcast; export never upgrades the opset automatically",
                category="representation-limit",
            )
    elif name == "cond":
        _params(eqn, {"branches"}, site, required={"branches"})
        site.require(
            bool(inputs),
            "conditional selector is missing",
            "a scalar int32 selector",
            category="lowering-defect",
        )
        selector = inputs[0]
        site.require(
            selector == _Tensor("int32", ()),
            "invalid conditional selector type/shape",
            "a scalar int32 selector",
            category="lowering-defect",
        )
        selector_var = eqn.invars[0]
        if isinstance(selector_var, JaxLiteral):
            site.require(
                int(selector_var.val) in (0, 1),
                "conditional literal selector is outside the profile",
                "a Boolean selector represented as 0 or 1",
            )
        site.require(
            all(value.dtype != "int32" for value in (*inputs[1:], *outputs)),
            "int32 escapes the conditional selector role",
            "floating, Boolean, or uint32 branch operands/results",
        )
        branches = eqn.params.get("branches", ())
        site.require(
            len(branches) == 2,
            "conditional branch count is outside the profile",
            "exactly two cond branches",
        )
        for index, branch in enumerate(branches):
            body = getattr(branch, "jaxpr", branch)
            branch_inputs = [_tensor(var, site) for var in body.invars]
            branch_outputs = [_tensor(var, site) for var in body.outvars]
            site.require(
                branch_inputs == inputs[1:]
                and branch_outputs == outputs
                and _signature(body.invars, site) == _signature(eqn.invars[1:], site)
                and _signature(body.outvars, site) == _signature(eqn.outvars, site),
                "branch signature disagrees with the conditional",
                "matching branch operand/result dtypes and shapes",
                category="lowering-defect",
            )
            try:
                _validate_scope(body, site.opset, ancestors)
            except SourcePrecisionError as exc:
                raise exc.with_context(f"{site.location}/branch {index}") from exc
    elif name is not None and name.startswith("onnx_fn::"):
        site.require(
            all(value.dtype != "int32" for value in (*inputs, *outputs)),
            "int32 crosses a function boundary outside its selector role",
            "floating, Boolean, or uint32 function operands/results",
        )
    else:
        site.fail(
            "primitive is outside the profile",
            "only the finite source-static-v1 operations; loops, scan, unaudited JIT bodies/options, and integer arithmetic are excluded",
        )


def validate_source_jaxpr(jaxpr: Any, opset: int) -> None:
    """Validate one scope and recursively inspect conditionals and plain JIT.

    Pass this function as the common converter's validator to also check actual
    retraced ONNX Function bodies. A callback alone does not identify the enclosing
    function name; its caller can prefix ``SourcePrecisionError.with_context``.
    """
    _validate_scope(jaxpr, opset, frozenset())


def _validate_scope(jaxpr: Any, opset: int, ancestors: frozenset[int]) -> None:
    site = _Site("graph", opset)
    site.require(
        not getattr(jaxpr, "is_high", False),
        "high-level traced bodies are outside the profile",
        "low-level pure tensor scopes",
    )
    site.require(
        id(jaxpr) not in ancestors and len(ancestors) < 32,
        "cyclic or excessively nested traced body",
        "an acyclic bounded tree of native scopes",
        category="lowering-defect",
    )
    ancestors = ancestors | {id(jaxpr)}
    bound: set[int] = set()
    for var in (*jaxpr.invars, *jaxpr.constvars):
        site.require(
            not isinstance(var, (JaxLiteral, DropVar)) and id(var) not in bound,
            "duplicate or invalid scope binding",
            "distinct input and constant definitions",
            category="lowering-defect",
        )
        bound.add(id(var))
    for index, eqn in enumerate(jaxpr.eqns):
        binding_site = _Site(f"equation {index}", opset, eqn.primitive.name)
        for var in eqn.invars:
            binding_site.require(
                isinstance(var, JaxLiteral) or id(var) in bound,
                "unresolved nonliteral operand",
                "an existing scope binding or literal before use",
                category="lowering-defect",
            )
        for var in eqn.outvars:
            if isinstance(var, DropVar):
                continue
            binding_site.require(
                not isinstance(var, JaxLiteral) and id(var) not in bound,
                "result redefines an existing binding",
                "one definition per non-dropped result",
                category="lowering-defect",
            )
            bound.add(id(var))
    for var in jaxpr.outvars:
        site.require(
            isinstance(var, JaxLiteral) or id(var) in bound,
            "unbound scope output",
            "a bound input, constant, result, or literal; repeated aliases are permitted",
            category="lowering-defect",
        )
    site.require(
        opset in (23, 26),
        f"opset {opset} is outside the profile",
        "explicit opset 23 or 26",
    )
    # Attribute an actual effectful dot to its equation before the graph-wide
    # purity guard. Effects elsewhere retain the existing profile exclusion.
    for index, eqn in enumerate(jaxpr.eqns):
        if eqn.primitive.name == "dot_general" and eqn.effects:
            _Site(
                f"equation {index}",
                opset,
                "dot_general",
                tuple(str(var.aval.dtype) for var in (*eqn.invars, *eqn.outvars)),
            ).fail(
                "matrix equation has effects",
                "an effect-free dot_general equation",
                category="lowering-defect",
            )
    site.require(
        not jaxpr.effects,
        "effects are outside the profile",
        "a pure, effect-free traced function",
    )
    graph_outputs = {id(var) for var in jaxpr.outvars}
    consumers: dict[int, list[tuple[str, int]]] = {}
    selector_producers: set[int] = set()
    for eqn in jaxpr.eqns:
        for index, var in enumerate(eqn.invars):
            consumers.setdefault(id(var), []).append((eqn.primitive.name, index))
        if (
            eqn.primitive.name == "convert_element_type"
            and len(eqn.invars) == len(eqn.outvars) == 1
        ):
            if (
                np.dtype(eqn.invars[0].aval.dtype).name == "bool"
                and np.dtype(eqn.outvars[0].aval.dtype).name == "int32"
            ):
                selector_producers.add(id(eqn.outvars[0]))
    for role, variables in (("input", jaxpr.invars), ("constant", jaxpr.constvars)):
        for index, var in enumerate(variables):
            boundary_site = _Site(
                f"{role} {index}",
                opset,
                requested_dtypes=(str(var.aval.dtype),),
            )
            value = _tensor(var, boundary_site)
            boundary_site.require(
                value.dtype != "int32",
                "int32 appears at a scope boundary",
                "int32 only as an internal Boolean conditional selector",
            )
    # Function formals may correctly remap caller symbols. Validate each actual
    # retraced scope locally while preserving identity within that scope.
    formals = (*jaxpr.invars, *jaxpr.constvars)
    try:
        validate_source_shapes([tuple(var.aval.shape) for var in formals])
    except ValueError as exc:
        site.fail("invalid symbolic scope metadata", str(exc))
    formal_symbols = {
        value.symbol
        for var in formals
        if (value := _tensor(var, site)).symbol is not None
    }
    for index, eqn in enumerate(jaxpr.eqns):
        variables = (*eqn.invars, *eqn.outvars)
        params = {
            name: repr(value)
            for name, value in eqn.params.items()
            if name
            in {
                "precision",
                "preferred_element_type",
                "dimension_numbers",
                "new_dtype",
                "out_dtype",
                "accuracy",
                "axes",
                "broadcast_dimensions",
                "shape",
            }
        }
        eqn_site = _Site(
            f"equation {index}",
            opset,
            eqn.primitive.name,
            tuple(str(var.aval.dtype) for var in variables),
            params,
        )
        eqn_site.require(
            not eqn.effects,
            "equation effects are outside the profile",
            "effect-free primitive equations",
        )
        if (
            eqn.primitive.name == "cond"
            and eqn.invars
            and not isinstance(eqn.invars[0], JaxLiteral)
        ):
            eqn_site.require(
                id(eqn.invars[0]) in selector_producers,
                "conditional selector lacks an audited Boolean conversion",
                "a scalar bool-to-int32 conversion in this scope",
            )
        try:
            _validate_equation(
                eqn, eqn_site, consumers, graph_outputs, formal_symbols, ancestors
            )
        except SourcePrecisionError:
            raise
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            eqn_site.fail(
                f"malformed traced primitive metadata: {exc}",
                "valid primitive attributes consistent with the traced avals",
                "report this tracing/lowering defect with a minimal reproduction",
                category="lowering-defect",
            )
    for index, var in enumerate(jaxpr.outvars):
        boundary_site = _Site(
            f"output {index}", opset, requested_dtypes=(str(var.aval.dtype),)
        )
        value = _tensor(var, boundary_site)
        boundary_site.require(
            value.symbol is None or value.symbol in formal_symbols,
            "symbolic output differs from this scope's formal batch contract",
            "the same atomic batch variable and scope as actual formals",
        )
        boundary_site.require(
            value.dtype != "int32",
            "int32 appears at a scope boundary",
            "int32 only as an internal Boolean conditional selector",
        )
