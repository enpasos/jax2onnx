# jax2onnx/converter/source_elementary.py

"""Native elementary and value-reduction contracts for SOURCE tracing.

These checks establish typed lowering eligibility. Numerical and CPU kernel
evidence remain separate. Sensitive decompositions without a source algorithm
are deliberately absent, while their original inventory rows remain obligations.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any, Final, TYPE_CHECKING

import jax
import numpy as np

if TYPE_CHECKING:
    from jax2onnx.converter.source_profile import _Site, _Tensor

_FLOATS: Final = frozenset({"float16", "bfloat16", "float32", "float64"})
_INTS: Final = frozenset(
    {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
)
_TRANSCENDENTALS: Final = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
    }
)
_REDUCTIONS: Final = frozenset(
    {
        "reduce_sum",
        "reduce_prod",
        "reduce_min",
        "reduce_max",
        "reduce_and",
        "reduce_or",
        "reduce_xor",
    }
)
ELEMENTARY_OPERATIONS: Final = (
    _TRANSCENDENTALS
    | _REDUCTIONS
    | {
        "log",
        "log1p",
        "expm1",
        "floor",
        "ceil",
        "abs",
        "sign",
        "is_finite",
        "min",
        "max",
        "clamp",
        "integer_pow",
        "round",
        "logistic",
        "pow",
        "rem",
    }
)


def validate_original_elementary_opset(
    inputs: Sequence[_Tensor], outputs: Sequence[_Tensor], site: _Site
) -> None:
    """The finite original Div21 / approximate-GELU18 expression schemas.

    Exact GELU18 still reaches the owning Erfc gate. This adds no implicit
    operator upgrade, matrix/PRNG eligibility, or BF16 support at old opsets.
    """
    if site.opset not in (18, 21):
        return
    name = site.primitive
    site.require(
        name is None
        or name
        in {
            "add",
            "sub",
            "mul",
            "div",
            "neg",
            "sqrt",
            "rsqrt",
            "square",
            "exp",
            "tanh",
            "integer_pow",
            "abs",
            "reduce_sum",
            "reshape",
            "broadcast_in_dim",
            "convert_element_type",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "select_n",
            "jit",
            "cond",
            "while",
            "erfc",
            "custom_jvp_call",
            "stop_gradient",
        }
        or bool(name and name.startswith("onnx_fn::")),
        "operation is outside the original opset18/21 elementary intersection",
        "the preserved FP32 Div21 or approximate-GELU18 native expression schemas",
    )
    site.require(
        all(
            value.dtype in {"float32", "bool", "int32", "int64"}
            for value in (*inputs, *outputs)
        ),
        "dtype is outside the original opset18/21 typed expression",
        "strong/weak FP32 data with Boolean predicates and native index roles",
    )


def validate_static_zeros(eqn: Any, outputs: Sequence[_Tensor], site: _Site) -> None:
    """Admit the existing strong FP32/FP64 zero-fill carrier at loop opsets.

    The normal tracing bindings already replace original ``jnp.zeros`` calls
    with this exact carrier. Its resolved dtype preserves native x64 behavior;
    other creation wrappers and dtype/opset routes retain their own exclusions.
    """
    from jax2onnx._compat.jax import ShapedArray
    from jax2onnx.converter.source_profile import _arity
    from jax2onnx.plugins.jax.numpy.zeros import JnpZerosPlugin

    _arity(eqn, 0, 1, site)
    site.require(
        eqn.primitive is JnpZerosPlugin._PRIM and set(eqn.params) == {"shape", "dtype"},
        "zero-fill carrier identity or parameters are not the native schema",
        "the existing zero-input jnp.zeros primitive with exactly shape and dtype",
        category="lowering-defect",
    )
    target = outputs[0]
    site.require(
        site.opset in {23, 26} and target.dtype in {"float32", "float64"},
        "zero-fill dtype or opset is outside the audited loop initialization route",
        "native FP32/FP64 zero fill at opset23/26",
    )
    shape, dtype = eqn.params["shape"], eqn.params["dtype"]
    site.require(
        isinstance(shape, tuple)
        and all(type(dim) is int and dim >= 0 for dim in shape)
        and isinstance(dtype, np.dtype)
        and dtype == np.dtype(target.dtype)
        and shape == target.shape
        and target.symbol is None
        and type(eqn.outvars[0].aval) is ShapedArray
        and eqn.outvars[0].aval.weak_type is False,
        "zero-fill attributes or result differ from the native typed contract",
        "static nonnegative native extents, the resolved dtype and a matching strong ShapedArray",
        category="lowering-defect",
    )


def validate_elementary_equation(
    eqn: Any, inputs: Sequence[_Tensor], outputs: Sequence[_Tensor], site: _Site
) -> None:
    # Import lazily: the owning scope supplies actual _Tensor/_Site instances.
    from jax2onnx.converter.source_profile import (
        _arity,
        _axes,
        _params,
        _pointwise_shape,
        _same_shape,
    )

    name = site.primitive
    reduction = name in _REDUCTIONS
    arity = 3 if name == "clamp" else 2 if name in {"min", "max", "pow", "rem"} else 1
    _arity(eqn, arity, 1, site)
    allowed = (
        {"axes"}
        if reduction
        else (
            {"y"}
            if name == "integer_pow"
            else (
                {"rounding_method"}
                if name == "round"
                else (
                    {"accuracy"}
                    if name
                    in _TRANSCENDENTALS | {"log", "log1p", "expm1", "logistic", "pow"}
                    else set()
                )
            )
        )
    )
    required = allowed if reduction or name in {"integer_pow", "round"} else set()
    _params(eqn, allowed, site, required=required)
    site.require(
        eqn.params.get("accuracy") is None
        and all(eqn.params.get(key) is None for key in ("sharding", "out_sharding")),
        "elementary accuracy or placement is outside this native contract",
        "absent/None accuracy and unspecified native placement",
    )
    source, target = inputs[0], outputs[0]
    dtype = source.dtype
    if name == "clamp":
        source = inputs[1]
        dtype = source.dtype
    site.require(
        all(value.dtype == dtype for value in inputs)
        and target.dtype == ("bool" if name == "is_finite" else dtype),
        "elementary operand/result dtypes disagree",
        "one exact native data dtype, or the native is_finite Boolean result",
        category="lowering-defect",
    )
    site.require(
        eqn.outvars[0].aval.weak_type == all(var.aval.weak_type for var in eqn.invars),
        "elementary result weakness differs from its native operands",
        "the native pointwise/reduction weak-type rule",
        category="lowering-defect",
    )
    if reduction:
        axes = _axes(eqn.params["axes"], len(source.shape), site, "reduction axes")
        boolean = name in {"reduce_and", "reduce_or", "reduce_xor"}
        site.require(
            dtype == "bool" if boolean else dtype in _FLOATS | _INTS,
            "reduction data type lacks a native lowering contract",
            "Boolean logical reductions separately from numeric value reductions; integer bitwise reductions remain implementation work",
        )
        if not boolean:
            if name in {"reduce_sum", "reduce_prod"}:
                site.require(
                    dtype in _FLOATS | {"int32", "int64", "uint32", "uint64"},
                    "small-integer accumulation lacks a native schema/overflow proof",
                    "explicit original promotion, or native 32/64-bit numeric accumulation",
                )
            if name == "reduce_sum" and dtype == "uint32":
                # Existing shared lowering uses exact INT64 work, then the
                # original U32 modulo cast. Prove every possible partial sum
                # fits; an unbounded reduced B cannot satisfy this proof.
                extents = [source.shape[axis] for axis in axes]
                site.require(
                    all(type(size) is int for size in extents)
                    and prod(int(size) for size in extents) * ((1 << 32) - 1)
                    <= (1 << 63) - 1,
                    "unsigned sum INT64 work is not bounded by the traced reduction extent",
                    "static reduced extents with every UINT32 partial sum bounded by INT64_MAX",
                )
        expected_shape = tuple(
            size for axis, size in enumerate(source.shape) if axis not in axes
        )
        _same_shape(target.shape, expected_shape, site)
        site.require(
            target.symbol == (None if 0 in axes else source.symbol),
            "reduction result changes its retained native symbolic scope",
            "remove B exactly when its actual leading axis is reduced",
            category="lowering-defect",
        )
        return
    if name in {"abs", "sign", "min", "max", "clamp"}:
        allowed_dtypes = _FLOATS | _INTS
        if name in {"min", "max"}:
            allowed_dtypes |= {"bool"}
        site.require(
            dtype in allowed_dtypes,
            "elementary data type is outside its native schema",
            "the operation's native real/integer types",
        )
        if name in {"min", "max", "clamp"} and dtype in _FLOATS:
            site.require(
                dtype != "bfloat16"
                and (site.opset == 23 or name == "max" and site.opset == 16),
                "floating minimum/maximum requires the scoped signed-zero lowering",
                "the measured FP16/FP32/FP64 Min at 23 or Max at 16/23 route",
            )
    else:
        site.require(
            dtype in _FLOATS or name in {"integer_pow", "rem"} and dtype in _INTS,
            "elementary data type is outside its native schema",
            "native floating operations or nonnegative integer_pow on integers",
        )
    if dtype == "bfloat16" and name in _TRANSCENDENTALS:
        site.require(
            site.opset >= 22,
            "BF16 transcendental schema requires opset 22",
            "the explicitly requested BF16-capable operator schema",
        )
    if name in {"log1p", "expm1"}:
        site.require(
            dtype in {"float32", "float64"},
            "low-precision stable elementary algorithms remain implementation work",
            "typed FP32/FP64 compensated log1p or small-range expm1 polynomial",
        )
    if name == "integer_pow":
        exponent = eqn.params["y"]
        site.require(
            type(exponent) is int and (dtype in _FLOATS or exponent >= 0),
            "integer_pow exponent is not a native Python integer",
            "the original static integer exponent; integer operands require a nonnegative exponent",
            category="lowering-defect",
        )
    if name == "round":
        method = eqn.params["rounding_method"]
        site.require(
            type(method) is jax.lax.RoundingMethod
            and method
            in (
                jax.lax.RoundingMethod.AWAY_FROM_ZERO,
                jax.lax.RoundingMethod.TO_NEAREST_EVEN,
            ),
            "rounding method is not a native supported tie mode",
            "the actual AWAY_FROM_ZERO or TO_NEAREST_EVEN enum",
            category="lowering-defect",
        )
    _pointwise_shape(inputs, target, site)


def validate_integer_arithmetic(
    eqn: Any, inputs: Sequence[_Tensor], outputs: Sequence[_Tensor], site: _Site
) -> bool:
    """Handle only discrete Mul/Neg/Div, preserving existing floating checks."""
    if site.primitive not in {"mul", "neg", "div"} or not all(
        value.dtype in _INTS for value in (*inputs, *outputs)
    ):
        return False
    from jax2onnx.converter.source_profile import (
        _arity,
        _params,
        _pointwise_shape,
        _tensor_result,
    )

    _arity(eqn, 1 if site.primitive == "neg" else 2, 1, site)
    _params(eqn, {"out_dtype"} if site.primitive == "mul" else set(), site)
    site.require(
        eqn.params.get("out_dtype") is None,
        "integer Mul changes its accumulator dtype",
        "native unchanged integer output dtype",
    )
    _tensor_result(eqn, inputs, outputs[0], site)
    _pointwise_shape(inputs, outputs[0], site)
    return True


def validate_custom_jvp_forward(
    eqn: Any, site: _Site, ancestors: frozenset[int]
) -> None:
    """Validate the already traced native forward body used by activations.

    The derivative WrappedFun is native metadata and is never invoked during
    export. Captures are actual leading inputs after native const conversion.
    The existing owning lowerer inlines exactly this concrete forward Jaxpr.
    """
    from jax2onnx._compat.jax import ClosedJaxpr, WrappedFun
    from jax2onnx.converter.source_profile import (
        SourcePrecisionError,
        _signature,
        _validate_scope,
    )

    site.require(
        set(eqn.params)
        == {"call_jaxpr", "jvp_jaxpr_fun", "num_consts", "symbolic_zeros"},
        "custom JVP forward carrier has foreign or missing fields",
        "the exact native concrete-forward four-field schema",
        category="lowering-defect",
    )
    params = eqn.params
    closed = params["call_jaxpr"]
    site.require(
        isinstance(closed, ClosedJaxpr)
        and isinstance(params["jvp_jaxpr_fun"], WrappedFun)
        and type(params["num_consts"]) is int
        and 0 <= params["num_consts"] <= len(eqn.invars)
        and type(params["symbolic_zeros"]) is bool,
        "custom JVP forward fields are not native concrete metadata",
        "a concrete native Jaxpr and WrappedFun, integer capture count and Boolean symbolic_zeros",
        category="lowering-defect",
    )
    body = closed.jaxpr  # One unwrap also handles JAX 0.11's self property.
    site.require(
        not body.constvars and not closed.consts,
        "custom JVP forward has unconverted constant bindings",
        "native converted captures as leading ordinary inputs",
        category="lowering-defect",
    )
    site.require(
        _signature(eqn.invars, site) == _signature(body.invars, site)
        and _signature(eqn.outvars, site) == _signature(body.outvars, site),
        "custom JVP outer and forward-body signatures disagree",
        "the exact input/output dtype, shape, weakness and actual symbolic scope",
        category="lowering-defect",
    )
    try:
        _validate_scope(body, site.opset, ancestors)
    except SourcePrecisionError as exc:
        raise exc.with_context(f"{site.location}/custom JVP forward") from exc
