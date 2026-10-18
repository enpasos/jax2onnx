# jax2onnx/converter/source_profile.py

"""Pure Jaxpr admission checks for the internal ``source-static-v1`` milestone.

The persisted identifier also covers one audited symbolic expansion: rank-two
equal-shape floating Add with one shared atomic leading batch variable, including
actual named Function bodies and returned aliases. Other symbolic rules remain
separate migration work.

These checks describe an export representation, not runtime numerical evidence.
The top-level facade owns floating input normalization. Recursive callbacks may
receive Boolean comparisons and uint32 bit representations as internal operands.
Function bodies are checked when their lowering retraces them with this callback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, NoReturn

import jax
import numpy as np

from jax2onnx._compat.jax import Literal as JaxLiteral
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
    try:
        dtype = np.dtype(var.aval.dtype).name
        shape = tuple(var.aval.shape)
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
    default = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    precision = eqn.params.get("precision")
    site.require(
        precision is None or precision == default,
        "matrix precision/algorithm is outside the profile",
        "default precision without HIGH, HIGHEST, or an explicit algorithm",
    )
    preferred = eqn.params.get("preferred_element_type")
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


def _validate_equation(
    eqn: Any,
    site: _Site,
    consumers: Mapping[int, list[tuple[str, int]]],
    graph_outputs: set[int],
    formal_symbols: set[tuple[str, object | None]],
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
        not symbolic or name == "add" or bool(name and name.startswith("onnx_fn::")),
        "symbolic operation is outside this audited batch packet",
        "equal-shape floating Add, named Function signatures, and returned aliases",
    )
    if name in {"add", "sub", "mul", "div", "neg", "sqrt", "jax.numpy.sqrt", "gt"}:
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
            site.require(
                inputs[0].symbol is not None
                and inputs[0].symbol == inputs[1].symbol == outputs[0].symbol
                and inputs[0].shape == inputs[1].shape,
                "symbolic Add operands do not share the same (B,K) shape",
                "equal operand shapes with the same native batch variable",
            )
            shape = inputs[0].shape
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
        if source.dtype == "bool" and target.dtype == "int32":
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
        _float_values((*inputs, *outputs), site)
        source, target = inputs[0], outputs[0]
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
                branch_inputs == inputs[1:] and branch_outputs == outputs,
                "branch signature disagrees with the conditional",
                "matching branch operand/result dtypes and shapes",
                category="lowering-defect",
            )
            try:
                validate_source_jaxpr(body, site.opset)
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
            "only the finite source-static-v1 operations; loops, scan, JIT regions, and integer arithmetic are excluded",
        )


def validate_source_jaxpr(jaxpr: Any, opset: int) -> None:
    """Validate one traced scope and recursively inspect two-branch conditionals.

    Pass this function as the common converter's validator to also check actual
    retraced ONNX Function bodies. A callback alone does not identify the enclosing
    function name; its caller can prefix ``SourcePrecisionError.with_context``.
    """
    site = _Site("graph", opset)
    site.require(
        opset in (23, 26),
        f"opset {opset} is outside the profile",
        "explicit opset 23 or 26",
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
            _validate_equation(eqn, eqn_site, consumers, graph_outputs, formal_symbols)
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
