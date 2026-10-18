# tests/extra_tests/converter/source_precision_diagnostics.py

"""Narrow structural checks for tooling diagnostics, never a lowering audit.

Only the explicit families below are inspected. A registered lowerer alone does
not admit an operation. Unknown operations, attributes, placement and effects
remain pending; malformed known structures fail. No model values are evaluated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Number
from typing import Any, Final, NoReturn

import numpy as np

from jax2onnx._compat.jax import DropVar, Literal, scan_arity
from jax2onnx.converter.source_profile import SourcePrecisionError
from scripts.source_precision_inventory import canonical_contract


_DTYPES: Final[frozenset[str]] = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    }
)
_PLACEMENT: Final[frozenset[str]] = frozenset(
    {
        "sharding",
        "out_sharding",
        "in_shardings",
        "out_shardings",
        "in_layouts",
        "out_layouts",
        "ctx_mesh",
    }
)
_RULES: dict[str, tuple[int | None, int, frozenset[str]]] = {
    **{
        name: (2, 1, frozenset())
        for name in (
            "add",
            "sub",
            "div",
            "max",
            "min",
            "lt",
            "le",
            "gt",
            "ge",
            "eq",
            "ne",
        )
    },
    "mul": (2, 1, frozenset({"out_dtype"})),
    **{
        name: (1, 1, frozenset({"accuracy"}))
        for name in ("sin", "cos", "neg", "real", "imag")
    },
    "complex": (2, 1, frozenset()),
    "select_n": (None, 1, frozenset()),
    "convert_element_type": (1, 1, frozenset({"new_dtype", "weak_type"})),
    "broadcast_in_dim": (1, 1, frozenset({"shape", "broadcast_dimensions"})),
    "reshape": (1, 1, frozenset({"new_sizes", "dimensions"})),
    "transpose": (1, 1, frozenset({"permutation"})),
    "squeeze": (1, 1, frozenset({"dimensions"})),
    "slice": (1, 1, frozenset({"start_indices", "limit_indices", "strides"})),
    "gather": (
        2,
        1,
        frozenset(
            {
                "dimension_numbers",
                "slice_sizes",
                "unique_indices",
                "indices_are_sorted",
                "mode",
                "fill_value",
            }
        ),
    ),
    "reduce_sum": (1, 1, frozenset({"axes"})),
    "dot_general": (
        2,
        1,
        frozenset(
            {"dimension_numbers", "precision", "preferred_element_type", "out_dtype"}
        ),
    ),
    "conv_general_dilated": (
        2,
        1,
        frozenset(
            {
                "window_strides",
                "padding",
                "lhs_dilation",
                "rhs_dilation",
                "dimension_numbers",
                "feature_group_count",
                "batch_group_count",
                "precision",
                "preferred_element_type",
            }
        ),
    ),
    "fft": (1, 1, frozenset({"fft_type", "fft_lengths"})),
}
_NESTED: Final[dict[str, frozenset[str]]] = {
    "cond": frozenset({"branches"}),
    "jit": frozenset(
        {
            "jaxpr",
            "call_jaxpr",
            "name",
            "inline",
            "keep_unused",
            "donated_invars",
            "compiler_options_kvs",
        }
    )
    | _PLACEMENT,
    "pjit": frozenset(
        {
            "jaxpr",
            "call_jaxpr",
            "name",
            "inline",
            "keep_unused",
            "donated_invars",
            "compiler_options_kvs",
        }
    )
    | _PLACEMENT,
    "while": frozenset({"cond_jaxpr", "body_jaxpr", "cond_nconsts", "body_nconsts"}),
    "scan": frozenset(
        {
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
    ),
}


def _body(value: Any) -> Any:
    inner = getattr(value, "jaxpr", value)
    return inner if inner is not value else value


def _typed(aval: Any) -> tuple[str, tuple[Any, ...], bool]:
    def dimension(dim: Any) -> Any:
        if isinstance(dim, bool):
            return dim
        if isinstance(dim, Integral):
            return int(dim)
        if isinstance(dim, str) or type(dim).__module__ == "jax._src.export.shape_poly":
            return str(dim)
        return {
            "unresolved_dimension": f"{type(dim).__module__}.{type(dim).__qualname__}"
        }

    return (
        np.dtype(aval.dtype).name,
        tuple(dimension(dim) for dim in aval.shape),
        bool(getattr(aval, "weak_type", False)),
    )


def typed_trace_manifest(closed: Any, _depth: int = 0) -> dict[str, Any]:
    """Snapshot original typed topology/attributes without address-bearing reprs."""
    if _depth > 20:
        return {"unresolved": "trace nesting limit"}
    body = _body(closed)
    unresolved: list[str] = []
    variables = {
        id(var): index for index, var in enumerate((*body.constvars, *body.invars))
    }

    def parameter(value: Any, key: str = "parameter") -> Any:
        candidate = _body(value)
        if all(hasattr(candidate, name) for name in ("eqns", "invars", "outvars")):
            return typed_trace_manifest(value, _depth + 1)
        if isinstance(value, (tuple, list)):
            return [parameter(item, key) for item in value]
        if isinstance(value, Mapping):
            if not all(isinstance(name, str) for name in value):
                unresolved.append(f"{key}: non-string trace parameter keys")
                return {
                    "unresolved_mapping_keys": sorted(
                        f"{type(name).__module__}.{type(name).__qualname__}"
                        for name in value
                    )
                }
            return {
                str(name): parameter(item, str(name))
                for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if key in {"ft_in", "ft_out"} and hasattr(value, "elts"):
            return {
                "flat_tree_role_partitions": [len(group) for group in value.elts],
                "leaf_count": len(value),
            }
        typename = f"{type(value).__module__}.{type(value).__name__}"
        if key in _PLACEMENT and typename in {
            "jax._src.sharding_impls.UnspecifiedValue",
            "jax._src.named_sharding.UnspecifiedValue",
        }:
            return {"placement": "unspecified"}
        if key in _PLACEMENT and typename in {
            "jax._src.mesh.AbstractMesh",
            "jax._src.mesh.Mesh",
        }:
            return {
                "mesh_type": typename,
                "axis_names": list(value.axis_names),
                "axis_sizes": dict(value.shape),
                "axis_types": parameter(
                    getattr(value, "axis_types", None), "axis_types"
                ),
            }
        if key in _PLACEMENT and hasattr(value, "is_fully_replicated"):
            return {
                "sharding_type": typename,
                "fully_replicated": bool(value.is_fully_replicated),
                "spec": parameter(getattr(value, "spec", None), "spec"),
                "mesh": parameter(getattr(value, "mesh", None), "ctx_mesh"),
                "memory_kind": getattr(value, "memory_kind", None),
            }
        return canonical_contract(value, unresolved=unresolved, location=key)

    def use(var: Any) -> Any:
        if isinstance(var, Literal):
            return {"literal": parameter(var.val), "aval": _typed(var.aval)}
        return {"var": variables.get(id(var), "unbound"), "aval": _typed(var.aval)}

    equations = []
    for eqn in body.eqns:
        inputs = [use(var) for var in eqn.invars]
        outputs = []
        for var in eqn.outvars:
            variables[id(var)] = len(variables)
            outputs.append(use(var))
        equations.append(
            {
                "primitive": eqn.primitive.name,
                "inputs": inputs,
                "outputs": outputs,
                "params": parameter(eqn.params),
                "effects": sorted(type(effect).__qualname__ for effect in eqn.effects),
            }
        )
        if eqn.primitive.name == "scan":
            equations[-1]["scan_role_arity"] = scan_arity(eqn.params, len(eqn.invars))
    return {
        "inputs": [_typed(var.aval) for var in body.invars],
        "consts": [_typed(var.aval) for var in body.constvars],
        "const_payloads": parameter(getattr(closed, "consts", ())),
        "outputs": [use(var) for var in body.outvars],
        "equations": equations,
        "effects": sorted(type(effect).__qualname__ for effect in body.effects),
        "unresolved_parameters": unresolved,
    }


@dataclass(frozen=True)
class StructuralScope:
    location: str
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    operations: tuple[tuple[str, str, tuple[str, ...]], ...]


class DiagnosticValidator:
    """A fresh per-export observer and bounded recursive structural validator."""

    def __init__(self, lowerers: Mapping[str, Any] | None = None) -> None:
        if lowerers is None:
            from jax2onnx.plugins.plugin_system import PLUGIN_REGISTRY

            lowerers = PLUGIN_REGISTRY
        self.lowerers = lowerers
        self.scopes: list[StructuralScope] = []
        self.failure: SourcePrecisionError | None = None

    def __call__(self, jaxpr: Any, opset: int) -> None:
        try:
            self._walk(jaxpr, opset, f"scope {len(self.scopes)}", frozenset())
        except SourcePrecisionError as exc:
            self.failure = exc
            raise

    def _fail(
        self,
        message: str,
        location: str,
        opset: int,
        primitive: str | None = None,
        *,
        pending: bool = False,
    ) -> NoReturn:
        raise SourcePrecisionError(
            message,
            phase="diagnostic-validation",
            category="profile-exclusion" if pending else "lowering-defect",
            location=location,
            primitive=primitive,
            opset=opset,
            requirement="the explicit effect-free typed static diagnostic families and matching nested signatures",
            action=(
                "retain this cell as pending and review its structural form"
                if pending
                else "repair the malformed typed structure before exporting"
            ),
        )

    def _tensor(
        self, var: Any, location: str, opset: int
    ) -> tuple[str, tuple[Any, ...], bool]:
        try:
            result = _typed(var.aval)
        except (TypeError, ValueError, AttributeError) as exc:
            self._fail(f"Missing typed tensor aval: {exc}", location, opset)
        dtype, shape, _ = result
        if dtype not in _DTYPES:
            self._fail(f"Unreviewed dtype {dtype}", location, opset, pending=True)
        if any(
            isinstance(dim, bool) or (isinstance(dim, Integral) and int(dim) < 0)
            for dim in shape
        ):
            self._fail(f"Invalid static tensor dimension {shape}", location, opset)
        if any(
            isinstance(dim, bool) or not isinstance(dim, Integral) or dim < 0
            for dim in shape
        ):
            self._fail(
                f"Unreviewed or invalid static shape {shape}",
                location,
                opset,
                pending=True,
            )
        return result

    def _parameter(self, value: Any, key: str) -> bool:
        if value is None or isinstance(
            value, (str, bool, Number, np.generic, np.dtype, Enum)
        ):
            return True
        if isinstance(value, type):
            try:
                np.dtype(value)
                return True
            except TypeError:
                return False
        if isinstance(value, (tuple, list)):
            return all(self._parameter(item, key) for item in value)
        if key in _PLACEMENT:
            typename = f"{type(value).__module__}.{type(value).__name__}"
            if typename in {
                "jax._src.sharding_impls.UnspecifiedValue",
                "jax._src.named_sharding.UnspecifiedValue",
            }:
                return True
            if typename in {"jax._src.mesh.AbstractMesh", "jax._src.mesh.Mesh"}:
                return bool(getattr(value, "empty", False))
            if typename in {
                "jax.sharding.NamedSharding",
                "jax.sharding.SingleDeviceSharding",
                "jaxlib._jax.NamedSharding",
                "jaxlib._jax.SingleDeviceSharding",
            }:
                return bool(value.is_fully_replicated)
        return False

    def _signature(self, graph: Any) -> tuple[list[Any], list[Any]]:
        body = _body(graph)
        return [(_typed(var.aval)[:2]) for var in body.invars], [
            (_typed(var.aval)[:2]) for var in body.outvars
        ]

    def _walk(
        self, closed: Any, opset: int, location: str, ancestors: frozenset[int]
    ) -> None:
        graph = _body(closed)
        if id(graph) in ancestors or len(ancestors) > 20:
            self._fail(
                "Cyclic or excessively nested Jaxpr", location, opset, pending=True
            )
        ancestors = ancestors | {id(graph)}
        if opset not in (23, 26) or getattr(graph, "effects", True):
            self._fail(
                "Unreviewed opset or effectful scope", location, opset, pending=True
            )
        bound: set[int] = set()
        for role, values in (("constant", graph.constvars), ("input", graph.invars)):
            for index, var in enumerate(values):
                self._tensor(var, f"{location}/{role} {index}", opset)
                if id(var) in bound:
                    self._fail("Duplicate scope binding", location, opset)
                bound.add(id(var))
        consts = getattr(closed, "consts", None)
        if consts is not None:
            if len(consts) != len(graph.constvars):
                self._fail("Constant payload arity mismatch", location, opset)
            for var, value in zip(graph.constvars, consts, strict=True):
                array = np.asarray(value)
                if (array.dtype.name, array.shape) != _typed(var.aval)[:2]:
                    self._fail("Constant payload dtype/shape mismatch", location, opset)
        operations = []
        for index, eqn in enumerate(graph.eqns):
            name = eqn.primitive.name
            site = f"{location}/equation {index}"
            if eqn.effects:
                self._fail("Effectful equation", site, opset, name, pending=True)
            for var in eqn.invars:
                self._tensor(var, site, opset)
                if not isinstance(var, Literal) and id(var) not in bound:
                    self._fail("Equation uses an unbound variable", site, opset, name)
            for var in eqn.outvars:
                self._tensor(var, site, opset)
                if not isinstance(var, DropVar):
                    if id(var) in bound:
                        self._fail(
                            "Equation redefines a bound variable", site, opset, name
                        )
                    bound.add(id(var))
            plugin = self.lowerers.get(name)
            if plugin is None:
                self._fail(
                    "No registered lowerer for this diagnostic operation",
                    site,
                    opset,
                    name,
                    pending=True,
                )
            if name.startswith("onnx_fn::"):
                self._function(eqn, site, opset)
            elif name in _NESTED:
                self._nested(eqn, site, opset, ancestors)
            elif name in _RULES:
                self._ordinary(eqn, site, opset)
            else:
                self._fail(
                    "Operation has no explicit diagnostic structural family",
                    site,
                    opset,
                    name,
                    pending=True,
                )
            operations.append(
                (
                    name,
                    f"{type(plugin).__module__}.{type(plugin).__qualname__}",
                    tuple(sorted(eqn.params)),
                )
            )
        for var in graph.outvars:
            self._tensor(var, f"{location}/output", opset)
            if not isinstance(var, Literal) and id(var) not in bound:
                self._fail("Scope returns an unbound variable", location, opset)
        self.scopes.append(
            StructuralScope(
                location,
                tuple(_typed(var.aval) for var in graph.invars),
                tuple(_typed(var.aval) for var in graph.outvars),
                tuple(operations),
            )
        )

    def _ordinary(self, eqn: Any, site: str, opset: int) -> None:
        name = eqn.primitive.name
        inputs, outputs, params = _RULES[name]
        if (inputs is not None and len(eqn.invars) != inputs) or len(
            eqn.outvars
        ) != outputs:
            self._fail("Known operation has inconsistent arity", site, opset, name)
        if set(eqn.params) - (params | _PLACEMENT) or any(
            not self._parameter(value, key) for key, value in eqn.params.items()
        ):
            self._fail(
                "Unreviewed operation parameter name/form",
                site,
                opset,
                name,
                pending=True,
            )
        if name == "mul" and "out_dtype" in eqn.params:
            dtypes = [_typed(var.aval)[0] for var in eqn.invars]
            if (
                eqn.params["out_dtype"] is not None
                or len(set(dtypes)) != 1
                or dtypes[0] not in {"float16", "bfloat16", "float32", "float64"}
            ):
                self._fail(
                    "Unreviewed Mul out_dtype or operand types",
                    site,
                    opset,
                    name,
                    pending=True,
                )
        try:
            inferred, effects = eqn.primitive.abstract_eval(
                *(var.aval for var in eqn.invars), **eqn.params
            )
            inferred = list(inferred) if eqn.primitive.multiple_results else [inferred]
            if effects or [_typed(aval) for aval in inferred] != [
                _typed(var.aval) for var in eqn.outvars
            ]:
                self._fail(
                    "JAX abstract result differs from equation dtype/shape/weak signature",
                    site,
                    opset,
                    name,
                )
        except SourcePrecisionError:
            raise
        except (TypeError, ValueError, AttributeError) as exc:
            self._fail(f"Invalid known operation parameters: {exc}", site, opset, name)

    def _function(self, eqn: Any, site: str, opset: int) -> None:
        trace = eqn.params.get("__jax2onnx_source_trace")
        if (
            set(eqn.params) - {"__jax2onnx_source_trace", "instance_key"}
            or type(trace).__name__ != "_SourceFunctionTrace"
        ):
            self._fail(
                "Unreviewed ONNX Function call parameter form",
                site,
                opset,
                eqn.primitive.name,
                pending=True,
            )
        if [_typed(aval) for aval in trace.output_avals] != [
            _typed(var.aval) for var in eqn.outvars
        ]:
            self._fail(
                "Function result differs from its original source trace",
                site,
                opset,
                eqn.primitive.name,
            )
        # Actual body retraces invoke this same per-call validator in shared dispatch.

    def _nested(
        self, eqn: Any, site: str, opset: int, ancestors: frozenset[int]
    ) -> None:
        name = eqn.primitive.name
        if set(eqn.params) - _NESTED[name]:
            self._fail(
                "Unreviewed nested operation parameters",
                site,
                opset,
                name,
                pending=True,
            )
        ins = [_typed(var.aval)[:2] for var in eqn.invars]
        outs = [_typed(var.aval)[:2] for var in eqn.outvars]
        children: list[tuple[str, Any]] = []
        valid = False
        try:
            if name in {"jit", "pjit"}:
                child = eqn.params.get("jaxpr", eqn.params.get("call_jaxpr"))
                valid = self._signature(child) == (ins, outs)
                children = [("body", child)]
                for key, value in eqn.params.items():
                    if key not in {"jaxpr", "call_jaxpr"} and not self._parameter(
                        value, key
                    ):
                        self._fail(
                            "Unreviewed JIT parameter form",
                            site,
                            opset,
                            name,
                            pending=True,
                        )
            elif name == "cond":
                branches = eqn.params["branches"]
                if len(branches) != 2:
                    self._fail(
                        "Only two-branch conditionals are inspected",
                        site,
                        opset,
                        name,
                        pending=True,
                    )
                valid = (
                    bool(ins)
                    and ins[0][0] in {"int32", "int64"}
                    and ins[0][1] == ()
                    and all(
                        self._signature(branch) == (ins[1:], outs)
                        for branch in branches
                    )
                )
                children = [
                    (f"branch {index}", branch) for index, branch in enumerate(branches)
                ]
            elif name == "while":
                nc, nb = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
                if not isinstance(nc, Integral) or not isinstance(nb, Integral):
                    self._fail("Invalid while constant counts", site, opset, name)
                nc, nb = int(nc), int(nb)
                carry = ins[nc + nb :]
                cond, body = eqn.params["cond_jaxpr"], eqn.params["body_jaxpr"]
                condition_signature = self._signature(cond)
                condition_outputs = condition_signature[1]
                if (
                    len(condition_outputs) == 1
                    and condition_outputs[0][0] == "bool"
                    and condition_outputs[0][1]
                ):
                    self._fail(
                        "Unreviewed batched while predicate",
                        site,
                        opset,
                        name,
                        pending=True,
                    )
                valid = (
                    nc >= 0
                    and nb >= 0
                    and condition_signature == (ins[:nc] + carry, [("bool", ())])
                    and self._signature(body) == (ins[nc : nc + nb] + carry, carry)
                    and outs == carry
                )
                children = [("condition", cond), ("body", body)]
            elif name == "scan":
                child = eqn.params["jaxpr"]
                nc, ny, _ = scan_arity(eqn.params, len(ins))
                length = eqn.params["length"]
                if not isinstance(length, Integral):
                    self._fail("Invalid scan length", site, opset, name)
                length = int(length)
                carry = ins[nc : nc + ny]
                xs = ins[nc + ny :]
                ys = outs[ny:]
                valid = (
                    length >= 0
                    and all(shape and shape[0] == length for _, shape in (*xs, *ys))
                    and self._signature(child)
                    == (
                        ins[:nc] + carry + [(dtype, shape[1:]) for dtype, shape in xs],
                        carry + [(dtype, shape[1:]) for dtype, shape in ys],
                    )
                    and outs[:ny] == carry
                )
                if "linear" in eqn.params:
                    valid = (
                        valid
                        and len(eqn.params["linear"]) == len(ins)
                        and all(
                            isinstance(value, bool) for value in eqn.params["linear"]
                        )
                    )
                if "ft_out" in eqn.params:
                    groups = getattr(eqn.params["ft_out"], "elts", ())
                    valid = (
                        valid
                        and len(groups) == 2
                        and [len(group) for group in groups] == [ny, len(ys)]
                    )
                if (
                    not isinstance(eqn.params["reverse"], bool)
                    or not isinstance(eqn.params["unroll"], Integral)
                    or eqn.params["unroll"] < 1
                    or eqn.params.get("_split_transpose", False)
                ):
                    self._fail(
                        "Unreviewed scan execution parameter form",
                        site,
                        opset,
                        name,
                        pending=True,
                    )
                children = [("body", child)]
        except SourcePrecisionError:
            raise
        except (TypeError, ValueError, KeyError, AttributeError, IndexError) as exc:
            self._fail(f"Malformed nested signature: {exc}", site, opset, name)
        if not valid:
            self._fail("Nested dtype/shape/arity signature mismatch", site, opset, name)
        for label, child in children:
            self._walk(child, opset, f"{site}/{label}", ancestors)
