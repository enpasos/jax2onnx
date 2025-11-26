# jax2onnx/sandbox/issue52_scatter_payload_repro.py


"""Reproduce jax2onnx issue #52 without jaxfluids dependencies."""

from __future__ import annotations

import importlib
import json
import re
import pathlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections.abc import Mapping
from enum import Enum

import jax
import jax.extend.core as jax_core_ext
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
from jax import core
from onnxruntime.capi.onnxruntime_pybind11_state import (  # type: ignore[import-untyped]
    Fail,
    InvalidArgument,
)

from jax2onnx import to_onnx
from jax2onnx.converter.typing_support import AxisOverrideInfo, AxisOverrideMap

jax.config.update("jax_enable_x64", True)


DATA_DIR = pathlib.Path(__file__).resolve().parent
PAYLOAD_PATH = DATA_DIR / "issue52_feedforward_payload.npz"
ONNX_PATH = pathlib.Path("sod_issue52_payload.onnx")


@dataclass
class ArrayLoader:
    arrays: Dict[str, np.ndarray]

    def get(self, ref: str) -> np.ndarray:
        return np.asarray(self.arrays[ref])


def _deserialize_aval(desc: Dict[str, Any]) -> core.AbstractValue:
    if desc["type"] == "ShapedArray":
        dtype = None if desc["dtype"] is None else np.dtype(desc["dtype"])
        return core.ShapedArray(
            tuple(desc["shape"]), dtype, weak_type=desc.get("weak_type", False)
        )
    if desc["type"] == "AbstractToken":
        return core.AbstractToken()
    raise TypeError(f"Unsupported aval description: {desc}")


def _deserialize_var(
    desc: Dict[str, Any], var_map: Dict[str, jax_core_ext.Var]
) -> jax_core_ext.Var:
    name = desc["name"]
    if name in var_map:
        return var_map[name]
    aval_desc = desc.get("aval")
    if not isinstance(aval_desc, dict):  # pragma: no cover - debug aid
        raise TypeError(f"Unexpected aval descriptor for {name!r}: {aval_desc!r}")
    aval = _deserialize_aval(aval_desc)
    if not isinstance(aval, core.AbstractValue):  # pragma: no cover - debug safeguard
        raise TypeError(f"Invalid aval for var {name!r}: {aval!r}")
    try:
        var = jax_core_ext.Var(aval)
    except TypeError:
        # Older JAX revisions require a suffix/name before the aval.
        var = jax_core_ext.Var(name, aval)
    var_map[name] = var
    return var


def _deserialize_literal(desc: Dict[str, Any], loader: ArrayLoader) -> _LiteralType:
    aval = _deserialize_aval(desc["aval"])
    value_desc = desc["value"]
    if value_desc["kind"] == "array":
        val = loader.get(value_desc["ref"])
    else:
        val = value_desc["value"]
    return _LiteralType(val, aval)


def _deserialize_value(desc: Any, loader: ArrayLoader) -> Any:
    if isinstance(desc, dict) and "__type__" in desc:
        kind = desc["__type__"]
        if kind == "ClosedJaxpr":
            return _deserialize_closed_jaxpr(desc, loader)
        if kind == "Jaxpr":
            return _deserialize_jaxpr(desc, loader)
        if kind == "array":
            return loader.get(desc["ref"])
        if kind == "list":
            return [_deserialize_value(v, loader) for v in desc["items"]]
        if kind == "tuple":
            return tuple(_deserialize_value(v, loader) for v in desc["items"])
        if kind == "namedtuple":
            cls = getattr(_import_module(desc["module"]), desc["name"])
            values = [_deserialize_value(v, loader) for v in desc["fields"]]
            try:
                return cls(*values)
            except TypeError:
                field_names = getattr(cls, "_fields", None)
                if field_names is not None:
                    mapping = dict(zip(field_names, values))
                    return cls(**mapping)
                raise
        if kind == "enum":
            enum_cls = getattr(_import_module(desc["module"]), desc["name"])
            return enum_cls[desc["member"]]
        if kind == "dtype":
            return np.dtype(desc["value"])
        raise TypeError(f"Unsupported descriptor: {desc}")
    if isinstance(desc, dict):
        return {k: _deserialize_value(v, loader) for k, v in desc.items()}
    return desc


def _deserialize_atom(
    desc: Dict[str, Any], loader: ArrayLoader, var_map: Dict[str, jax_core_ext.Var]
) -> core.Atom:
    kind = desc["kind"]
    if kind == "var":
        name = desc["name"]
        if name not in var_map:
            raise KeyError(f"Variable '{name}' referenced before definition")
        return var_map[name]
    if kind == "literal":
        return _deserialize_literal(desc, loader)
    raise TypeError(f"Unsupported atom kind: {kind}")


def _deserialize_eqn(
    desc: Dict[str, Any], loader: ArrayLoader, var_map: Dict[str, jax_core_ext.Var]
) -> core.JaxprEqn:
    primitive = _primitive_registry()[desc["primitive"]]
    invars = [_deserialize_atom(atom, loader, var_map) for atom in desc["invars"]]
    outvars = [_deserialize_var(var, var_map) for var in desc["outvars"]]
    params = _sanitize_params(_deserialize_value(desc["params"], loader))
    eqn = core.new_jaxpr_eqn(
        invars,
        outvars,
        primitive,
        params,
        effects=(),
    )
    return eqn


def _deserialize_jaxpr(desc: Dict[str, Any], loader: ArrayLoader) -> core.Jaxpr:
    var_map: Dict[str, jax_core_ext.Var] = {}
    constvars = [_deserialize_var(var, var_map) for var in desc["constvars"]]
    invars = [_deserialize_var(var, var_map) for var in desc["invars"]]
    outvars = [_deserialize_var(var, var_map) for var in desc["outvars"]]
    eqns = [_deserialize_eqn(eqn, loader, var_map) for eqn in desc["eqns"]]
    return core.Jaxpr(constvars, invars, outvars, eqns)


_PrimitiveType = getattr(jax_core_ext, "Primitive")
_LiteralType = getattr(jax_core_ext, "Literal")
_FALLBACK_IMPLS: Dict[str, Callable[..., Any]] = {
    "square": lambda x: jnp.square(x),
}


class _ArrayCopySemantics(Enum):
    REUSE_INPUT = "REUSE_INPUT"
    DONATE_INPUT = "DONATE_INPUT"
    UNKNOWN = "UNKNOWN"


_ENUM_FALLBACKS: Dict[tuple[str, str], type[Enum]] = {
    ("jaxlib._jax", "ArrayCopySemantics"): _ArrayCopySemantics,
    ("jaxlib.xla_extension", "ArrayCopySemantics"): _ArrayCopySemantics,
}


def _wrap_primitive_bind(primitive: _PrimitiveType) -> None:
    if getattr(primitive, "_sharding_bind_wrapped", False):
        return
    orig_bind = primitive.bind

    def _compat_bind(*args, **params):
        current = params
        while True:
            try:
                return orig_bind(*args, **current)
            except TypeError as exc:
                message = str(exc)
                unexpected = re.search(
                    r"unexpected keyword argument '([^']+)'", message
                )
                if unexpected:
                    key = unexpected.group(1)
                    if key in current:
                        new_params = dict(current)
                        new_params.pop(key, None)
                        current = new_params
                        continue
                missing = re.search(
                    r"required keyword-only argument: '([^']+)'", message
                )
                if missing:
                    key = missing.group(1)
                    if key not in current:
                        new_params = dict(current)
                        new_params[key] = None
                        current = new_params
                        continue
                raise

    primitive.bind = _compat_bind  # type: ignore[assignment]
    primitive._sharding_bind_wrapped = True  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def _primitive_registry() -> Dict[str, _PrimitiveType]:
    def _collect(module: Any, registry: Dict[str, _PrimitiveType]) -> None:
        for attr in getattr(module, "__dict__", {}).values():
            if isinstance(attr, _PrimitiveType):
                _wrap_primitive_bind(attr)
                registry.setdefault(attr.name, attr)

    registry: Dict[str, _PrimitiveType] = {}
    primitives_mod = getattr(jax_core_ext, "primitives", None)
    if primitives_mod is not None:
        _collect(primitives_mod, registry)

    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if not name.startswith("jax"):
            continue
        _collect(module, registry)

    safe_modules = (
        "jax",
        "jax.core",
        "jax.lax",
        "jax.lax.control_flow",
        "jax.lax.parallel",
        "jax.lax.slicing",
        "jax.scipy",
        "jax.numpy",
        "jax.numpy.linalg",
        "jax.numpy.fft",
        "jax.nn",
        "jax.random",
    )
    for module_name in safe_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        _collect(module, registry)

    for attr in core.__dict__.values():
        if isinstance(attr, _PrimitiveType):
            registry.setdefault(attr.name, attr)
    for name, impl in _FALLBACK_IMPLS.items():
        registry.setdefault(name, _build_fallback_primitive(name, impl))
    return registry


def _build_fallback_primitive(name: str, fn: Callable[..., Any]) -> _PrimitiveType:
    prim = _PrimitiveType(name)
    prim.def_impl(fn)
    _wrap_primitive_bind(prim)
    return prim


def _import_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        if module_name == "jaxlib._jax":
            for candidate in ("jaxlib.xla_extension", "jaxlib._xla"):
                try:
                    return importlib.import_module(candidate)
                except ModuleNotFoundError:
                    continue
        raise


def _deserialize_closed_jaxpr(
    desc: Dict[str, Any], loader: ArrayLoader
) -> core.ClosedJaxpr:
    jaxpr = _deserialize_jaxpr(desc["jaxpr"], loader)
    consts = [_deserialize_value(c, loader) for c in desc["consts"]]
    return core.ClosedJaxpr(jaxpr, consts)


def _load_payload() -> Tuple[
    core.ClosedJaxpr,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    data = np.load(PAYLOAD_PATH, allow_pickle=False)
    meta_json = data["meta"].tobytes().decode("utf-8")
    meta = json.loads(meta_json)
    arrays = {k: data[k] for k in data.files if k != "meta"}
    loader = ArrayLoader(arrays)

    closed = _deserialize_closed_jaxpr(meta["closed_jaxpr"], loader)
    prim0 = jnp.asarray(_deserialize_value(meta["prim0"], loader), dtype=jnp.float64)
    initial_time = jnp.array([meta["initial_time"]], dtype=jnp.float64)
    time_step = jnp.array([meta["time_step"]], dtype=jnp.float64)

    return closed, prim0, initial_time, time_step


def _feed_forward_fn(
    closed: core.ClosedJaxpr,
) -> Callable[[jax.Array, jax.Array, jax.Array], Any]:
    def ff(
        y_current: jax.Array,
        t_arr: jax.Array,
        dt_arr: jax.Array,
    ) -> Any:
        return core.eval_jaxpr(closed.jaxpr, closed.consts, y_current, t_arr, dt_arr)

    return ff


def _axis0_override(
    value: Any,
    *,
    op_type: Optional[str] = None,
) -> Optional[AxisOverrideInfo]:
    meta = getattr(value, "meta", None)
    if meta is None:
        return None
    maybe = meta.get("loop_axis0_override")
    if isinstance(maybe, (int, np.integer)):
        return AxisOverrideInfo(extent=int(maybe), op_type=op_type)
    return None


def _shape_tuple(value: Any) -> Tuple[Any, ...]:
    dims = getattr(getattr(value, "shape", None), "dims", None)
    if not dims:
        return ()
    result: List[Any] = []
    for dim in dims:
        if isinstance(dim, (int, np.integer)):
            result.append(int(dim))
            continue
        val = getattr(dim, "value", None)
        if isinstance(val, (int, np.integer)):
            result.append(int(val))
            continue
        result.append(str(dim))
    return tuple(result)


def _format_value(value: Any) -> str:
    shape = _shape_tuple(value)
    override = _axis0_override(value)
    axis_extent = override.extent if override is not None else None
    return f"{value.name} axis0={axis_extent} shape={shape}"


def _find_value(
    ir_model: Any, prefix: str, predicate: Optional[Callable[[Any], bool]] = None
) -> Tuple[Optional[Any], Optional[Any]]:
    for node in ir_model.graph.all_nodes():
        for out in getattr(node, "outputs", ()):
            name = getattr(out, "name", "")
            if not name.startswith(prefix):
                continue
            if predicate is not None and not predicate(out):
                continue
            return node, out
    return None, None


def _trace_axis0(ir_model: ir.Model) -> None:
    print("[AXIS0] ---- Loop / Slice trace ----")
    loop_node, loop_val = _find_value(
        ir_model,
        "loop_out_",
        predicate=lambda v: _axis0_override(v) is not None,
    )
    if loop_val is not None:
        print(f"[AXIS0] loop_out     -> {_format_value(loop_val)}")
    else:
        print("[AXIS0] loop_out     -> <missing>")

    dyn_node, dyn_val = _find_value(
        ir_model,
        "dyn_slice_out_",
        predicate=lambda v: _shape_tuple(v)[:2] == (1, 5),
    )
    if dyn_val is not None:
        print(f"[AXIS0] dyn_slice_out -> {_format_value(dyn_val)}")
        if dyn_node is not None:
            inputs = getattr(dyn_node, "inputs", ())
            for idx, inp in enumerate(inputs):
                print(f"           input[{idx}] {_format_value(inp)}")
    else:
        print("[AXIS0] dyn_slice_out -> <missing>")

    print("[AXIS0] ---- Scatter / Broadcast trace ----")
    reshape_node, reshape_val = _find_value(
        ir_model,
        "bcast_reshape_out_",
        predicate=lambda v: _shape_tuple(v)[:2] == (1, 5),
    )
    if reshape_val is None or reshape_node is None:
        print("[AXIS0] bcast_reshape -> <missing>")
        return

    print(f"[AXIS0] bcast_reshape -> {_format_value(reshape_val)}")

    mul_val = reshape_node.inputs[0] if reshape_node.inputs else None
    mul_node = mul_val.producer() if mul_val is not None else None
    if mul_val is not None and mul_node is not None:
        print(f"[AXIS0] mul_out      -> {_format_value(mul_val)}")
        for idx, inp in enumerate(mul_node.inputs):
            print(f"           input[{idx}] {_format_value(inp)}")
    else:
        print("[AXIS0] mul_out      -> <missing>")
        mul_node = None

    scatter_node = None
    scatter_val = None
    if mul_node is not None:
        for inp in mul_node.inputs:
            producer = inp.producer()
            if producer is not None and producer.op_type == "ScatterND":
                scatter_node = producer
                scatter_val = inp
                break
    if scatter_val is not None and scatter_node is not None:
        print(f"[AXIS0] scatter_out  -> {_format_value(scatter_val)}")
        for idx, inp in enumerate(scatter_node.inputs):
            print(f"           input[{idx}] {_format_value(inp)}")
    else:
        print("[AXIS0] scatter_out  -> <missing>")

    expand_node: Optional[Any] = None
    expand_val: Optional[Any] = None
    uses_raw = reshape_val.uses()
    if isinstance(uses_raw, (list, tuple)):
        uses_list = list(uses_raw)
    else:
        uses_list = list(uses_raw)
    if uses_list:
        expand_node = uses_list[0][0]
        if expand_node is not None:
            outputs = getattr(expand_node, "outputs", None)
            if outputs:
                expand_val = outputs[0]
    if expand_val is not None and expand_node is not None:
        print(f"[AXIS0] expand_out   -> {_format_value(expand_val)}")
        for idx, inp in enumerate(expand_node.inputs):
            print(f"           input[{idx}] {_format_value(inp)}")
    else:
        print("[AXIS0] expand_out   -> <missing>")


def _collect_axis0_overrides(graph: Any, overrides: AxisOverrideMap) -> None:
    for node in graph.all_nodes():
        for out in getattr(node, "outputs", ()):
            producer = out.producer() if hasattr(out, "producer") else None
            op_type = getattr(producer, "op_type", None)
            override = _axis0_override(out, op_type=op_type)
            if isinstance(override, AxisOverrideInfo) and override.extent > 1:
                overrides.setdefault(out.name, override)
        attrs = getattr(node, "attributes", {})
        if not isinstance(attrs, dict):
            continue
        for attr in attrs.values():
            attr_type = getattr(attr, "type", None)
            if attr_type == "GRAPH":
                subgraph = attr.as_graph()
                if subgraph is not None:
                    _collect_axis0_overrides(subgraph, overrides)
            elif attr_type == "GRAPHS":
                for subgraph in attr.as_graphs():
                    _collect_axis0_overrides(subgraph, overrides)


def _restamp_onnx_axis0(graph: onnx.GraphProto, overrides: AxisOverrideMap) -> None:
    _ALLOWED_OPS = {
        "Expand",
        "Mul",
        "Div",
        "Add",
        "Sub",
        "ScatterND",
    }

    def _apply(vi: onnx.ValueInfoProto) -> None:
        info = overrides.get(vi.name)
        if info is None or not info.allows_restamp(_ALLOWED_OPS):
            return
        tensor_type = getattr(vi, "type", None)
        if tensor_type is None or tensor_type.tensor_type is None:
            return
        shape = tensor_type.tensor_type.shape
        if shape is None or not shape.dim:
            return
        dim0 = shape.dim[0]
        if dim0.HasField("dim_value") and dim0.dim_value not in (
            0,
            info.extent,
            1,
        ):
            return
        dim0.ClearField("dim_param")
        dim0.dim_value = info.extent

    for vi in graph.value_info:
        _apply(vi)
    for vi in graph.output:
        _apply(vi)
    for vi in graph.input:
        _apply(vi)

    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _restamp_onnx_axis0(attr.g, overrides)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    _restamp_onnx_axis0(subgraph, overrides)


def _export_to_onnx(
    ff: Callable[[jax.Array, jax.Array, jax.Array], Any],
    prim0: jax.Array,
    t_arr: jax.Array,
    dt_arr: jax.Array,
    *,
    trace_axis0: bool = True,
) -> tuple[onnx.ModelProto, ir.Model]:
    inputs: List[Any] = [prim0, t_arr, dt_arr]
    to_kwargs: Dict[str, Any] = {
        "inputs": inputs,
        "model_name": "feed_forward_step",
        "enable_double_precision": True,
        "return_mode": "ir",
    }
    try:
        ir_model = to_onnx(ff, **to_kwargs)
    except TypeError:
        to_kwargs.pop("enable_double_precision", None)
        ir_model = to_onnx(ff, **to_kwargs)

    if trace_axis0:
        _trace_axis0(ir_model)

    overrides: AxisOverrideMap = {}
    _collect_axis0_overrides(ir_model.graph, overrides)

    model_proto = ir.to_proto(ir_model)
    _restamp_onnx_axis0(model_proto.graph, overrides)
    ONNX_PATH.write_bytes(model_proto.SerializeToString())
    print(f"[INFO] ONNX payload written to {ONNX_PATH}")
    return model_proto, ir_model


def export_models(
    trace_axis0: bool = False,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return ``(onnx_model, ir_model, prim0, t_arr, dt_arr)`` for tests."""
    closed, prim0, t_arr, dt_arr = _load_payload()
    ff = _feed_forward_fn(closed)
    model_proto, ir_model = _export_to_onnx(
        ff, prim0, t_arr, dt_arr, trace_axis0=trace_axis0
    )
    return model_proto, ir_model, prim0, t_arr, dt_arr


def _run_onnx(prim0: jax.Array, t_arr: jax.Array, dt_arr: jax.Array) -> None:
    try:
        sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    except Fail as err:
        message = str(err)
        if "ScatterElements" in message or "Incompatible dimensions" in message:
            print("[EXPECTED] onnxruntime failure triggered during session creation:")
            print(f"          {message}")
            return
        raise
    feeds = {
        sess.get_inputs()[0].name: np.asarray(prim0, dtype=np.float64),
        sess.get_inputs()[1].name: np.asarray(t_arr, dtype=np.float64),
        sess.get_inputs()[2].name: np.asarray(dt_arr, dtype=np.float64),
    }

    try:
        sess.run(None, feeds)
    except (InvalidArgument, Fail) as err:
        message = str(err)
        if "ScatterElements" in message or "Incompatible dimensions" in message:
            print("[EXPECTED] onnxruntime failure triggered:")
            print(f"          {message}")
            return
        raise

    raise AssertionError(
        "onnxruntime unexpectedly succeeded – jax2onnx issue #52 appears fixed"
    )


def main() -> int:
    closed, prim0, t_arr, dt_arr = _load_payload()
    ff = _feed_forward_fn(closed)
    model_proto, _ = _export_to_onnx(ff, prim0, t_arr, dt_arr)
    _run_onnx(prim0, t_arr, dt_arr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _sanitize_params(params: Any) -> Any:
    if isinstance(params, Mapping):
        sanitized = dict(params.items())
    elif hasattr(params, "items"):
        sanitized = dict(params.items())
    else:
        return params
    sanitized.pop("copy_semantics", None)
    return sanitized
