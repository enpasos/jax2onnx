# tests/extra_tests/helpers/issue52_scatter_payload_fixture.py

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from jax._src import core, source_info_util
from onnx import AttributeProto
import onnx_ir as ir

from jax2onnx import to_onnx


jax.config.update("jax_enable_x64", True)

_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = _ROOT / "jax2onnx" / "sandbox"
PAYLOAD_PATH = DATA_DIR / "issue52_feedforward_payload.npz"


@dataclass
class ArrayLoader:
    arrays: Dict[str, np.ndarray]

    def get(self, ref: str) -> np.ndarray:
        return np.asarray(self.arrays[ref])


def _import_module(module_name: str):
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


def _deserialize_aval(desc: Dict[str, Any]) -> core.AbstractValue:
    if desc["type"] == "ShapedArray":
        dtype = None if desc["dtype"] is None else np.dtype(desc["dtype"])
        return core.ShapedArray(
            tuple(desc["shape"]), dtype, desc.get("weak_type", False)
        )
    if desc["type"] == "AbstractToken":
        return core.AbstractToken()
    raise TypeError(f"Unsupported aval description: {desc}")


def _deserialize_var(desc: Dict[str, Any], var_map: Dict[str, core.Var]) -> core.Var:
    name = desc["name"]
    if name in var_map:
        return var_map[name]
    aval_desc = desc.get("aval")
    if not isinstance(aval_desc, dict):
        raise TypeError(f"Unexpected aval descriptor for {name!r}: {aval_desc!r}")
    aval = _deserialize_aval(aval_desc)
    try:
        var = core.Var(aval)
    except TypeError:
        var = core.Var(name, aval)
    var_map[name] = var
    return var


def _deserialize_literal(desc: Dict[str, Any], loader: ArrayLoader) -> core.Literal:
    aval = _deserialize_aval(desc["aval"])
    value_desc = desc["value"]
    if value_desc["kind"] == "array":
        val = loader.get(value_desc["ref"])
    else:
        val = value_desc["value"]
    return core.Literal(val, aval)


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
    desc: Dict[str, Any], loader: ArrayLoader, var_map: Dict[str, core.Var]
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
    desc: Dict[str, Any], loader: ArrayLoader, var_map: Dict[str, core.Var]
) -> core.JaxprEqn:
    primitive = _primitive_registry()[desc["primitive"]]
    invars = [_deserialize_atom(atom, loader, var_map) for atom in desc["invars"]]
    outvars = [_deserialize_var(var, var_map) for var in desc["outvars"]]
    params = _deserialize_value(desc["params"], loader)
    return core.new_jaxpr_eqn(
        invars,
        outvars,
        primitive,
        params,
        effects=(),
        source_info=source_info_util.new_source_info(),
    )


def _deserialize_jaxpr(desc: Dict[str, Any], loader: ArrayLoader) -> core.Jaxpr:
    var_map: Dict[str, core.Var] = {}
    constvars = [_deserialize_var(var, var_map) for var in desc["constvars"]]
    invars = [_deserialize_var(var, var_map) for var in desc["invars"]]
    outvars = [_deserialize_var(var, var_map) for var in desc["outvars"]]
    eqns = [_deserialize_eqn(eqn, loader, var_map) for eqn in desc["eqns"]]
    return core.Jaxpr(constvars, invars, outvars, eqns)


def _deserialize_closed_jaxpr(
    desc: Dict[str, Any], loader: ArrayLoader
) -> core.ClosedJaxpr:
    jaxpr = _deserialize_jaxpr(desc["jaxpr"], loader)
    consts = [_deserialize_value(c, loader) for c in desc["consts"]]
    return core.ClosedJaxpr(jaxpr, consts)


@lru_cache(maxsize=1)
def _primitive_registry() -> Dict[str, core.Primitive]:
    def _collect(module: Any, registry: Dict[str, core.Primitive]) -> None:
        for attr in getattr(module, "__dict__", {}).values():
            if isinstance(attr, core.Primitive):
                registry.setdefault(attr.name, attr)

    registry: Dict[str, core.Primitive] = {}
    for module in list(sys.modules.values()):
        if module is None:
            continue
        name = getattr(module, "__name__", "")
        if not name.startswith("jax"):
            continue
        _collect(module, registry)

    safe_modules = (
        "jax",
        "jax.core",
        "jax.lax",
        "jax.numpy",
        "jax.scipy",
        "jax._src.lax.lax",
        "jax._src.lax.control_flow",
        "jax._src.lax.parallel",
        "jax._src.lax.slicing",
        "jax._src.lax.lax_control_flow",
        "jax._src.numpy.lax_numpy",
        "jax._src.numpy.reductions",
        "jax._src.nn.functions",
    )
    for module_name in safe_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        _collect(module, registry)

    try:
        lax_impl = importlib.import_module("jax._src.lax.lax")
    except Exception:
        lax_impl = None
    if lax_impl is not None:
        _collect(lax_impl, registry)

    for attr in core.__dict__.values():
        if isinstance(attr, core.Primitive):
            registry.setdefault(attr.name, attr)
    return registry


def _load_payload():
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


def _feed_forward_fn(closed: core.ClosedJaxpr):
    def ff(y_current, t_arr, dt_arr):
        return core.eval_jaxpr(closed.jaxpr, closed.consts, y_current, t_arr, dt_arr)

    return ff


def _axis0_override(value: Any) -> Optional[int]:
    meta = getattr(value, "meta", None)
    if meta is None:
        return None
    maybe = meta.get("loop_axis0_override")
    if isinstance(maybe, (int, np.integer)):
        return int(maybe)
    return None


def _collect_axis0_overrides(
    graph: Any, overrides: Dict[str, Tuple[int, Optional[str]]]
) -> None:
    for node in graph.all_nodes():
        for out in getattr(node, "outputs", ()):
            override = _axis0_override(out)
            if isinstance(override, int) and override > 1:
                producer = out.producer() if hasattr(out, "producer") else None
                op_type = getattr(producer, "op_type", None)
                overrides.setdefault(out.name, (override, op_type))
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


def _restamp_onnx_axis0(
    graph: onnx.GraphProto, overrides: Dict[str, Tuple[int, Optional[str]]]
) -> None:
    allowed_ops = {"Expand", "Mul", "Div", "Add", "Sub", "ScatterND"}

    def _apply(value_info: onnx.ValueInfoProto) -> None:
        data = overrides.get(value_info.name)
        if data is None:
            return
        override, op_type = data
        if op_type not in allowed_ops or not isinstance(override, int) or override <= 1:
            return
        tensor_type = value_info.type.tensor_type
        if tensor_type is None or not tensor_type.shape.dim:
            return
        dim0 = tensor_type.shape.dim[0]
        if dim0.HasField("dim_value") and dim0.dim_value not in (0, override, 1):
            return
        dim0.ClearField("dim_param")
        dim0.dim_value = override

    for collection in (graph.value_info, graph.output, graph.input):
        for value_info in collection:
            _apply(value_info)

    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                _restamp_onnx_axis0(attr.g, overrides)
            elif attr.type == AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    _restamp_onnx_axis0(subgraph, overrides)


def export_models(
    trace_axis0: bool = False,
) -> Tuple[Any, Any, Any, Any, Any]:
    closed, prim0, t_arr, dt_arr = _load_payload()
    ff = _feed_forward_fn(closed)
    inputs: List[Any] = [prim0, t_arr, dt_arr]
    kwargs: Dict[str, Any] = {
        "inputs": inputs,
        "model_name": "feed_forward_step",
        "enable_double_precision": True,
        "return_mode": "ir",
    }
    try:
        ir_model = to_onnx(ff, **kwargs)
    except TypeError:
        kwargs.pop("enable_double_precision", None)
        ir_model = to_onnx(ff, **kwargs)

    if trace_axis0:
        # The original sandbox prints a detailed trace; tests only care about the side
        # effects of restamping, so we skip verbose logging here.
        pass

    overrides: Dict[str, Tuple[int, Optional[str]]] = {}
    _collect_axis0_overrides(ir_model.graph, overrides)

    model_proto = ir.to_proto(ir_model)
    _restamp_onnx_axis0(model_proto.graph, overrides)
    return model_proto, ir_model, prim0, t_arr, dt_arr


__all__ = [
    "_feed_forward_fn",
    "_load_payload",
    "_primitive_registry",
    "export_models",
]
