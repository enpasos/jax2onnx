# file: jax2onnx/sandbox/issue52_scatter_payload_repro.py


"""Reproduce jax2onnx issue #52 without jaxfluids dependencies."""

from __future__ import annotations

import importlib
import json
import pathlib
import pkgutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
from jax._src import core, source_info_util
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument

from jax2onnx import to_onnx

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
        return core.ShapedArray(tuple(desc["shape"]), dtype, weak_type=desc.get("weak_type", False))
    if desc["type"] == "AbstractToken":
        return core.AbstractToken()
    raise TypeError(f"Unsupported aval description: {desc}")


def _deserialize_var(desc: Dict[str, Any], var_map: Dict[str, core.Var]) -> core.Var:
    name = desc["name"]
    if name in var_map:
        return var_map[name]
    aval_desc = desc.get("aval")
    if not isinstance(aval_desc, dict):  # pragma: no cover - debug aid
        raise TypeError(f"Unexpected aval descriptor for {name!r}: {aval_desc!r}")
    aval = _deserialize_aval(aval_desc)
    if not isinstance(aval, core.AbstractValue):  # pragma: no cover - debug safeguard
        raise TypeError(f"Invalid aval for var {name!r}: {aval!r}")
    var = core.Var(aval)
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
            return cls(*values)
        if kind == "enum":
            enum_cls = getattr(_import_module(desc["module"]), desc["name"])
            return enum_cls[desc["member"]]
        if kind == "dtype":
            return np.dtype(desc["value"])
        raise TypeError(f"Unsupported descriptor: {desc}")
    if isinstance(desc, dict):
        return {k: _deserialize_value(v, loader) for k, v in desc.items()}
    return desc


def _deserialize_atom(desc: Dict[str, Any], loader: ArrayLoader, var_map: Dict[str, core.Var]) -> core.Atom:
    kind = desc["kind"]
    if kind == "var":
        name = desc["name"]
        if name not in var_map:
            raise KeyError(f"Variable '{name}' referenced before definition")
        return var_map[name]
    if kind == "literal":
        return _deserialize_literal(desc, loader)
    raise TypeError(f"Unsupported atom kind: {kind}")


def _deserialize_eqn(desc: Dict[str, Any], loader: ArrayLoader, var_map: Dict[str, core.Var]) -> core.JaxprEqn:
    primitive = _primitive_registry()[desc["primitive"]]
    invars = [_deserialize_atom(atom, loader, var_map) for atom in desc["invars"]]
    outvars = [_deserialize_var(var, var_map) for var in desc["outvars"]]
    params = _deserialize_value(desc["params"], loader)
    return core.new_jaxpr_eqn(invars, outvars, primitive, params, effects=(), source_info=source_info_util.new_source_info())


def _deserialize_jaxpr(desc: Dict[str, Any], loader: ArrayLoader) -> core.Jaxpr:
    var_map: Dict[str, core.Var] = {}
    constvars = [_deserialize_var(var, var_map) for var in desc["constvars"]]
    invars = [_deserialize_var(var, var_map) for var in desc["invars"]]
    outvars = [_deserialize_var(var, var_map) for var in desc["outvars"]]
    eqns = [_deserialize_eqn(eqn, loader, var_map) for eqn in desc["eqns"]]
    return core.Jaxpr(constvars, invars, outvars, eqns)


@lru_cache(maxsize=1)
def _primitive_registry() -> Dict[str, core.Primitive]:
    registry: Dict[str, core.Primitive] = {}
    for _, module_name, _ in pkgutil.walk_packages(jax.__path__, prefix="jax."):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attr in module.__dict__.values():
            if isinstance(attr, core.Primitive):
                registry.setdefault(attr.name, attr)
    # Ensure core module primitives are included
    for attr in core.__dict__.values():
        if isinstance(attr, core.Primitive):
            registry.setdefault(attr.name, attr)
    return registry


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


def _deserialize_closed_jaxpr(desc: Dict[str, Any], loader: ArrayLoader) -> core.ClosedJaxpr:
    jaxpr = _deserialize_jaxpr(desc["jaxpr"], loader)
    consts = [_deserialize_value(c, loader) for c in desc["consts"]]
    return core.ClosedJaxpr(jaxpr, consts)


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


def _export_to_onnx(ff, prim0, t_arr, dt_arr) -> None:
    kwargs = dict(inputs=[prim0, t_arr, dt_arr], model_name="feed_forward_step")
    try:
        model = to_onnx(ff, enable_double_precision=True, **kwargs)
    except TypeError:
        model = to_onnx(ff, **kwargs)
    ONNX_PATH.write_bytes(model.SerializeToString())
    print(f"[INFO] ONNX payload written to {ONNX_PATH}")


def _run_onnx(prim0, t_arr, dt_arr) -> None:
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
    _export_to_onnx(ff, prim0, t_arr, dt_arr)
    _run_onnx(prim0, t_arr, dt_arr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
