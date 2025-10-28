# tests/extra_tests/helpers/issue52_broadcast_fixture.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
from onnx import AttributeProto

from jax2onnx import to_onnx


STACK_WIDTH = 5
_MODEL_NAME = "issue52_broadcast"
_IR_VERSION_CAP = 11

jax.config.update("jax_enable_x64", True)


def _stack_block(state: jax.Array) -> jax.Array:
    rho = state[0]
    vel = state[1:4]
    energy = state[4]

    momentum = rho * vel
    vel_sq = jnp.square(vel)
    sum_sq = jnp.sum(vel_sq, axis=0)
    specific = energy / (rho * 0.4)
    big_e = rho * (0.5 * sum_sq + specific)

    return jnp.stack([rho, momentum[0], momentum[1], momentum[2], big_e], axis=0)


def _inner_scan(state: jax.Array):
    def body(carry, _):
        stacked = _stack_block(carry)
        return carry, stacked

    carry, scans = jax.lax.scan(body, state, xs=None, length=2)
    return carry, scans[-1]


def _model(state: jax.Array, t_arr: jax.Array, dt_arr: jax.Array):
    def body(carry, _):
        new_carry, stacked = _inner_scan(carry)
        return new_carry, stacked

    _, seq = jax.lax.scan(body, state, xs=None, length=1)
    squeezed = seq[0]
    filler = jnp.broadcast_to(squeezed[:1], (STACK_WIDTH, 210, 1, 1))
    return jnp.concatenate([filler, squeezed], axis=0), t_arr + dt_arr


def _inputs() -> list[jax.ShapeDtypeStruct]:
    return [
        jax.ShapeDtypeStruct((STACK_WIDTH, 210, 1, 1), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.float64),
    ]


def export_model(path: Optional[Path] = None) -> onnx.ModelProto:
    model = to_onnx(
        _model,
        inputs=_inputs(),
        enable_double_precision=True,
        opset=21,
        model_name=_MODEL_NAME,
    )
    try:
        model.ir_version = min(model.ir_version, _IR_VERSION_CAP)
    except Exception:
        pass
    if path is not None:
        onnx.save(model, path)
    return model


def export_ir_model() -> ir.Model:
    return to_onnx(
        _model,
        inputs=_inputs(),
        enable_double_precision=True,
        opset=21,
        model_name=_MODEL_NAME,
        return_mode="ir",
    )


def _walk_graphs(graph: onnx.GraphProto) -> Iterable[onnx.GraphProto]:
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                yield from _walk_graphs(attr.g)
            elif attr.type == AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    yield from _walk_graphs(sub)


def _value_info_map(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    mapping: Dict[str, onnx.ValueInfoProto] = {}
    for graph in _walk_graphs(model.graph):
        for vi in graph.value_info:
            mapping[vi.name] = vi
        for output in graph.output:
            mapping[output.name] = output
    return mapping


def _dims_for(name: str, vi_map: Dict[str, onnx.ValueInfoProto]) -> List[int | str | None]:
    vi = vi_map.get(name)
    if vi is None or not vi.type.HasField("tensor_type"):
        return []
    dims: List[int | str | None] = []
    for dim in vi.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def dims_for(name: str, model: onnx.ModelProto) -> List[int | str | None]:
    return _dims_for(name, _value_info_map(model))


def loop_axis_override() -> Optional[int]:
    ir_model = export_ir_model()
    loop_node = next(
        node for node in ir_model.graph.all_nodes() if node.op_type == "Loop"
    )
    return loop_node.outputs[1].meta.get("loop_axis0_override")


def metadata_ok(model: Optional[onnx.ModelProto] = None) -> bool:
    if model is None:
        model = export_model()
    vi_map = _value_info_map(model)
    bcast_dims = _dims_for("bcast_out_0", vi_map)
    concat_dims = _dims_for("jnp_concat_out_0", vi_map)
    loop_dims = _dims_for("loop_out_0", vi_map)
    override = loop_axis_override()

    if not bcast_dims or bcast_dims[0] not in (
        STACK_WIDTH,
        "JAX2ONNX_DYNAMIC_DIM_SENTINEL",
    ):
        return False
    if not concat_dims or concat_dims[0] != STACK_WIDTH * 2:
        return False
    if override not in (STACK_WIDTH,):
        return False
    if loop_dims and isinstance(loop_dims[0], int) and loop_dims[0] != STACK_WIDTH:
        return False
    return True
