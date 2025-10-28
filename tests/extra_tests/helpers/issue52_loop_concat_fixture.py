# tests/extra_tests/helpers/issue52_loop_concat_fixture.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
from onnx import AttributeProto

from jax2onnx import to_onnx


STACK_WIDTH = 5
_MODEL_NAME = "issue52_loop_concat"

jax.config.update("jax_enable_x64", True)


def _stack_block(state: jax.Array) -> jax.Array:
    rho = state[0]
    vel = state[1:4]
    energy = state[4]

    momentum = rho * vel
    vel_sq = jnp.square(vel)
    sum_sq = vel_sq[0] + vel_sq[1] + vel_sq[2]
    specific = energy / (rho * 0.4)
    big_e = rho * (0.5 * sum_sq + specific)

    comps = [
        rho,
        momentum[0],
        momentum[1],
        momentum[2],
        big_e,
    ]
    return jnp.stack(comps, axis=0)


def _inner_scan(state: jax.Array):
    def body(carry, _):
        stacked = _stack_block(carry)
        return carry, stacked

    carry, scans = jax.lax.scan(body, state, xs=None, length=2)
    return carry, scans[-1]


def _model_fn(state: jax.Array, t_arr: jax.Array, dt_arr: jax.Array):
    def body(carry, _):
        new_carry, stacked = _inner_scan(carry)
        return new_carry, stacked

    _, seq = jax.lax.scan(body, state, xs=None, length=1)
    squeezed = seq[0]
    filler = jnp.broadcast_to(squeezed[:1], (STACK_WIDTH, 210, 1, 1))
    bad_concat = jnp.concatenate([filler, squeezed], axis=0)
    return bad_concat, t_arr + dt_arr


def _inputs() -> list[jax.ShapeDtypeStruct]:
    return [
        jax.ShapeDtypeStruct((STACK_WIDTH, 210, 1, 1), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.float64),
    ]


def export_model(path: Optional[Path] = None) -> onnx.ModelProto:
    model = to_onnx(
        _model_fn,
        inputs=_inputs(),
        enable_double_precision=True,
        model_name=_MODEL_NAME,
        opset=21,
    )
    if path is not None:
        onnx.save(model, path)
    return model


def export_ir_model() -> ir.Model:
    return to_onnx(
        _model_fn,
        inputs=_inputs(),
        enable_double_precision=True,
        model_name=_MODEL_NAME,
        opset=21,
        return_mode="ir",
    )


def _enumerate_graphs(graph: onnx.GraphProto) -> Iterable[onnx.GraphProto]:
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                yield from _enumerate_graphs(attr.g)
            elif attr.type == AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    yield from _enumerate_graphs(subgraph)


def dims_for(name: str, model: onnx.ModelProto) -> list[int | str | None]:
    for graph in _enumerate_graphs(model.graph):
        for vi in graph.value_info:
            if vi.name != name:
                continue
            dims: list[int | str | None] = []
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                elif dim.HasField("dim_param"):
                    dims.append(dim.dim_param)
                else:
                    dims.append(None)
            return dims
    return []


def loop_axis_override() -> Optional[int]:
    ir_model = export_ir_model()
    loop_node = next(
        node for node in ir_model.graph.all_nodes() if node.op_type == "Loop"
    )
    override = loop_node.outputs[1].meta.get("loop_axis0_override")
    if isinstance(override, (int, np.integer)):
        return int(override)
    return None


def metadata_ok(model: Optional[onnx.ModelProto] = None) -> bool:
    if model is None:
        model = export_model()
    squeeze_dims = dims_for("squeeze_out_0", model)
    override = loop_axis_override()
    squeeze_ok = bool(squeeze_dims) and squeeze_dims[0] == STACK_WIDTH
    override_ok = override == STACK_WIDTH
    return squeeze_ok and override_ok


__all__ = [
    "STACK_WIDTH",
    "dims_for",
    "export_ir_model",
    "export_model",
    "loop_axis_override",
    "metadata_ok",
]
