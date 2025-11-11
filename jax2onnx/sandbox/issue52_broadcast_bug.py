# jax2onnx/sandbox/issue52_broadcast_bug.py

"""
Issue #52 broadcast metadata smoke-test.

Exports a tiny nested scan, reports the loop/broadcast metadata, and (if requested)
forces the `1 × …` restamp so onnxruntime reproduces the mismatch. The converter is
expected to emit the correct leading extent automatically, so by default the script
simply verifies the metadata and exits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper

from jax2onnx import to_onnx

STACK_WIDTH = 5
OUTPUT_PATH = Path("issue52_broadcast_bug.onnx")

os.environ.setdefault("JAX2ONNX_ENABLE_STACKTRACE_METADATA", "1")
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


def export_model(path: Path | None = None) -> onnx.ModelProto:
    model = to_onnx(
        _model,
        inputs=_inputs(),
        enable_double_precision=True,
        opset=21,
        model_name="issue52_broadcast",
    )
    model.ir_version = min(model.ir_version, 11)
    if path is not None:
        onnx.save(model, path)
    return model


def export_ir_model() -> ir.Model:
    return to_onnx(
        _model,
        inputs=_inputs(),
        enable_double_precision=True,
        opset=21,
        model_name="issue52_broadcast",
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


def _value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    mapping: dict[str, onnx.ValueInfoProto] = {}
    for graph in _walk_graphs(model.graph):
        for vi in graph.value_info:
            mapping[vi.name] = vi
        for output in graph.output:
            mapping[output.name] = output
    return mapping


def _dims_for(
    name: str, vi_map: dict[str, onnx.ValueInfoProto]
) -> list[int | str | None]:
    vi = vi_map.get(name)
    if vi is None or not vi.type.HasField("tensor_type"):
        return []
    dims: list[int | str | None] = []
    for dim in vi.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def dims_for(name: str, model: onnx.ModelProto) -> list[int | str | None]:
    return _dims_for(name, _value_info_map(model))


def _stamp_dim(vi: onnx.ValueInfoProto, dim0: int) -> None:
    tensor_type = vi.type.tensor_type
    if tensor_type is None:
        return
    shape = tensor_type.shape
    if not shape.dim:
        shape.dim.add()
    dim_proto = shape.dim[0]
    dim_proto.ClearField("dim_param")
    dim_proto.dim_value = dim0


def _tensor_elem_type(name: str, vi_map: dict[str, onnx.ValueInfoProto]) -> int:
    vi = vi_map.get(name)
    if vi is None or not vi.type.HasField("tensor_type"):
        return TensorProto.FLOAT
    elem_type = vi.type.tensor_type.elem_type
    return elem_type if elem_type != 0 else TensorProto.FLOAT


def _inject_runtime_failure(
    model: onnx.ModelProto, vi_map: dict[str, onnx.ValueInfoProto]
) -> None:
    elem_type = _tensor_elem_type("jnp_concat_out_0", vi_map)
    reshape_const = helper.make_tensor(
        "reshape_target",
        TensorProto.INT64,
        [4],
        np.array([1, 210, 1, 1], dtype=np.int64),
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["jnp_concat_out_0", "reshape_target"],
        outputs=["reshape_to_bad_out"],
        name="reshape_to_bad",
    )
    model.graph.initializer.append(reshape_const)
    model.graph.node.append(reshape_node)
    model.graph.output.clear()
    model.graph.output.append(
        helper.make_tensor_value_info("reshape_to_bad_out", elem_type, [1, 210, 1, 1])
    )


def loop_axis_override() -> int | None:
    ir_model = export_ir_model()
    loop_node = next(
        node for node in ir_model.graph.all_nodes() if node.op_type == "Loop"
    )
    return loop_node.outputs[1].meta.get("loop_axis0_override")


def metadata_ok(model: onnx.ModelProto | None = None) -> bool:
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


def main() -> None:
    OUTPUT_PATH.unlink(missing_ok=True)
    model = export_model()
    vi_map = _value_info_map(model)

    print(f"[INFO] loop_out_0 metadata       : {_dims_for('loop_out_0', vi_map)}")
    print(f"[INFO] bcast_out_0 metadata (pre): {_dims_for('bcast_out_0', vi_map)}")

    if metadata_ok(model):
        print("[INFO] No suspicious value infos detected; metadata appears fixed.")
        return

    _stamp_dim(vi_map["bcast_out_0"], dim0=1)
    _stamp_dim(vi_map["jnp_concat_out_0"], dim0=1)
    _inject_runtime_failure(model, vi_map)
    onnx.save(model, OUTPUT_PATH)
    print(f"[INFO] Wrote patched model to {OUTPUT_PATH.resolve()}")

    try:
        session = ort.InferenceSession(
            str(OUTPUT_PATH), providers=["CPUExecutionProvider"]
        )
        feeds = {}
        for inp in session.get_inputs():
            dtype = np.float64 if inp.type == "tensor(double)" else np.float32
            shape = [
                dim if isinstance(dim, int) and dim > 0 else STACK_WIDTH
                for dim in inp.shape
            ]
            feeds[inp.name] = np.ones(shape, dtype=dtype)
        session.run(None, feeds)
    except Exception as err:
        print("[EXPECTED] onnxruntime failure triggered:")
        print(err)
    else:
        print(
            "[WARN] onnxruntime succeeded; adjust the script to tighten the mismatch."
        )


if __name__ == "__main__":
    main()
