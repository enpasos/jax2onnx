# jax2onnx/sandbox/issue52_loop_concat_bug.py

"""
Minimal repro for the Issue 52 loop/concat metadata bug.

Historically the loop output recorded its axis-0 extent as ``1`` even though the
squeezed tensor that feeds the concat carries the true stack width of ``5``.
The new scan lowering now keeps that extent in the IR metadata so downstream
ops (and the debugger) can recover it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from jax2onnx import to_onnx

STACK_WIDTH = 5
OUTER_SCAN_STEPS = 1
INNER_SCAN_STEPS = 2
OUTPUT_PATH = Path("issue52_loop_concat_bug.onnx")

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


def _inner_scan(state: jax.Array) -> Tuple[jax.Array, jax.Array]:
    def body(carry, _):
        stacked = _stack_block(carry)
        return carry, stacked

    carry, scans = jax.lax.scan(body, state, xs=None, length=INNER_SCAN_STEPS)
    return carry, scans[-1]


def _outer_scan(
    state: jax.Array, t_arr: jax.Array, dt_arr: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    def body(carry, _):
        new_carry, stacked = _inner_scan(carry)
        return new_carry, stacked

    final, seq = jax.lax.scan(body, state, xs=None, length=OUTER_SCAN_STEPS)
    return final, seq


def _model_fn(
    state: jax.Array, t_arr: jax.Array, dt_arr: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    final_state, seq = _outer_scan(state, t_arr, dt_arr)
    squeezed = seq[0]  # (STACK_WIDTH, 210, 1, 1)
    filler = jnp.broadcast_to(squeezed[:1], (STACK_WIDTH, 210, 1, 1))
    bad_concat = jnp.concatenate([filler, squeezed], axis=0)
    return bad_concat, t_arr + dt_arr


def _inputs() -> list[jax.ShapeDtypeStruct]:
    return [
        jax.ShapeDtypeStruct((STACK_WIDTH, 210, 1, 1), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.float64),
    ]


def export_model(path: Path | None = None) -> onnx.ModelProto:
    model = to_onnx(
        _model_fn,
        inputs=_inputs(),
        enable_double_precision=True,
        model_name="issue52_loop_concat",
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
        model_name="issue52_loop_concat",
        opset=21,
        return_mode="ir",
    )


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


def _enumerate_graphs(graph: onnx.GraphProto) -> Iterable[onnx.GraphProto]:
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _enumerate_graphs(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    yield from _enumerate_graphs(subgraph)


def loop_axis_override() -> int | None:
    ir_model = export_ir_model()
    loop_node = next(
        node for node in ir_model.graph.all_nodes() if node.op_type == "Loop"
    )
    return loop_node.outputs[1].meta.get("loop_axis0_override")


def metadata_ok(model: onnx.ModelProto | None = None) -> bool:
    if model is None:
        model = export_model()
    squeeze_dims = dims_for("squeeze_out_0", model)
    override = loop_axis_override()
    squeeze_ok = bool(squeeze_dims) and squeeze_dims[0] == STACK_WIDTH
    override_ok = override == STACK_WIDTH
    return squeeze_ok and override_ok


def _inject_runtime_failure(model: onnx.ModelProto) -> onnx.ModelProto:
    """Force the historical failure by clobbering metadata and adding a reshape."""
    target_dims = [1, 210, 1, 1]
    for graph in _enumerate_graphs(model.graph):
        for vi in graph.value_info:
            if vi.name != "loop_out_0":
                continue
            shape = vi.type.tensor_type.shape
            shape.ClearField("dim")
            for dim in target_dims:
                shape.dim.add().dim_value = dim

    reshape_const = helper.make_tensor(
        "reshape_target",
        TensorProto.INT64,
        [4],
        np.array([1, 210, 1, 1], dtype=np.int64),
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["loop_out_0", "reshape_target"],
        outputs=["reshape_to_bad_out"],
        name="reshape_to_bad",
    )
    model.graph.initializer.append(reshape_const)
    model.graph.node.append(reshape_node)
    model.graph.output.clear()
    model.graph.output.append(
        helper.make_tensor_value_info(
            "reshape_to_bad_out", TensorProto.FLOAT, [1, 210, 1, 1]
        )
    )
    return model


def main() -> None:
    OUTPUT_PATH.unlink(missing_ok=True)
    model = export_model(OUTPUT_PATH)
    print(f"[INFO] Exported ONNX model to {OUTPUT_PATH.resolve()}")

    loop_dims = dims_for("loop_out_0", model)
    squeeze_dims = dims_for("squeeze_out_0", model)
    override = loop_axis_override()
    print(f"[CHECK] loop_out_0 dims: {loop_dims}")
    print(f"[CHECK] squeeze_out_0 dims: {squeeze_dims}")
    print(f"[CHECK] loop_axis0_override: {override}")

    if metadata_ok(model):
        print("[PASS] Loop metadata preserves the 5-wide stack extent.")
    else:
        print("[FAIL] Loop metadata lost the stack width; forcing runtime failure.")
        failing = _inject_runtime_failure(model)
        failing.ir_version = min(failing.ir_version, 10)
        onnx.save(failing, OUTPUT_PATH)
        try:
            sess = ort.InferenceSession(
                str(OUTPUT_PATH), providers=["CPUExecutionProvider"]
            )
            feeds = {
                inp.name: np.ones(
                    tuple(
                        dim if isinstance(dim, int) and dim > 0 else STACK_WIDTH
                        for dim in inp.shape
                    ),
                    dtype=np.float64,
                )
                for inp in sess.get_inputs()
            }
            sess.run(None, feeds)
        except Exception as err:
            print("[EXPECTED] onnxruntime failed to load the forced-mismatch model:")
            print(err)


if __name__ == "__main__":
    main()
