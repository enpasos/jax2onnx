# jax2onnx/sandbox/issue52_stack_bug.py

"""
issue52_stack_bug.py
--------------------

Self-contained repro of the shape-mismatch crash that originally showed up in
the feed-forward export.  This version keeps the JAX function tiny (no
dependencies on jaxfluids) but still walks through the real `jax2onnx`
conversion pipeline so we can inspect metadata and hand over a failing ONNX
artefact.

Workflow:
1. Define a small nested `jax.lax.scan` that mimics the stack/unsqueeze pattern.
2. Export it with the in-repo `jax2onnx` (stacktrace metadata enabled).
3. Post-process the resulting ModelProto to mirror the erroneous metadata
   observed in the wild: the concat output is declared as `1×210×1×1`, even
   though the inferred shape is `5×210×1×1`.
4. Save the ONNX file and attempt to load it with onnxruntime – this produces
   the same failure (`ShapeInferenceError` on `node_Concat_*`) that we see in
   the full feed-forward export.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper

# Use the local development checkout of jax2onnx.
sys.path.insert(0, "/home/enpasos/projects/jax2onnx/src")
from jax2onnx import to_onnx  # noqa: E402

TARGET_ONNX = Path("issue52_stack_bug.onnx")

# Keep stacktrace metadata so we can see provenance in Netron.
os.environ.setdefault("JAX2ONNX_ENABLE_STACKTRACE_METADATA", "1")
jax.config.update("jax_enable_x64", True)


def _stack_block(state: jax.Array) -> jax.Array:
    """Builds the 5-wide stack used in the failing graph."""
    rho = state[0]
    vel = state[1:4]
    energy = state[4]

    momentum = rho * vel
    vel_sq = jnp.square(vel)
    sum_sq = vel_sq[0] + vel_sq[1] + vel_sq[2]
    specific = energy / (rho * 0.4)
    big_e = rho * (0.5 * sum_sq + specific)

    comps = [
        jnp.expand_dims(rho, axis=0),
        jnp.expand_dims(momentum[0], axis=0),
        jnp.expand_dims(momentum[1], axis=0),
        jnp.expand_dims(momentum[2], axis=0),
        jnp.expand_dims(big_e, axis=0),
    ]
    return jnp.concatenate(comps, axis=0)


def _inner_scan(state: jax.Array) -> tuple[jax.Array, jax.Array]:
    def body(carry, _):
        stacked = _stack_block(carry)
        return carry, stacked

    carry, scans = jax.lax.scan(body, state, xs=None, length=2)
    return carry, scans[-1]


def _outer_scan(
    state: jax.Array, t_arr: jax.Array, dt_arr: jax.Array
) -> tuple[jax.Array, jax.Array]:
    def body(carry, _):
        new_carry, stacked = _inner_scan(carry)
        return new_carry, stacked

    final, _ = jax.lax.scan(body, state, xs=None, length=2)
    return final, t_arr + dt_arr


def _lower_to_onnx() -> onnx.ModelProto:
    inputs = [
        jax.ShapeDtypeStruct((5, 210, 1, 1), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.float64),
    ]
    return to_onnx(_outer_scan, inputs=inputs)


def _enumerate_subgraphs(graph: onnx.GraphProto) -> Iterable[onnx.GraphProto]:
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                yield from _enumerate_subgraphs(attr.g)
            elif attr.type == AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    yield from _enumerate_subgraphs(subgraph)


def _force_concat_metadata(model: onnx.ModelProto) -> bool:
    """Clamp stack outputs to 1×210×1×1 to mirror the faulty metadata.

    Returns True iff any concat metadata was adjusted (i.e. the bug is still present).
    """
    target_shape = [1, 210, 1, 1]
    expected_head = 5
    mutated = False
    for graph in _enumerate_subgraphs(model.graph):
        for vi in graph.value_info:
            if "stack_out" in vi.name:
                shape = vi.type.tensor_type.shape
                current_dims: list[int | str | None] = []
                for dim in shape.dim:
                    if dim.HasField("dim_value"):
                        current_dims.append(dim.dim_value)
                    elif dim.HasField("dim_param"):
                        current_dims.append(dim.dim_param)
                    else:
                        current_dims.append(None)
                dim0 = current_dims[0] if current_dims else None
                if isinstance(dim0, int) and dim0 == expected_head:
                    continue  # already correct
                if dim0 is None or not isinstance(dim0, int):
                    # Dynamic metadata (sentinel / symbolic) — keep as-is.
                    continue
                mutated = True
                shape.ClearField("dim")
                for dim in target_shape:
                    shape.dim.add().dim_value = dim
    return mutated


def _inject_runtime_check(model: onnx.ModelProto) -> None:
    """Append a reshape layer so the mismatch manifests as a runtime error."""
    loop_node = model.graph.node[0]
    reshape_const = helper.make_tensor(
        "reshape_target",
        TensorProto.INT64,
        [4],
        np.array([1, 210, 1, 1], dtype=np.int64),
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[loop_node.output[0], "reshape_target"],
        outputs=["reshape_out"],
        name="reshape_to_bad",
    )
    model.graph.initializer.append(reshape_const)
    model.graph.node.extend([reshape_node])
    model.graph.output.clear()
    model.graph.output.append(
        helper.make_tensor_value_info("reshape_out", TensorProto.FLOAT, [1, 210, 1, 1])
    )


def main() -> None:
    model = _lower_to_onnx()
    metadata_patched = _force_concat_metadata(model)
    if metadata_patched:
        _inject_runtime_check(model)
    model.ir_version = 7  # keep onnxruntime happy on older builds
    onnx.save(model, TARGET_ONNX)

    print(f"[INFO] Wrote {TARGET_ONNX}")
    try:
        sess = ort.InferenceSession(
            str(TARGET_ONNX), providers=["CPUExecutionProvider"]
        )
        feeds = {
            sess.get_inputs()[0].name: np.ones((5, 210, 1, 1), dtype=np.float32),
            sess.get_inputs()[1].name: np.array([0.0], dtype=np.float32),
            sess.get_inputs()[2].name: np.array([1e-3], dtype=np.float32),
        }
        sess.run(None, feeds)
    except Exception as err:
        print("[EXPECTED] onnxruntime failure triggered:")
        print(f"          {err}")
    else:
        if metadata_patched:
            print(
                "[WARN] onnxruntime succeeded; check your ORT build or tighten the test."
            )
        else:
            print("[PASS] onnxruntime succeeded with correct concat metadata.")


if __name__ == "__main__":
    main()
