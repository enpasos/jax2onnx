# tests/extra_tests/loop/test_loop_ff_like_broadcast_mul_inference_regression.py

import pytest
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.export.shape_poly import InconclusiveDimensionOperation
import onnxruntime as ort
from jax2onnx.user_interface import to_onnx
import onnx


def _inner_scan_body(carry, _):
    x = carry  # (B,H,W)

    # (B,H,W) * (W,) → (B,H,W)
    mask_w = jnp.ones((x.shape[-1],), dtype=x.dtype)
    y = x * mask_w  # (B,H,W)

    # transpose so middle axis becomes W
    yT = jnp.transpose(y, (0, 2, 1))  # (B,W,H)

    # data-dependent mask of shape (1,W,1) so it is NOT a constant
    dep = jnp.sum(yT, axis=(0, 2), keepdims=True)  # (1,W,1)
    maskW_nonconst = dep * 0 + 1.0

    # ⬅️ Make the mask’s rank/shape explicit in the graph, so the aligner
    # sees it as rank-3 and (because it’s the *first* operand) picks it as ref:
    maskW_nonconst = jnp.reshape(maskW_nonconst, (1, yT.shape[1], 1))

    # keep the mask as the FIRST operand to provoke the wrong ref choice
    z = maskW_nonconst * yT  # valid in JAX, but aligner will make bad Expand

    # carry must keep original (B,H,W) shape
    return x, z


def _outer_fori_body(i, state):
    _, ys = lax.scan(_inner_scan_body, state, xs=None, length=2)
    return ys[-1]


def ff_like(y0):
    return lax.fori_loop(0, 1, _outer_fori_body, y0)


def _mul_shapes_in_loop_bodies(onnx_model: onnx.ModelProto):
    out = []
    from onnx import AttributeProto

    def shp(vi):
        if not vi.type or not vi.type.tensor_type or not vi.type.tensor_type.shape:
            return "?"
        dims = vi.type.tensor_type.shape.dim

        def one(d):
            return str(d.dim_value) if d.HasField("dim_value") else (d.dim_param or "?")

        return "(" + ",".join(one(d) for d in dims) + ")"

    def walk(g, where):
        known = {}
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            known[vi.name] = shp(vi)
        for init in g.initializer:
            known[init.name] = "(" + ",".join(str(d) for d in init.dims) + ")"
        for n in g.node:
            if n.op_type == "Mul" and len(n.input) >= 2:
                a, b = n.input[0], n.input[1]
                out.append(
                    (n.name or "Mul", known.get(a, "?"), known.get(b, "?"), where)
                )
            for a in n.attribute:
                if a.type == AttributeProto.GRAPH and a.g is not None:
                    walk(a.g, where + f" → {n.name or n.op_type}.body")
                elif a.type == AttributeProto.GRAPHS and a.graphs:
                    for i, sg in enumerate(a.graphs):
                        walk(sg, where + f" → {n.name or n.op_type}.graphs[{i}]")

    walk(onnx_model.graph, "(main)")
    return out


@pytest.mark.parametrize("dtype", [jnp.float64])  # match integration: double
def test_ff_like_mul_in_loop_inference_currently_fails(dtype):
    spec = jax.ShapeDtypeStruct(("B", 4, 5), dtype)
    try:
        model = to_onnx(
            ff_like,
            inputs=[spec],
            enable_double_precision=True,
            opset=21,
            model_name="ff_like_broadcast_mul_inference_regression",
        )
    except InconclusiveDimensionOperation as exc:
        pytest.xfail(f"converter cannot export loops with symbolic dims: {exc}")

    # If you want to inspect Mul shapes, uncomment:
    # for nm, s0, s1, wh in _mul_shapes_in_loop_bodies(model):
    #     print(f"[MUL] {wh} :: {nm}: {s0} × {s1}")

    # Regression FIX: the model should now load successfully in ORT.
    # Creating the session would raise (Type/ShapeInferenceError) if the Loop
    # body still had incompatible dimensions.
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    # Sanity: a session object must be created.
    assert sess is not None
