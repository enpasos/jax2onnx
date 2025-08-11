"""
Regression: nested Loop bodies with a float32 GatherND feeding float64 outputs.

This reproduces an external failure where ONNX type-inference complained that
the output of a GatherND inside a Loop body was float32 while the body graph
declared (or downstream expected) float64. Our Loop lowering now aligns body
output dtypes via explicit Casts when needed.
"""

from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax import lax

from jax2onnx import to_onnx
onnx = pytest.importorskip("onnx", reason="onnx is required for this test")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime is required for this test")


def _fn_nested_loop_gathernd_mixed_dtypes():
    """Return (acc_f64[2], y_last_f32[2]) via two nested scans."""
    table = jnp.array([[0.1], [0.2], [0.3], [0.4]], dtype=jnp.float32)  # (4, 1)
    gdn = lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1),
    )

    def inner(carry, _):
        i, acc = carry  # i:int32, acc:float64
        j = jnp.mod(i, jnp.int32(4))
        idx = jnp.stack([j, jnp.int32(0)], axis=0)[None, :]  # (1, 2)
        val = lax.gather(table, idx, dimension_numbers=gdn, slice_sizes=(1, 1))  # (1,)
        val = val[0]  # -> ()
        acc = acc + jnp.asarray(val, dtype=jnp.float64)
        return (i + jnp.int32(1), acc), val  # per-step y is float32

    def outer(carry, _):
        (_, acc_final), ys = lax.scan(inner, (jnp.int32(0), jnp.float64(0.0)), xs=None, length=3)
        y_last = ys[-1]  # float32
        return carry + jnp.asarray(y_last, dtype=jnp.float64), (acc_final, y_last)

    _, ys_outer = lax.scan(outer, jnp.float64(0.0), xs=None, length=2)
    return ys_outer


def _iter_loop_bodies(graph):
    for n in graph.node:
        if n.op_type == "Loop":
            for a in n.attribute:
                if a.name == "body" and a.g is not None:
                    yield a.g
                    yield from _iter_loop_bodies(a.g)

@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_nested_loop_gathernd_mixed_dtypes_fails_without_harmonization(tmp_path, monkeypatch):
    """
    Negative test: disable Loop-body binop harmonization and assert there are
    NO Cast nodes named like our harmonizer (“CastAlignBinOp”) inside any Loop body.
    We don't depend on ORT failing to load (that varies by ORT/version and the
    JAX program can contain explicit Casts).
    """
    monkeypatch.setenv("JAX2ONNX_DISABLE_LOOP_BINOP_CAST", "1")

    model = to_onnx(
        _fn_nested_loop_gathernd_mixed_dtypes,
        inputs=[],
        enable_double_precision=True,
        opset=21,
        model_name="nested_loop_gathernd_mixed_no_cast",
    )
    p = tmp_path / "nested_loop_gathernd_mixed_no_cast.onnx"
    p.write_bytes(model.SerializeToString())

    m = onnx.load(str(p))

    def iter_loop_bodies(graph):
        for n in graph.node:
            if n.op_type == "Loop":
                for a in n.attribute:
                    if a.name == "body" and a.g is not None:
                        yield a.g
                        yield from iter_loop_bodies(a.g)

    def body_contains_castalign(g):
        for n in g.node:
            if n.op_type == "Cast" and (n.name or "").find("CastAlignBinOp") != -1:
                return True
        return False

    assert not any(body_contains_castalign(g) for g in iter_loop_bodies(m.graph)), \
        "Harmonization casts ('CastAlignBinOp') should NOT be present when disabled."

    monkeypatch.delenv("JAX2ONNX_DISABLE_LOOP_BINOP_CAST", raising=False)
