# tests/extra_tests/test_issue_174_dm_pix_depth_to_space.py

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx import to_onnx

pix = pytest.importorskip("dm_pix", reason="dm_pix is required for issue #174 repro")


class SimpleModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(1, 4, kernel_size=(3, 3), rngs=rngs)

    def __call__(self, input):
        x = self.conv(input)
        x = pix.depth_to_space(x, 2)
        return x


def _get_depth_to_space_node(model):
    nodes = [node for node in model.graph.node if node.op_type == "DepthToSpace"]
    assert len(nodes) == 1
    return nodes[0]


def test_issue_174_dm_pix_depth_to_space_uses_onnx_depth_to_space() -> None:
    model_obj = SimpleModel(rngs=nnx.Rngs(0))

    def fn(x):
        return model_obj(x)

    input_shape = (1, 8, 8, 1)
    model = to_onnx(
        fn,
        inputs=[jax.ShapeDtypeStruct(input_shape, jnp.float32)],
        model_name="issue174_dm_pix_depth_to_space",
        inputs_as_nchw=[0],
        outputs_as_nchw=[0],
    )

    op_types = [node.op_type for node in model.graph.node]
    assert op_types.count("DepthToSpace") == 1
    assert "Reshape" not in op_types

    d2s = _get_depth_to_space_node(model)
    attrs = {attr.name: attr for attr in d2s.attribute}
    assert attrs["blocksize"].i == 2
    assert attrs["mode"].s.decode() == "DCR"

    output_dims = [
        dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim
    ]
    assert output_dims == [1, 1, 16, 16]

    ort = pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required for issue #174 regression test"
    )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    x_nhwc = jax.random.normal(jax.random.PRNGKey(123), input_shape, jnp.float32)
    x_nchw = np.transpose(np.asarray(x_nhwc, dtype=np.float32), (0, 3, 1, 2))
    y_jax_nchw = np.transpose(np.asarray(fn(x_nhwc), dtype=np.float32), (0, 3, 1, 2))
    (y_onnx,) = session.run(None, {input_name: x_nchw})
    np.testing.assert_allclose(y_onnx, y_jax_nchw, rtol=1e-4, atol=1e-4)
