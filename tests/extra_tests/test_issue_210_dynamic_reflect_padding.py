from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx import to_onnx


def test_issue_210_nnx_conv_reflect_padding_exports_with_symbolic_length() -> None:
    conv = nnx.Conv(
        in_features=1,
        out_features=1,
        kernel_size=3,
        padding="REFLECT",
        rngs=nnx.Rngs(0),
    )
    length = jax.export.symbolic_shape("N", constraints=("N >= 2",))

    model = to_onnx(
        fn=lambda x: conv(x[None, :, None]),
        inputs=[jax.ShapeDtypeStruct(length, dtype=jnp.float32)],
        model_name="issue_210_dynamic_reflect_padding",
    )

    assert model is not None
