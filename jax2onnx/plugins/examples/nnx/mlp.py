# file: jax2onnx/examples/mlp.py

import jax
import numpy as np
from flax import nnx
from flax.nnx import BatchNorm, Dropout, Linear

from jax2onnx.plugin_system import onnx_function, register_example, construct_and_call


def _safe_kernel_init(key, shape, dtype=jax.numpy.float32):
    return jax.random.normal(key, shape, dtype) * 0.02


@onnx_function
class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = Linear(din, dmid, kernel_init=_safe_kernel_init, rngs=rngs)
        # Defer "deterministic" to call-time to avoid import-time PRNG use.
        self.dropout = Dropout(rate=0.1, rngs=rngs)
        self.bn = BatchNorm(dmid, use_running_average=True, rngs=rngs)
        self.linear2 = Linear(dmid, dout, kernel_init=_safe_kernel_init, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True):
        x = self.linear1(x)
        x = self.bn(x)
        x = self.dropout(x, deterministic=deterministic)
        x = nnx.gelu(x)
        return self.linear2(x)


register_example(
    component="MLP",
    description="A simple Multi-Layer Perceptron (MLP) with BatchNorm, Dropout, and GELU activation.",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.7.0",
    context="examples.nnx.mlp",
    testcases=[
        {
            "testcase": "mlp_basic",
            # Strict late-construction pattern: construct_and_call(Class, **init_kwargs)
            "callable": construct_and_call(
                MLP,
                din=30,
                dmid=20,
                dout=10,
                rngs=nnx.Rngs(17),
            ),
            "input_values": [np.random.randn(4, 30).astype(np.float32)],
            "input_params": {"deterministic": True},
            "run_only_f32_variant": True,
        },
    ],
)


