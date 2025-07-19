# file: jax2onnx/examples/nnx/sequential_relu.py
#
# Examples that exercise support for `nnx.Sequential`.

from flax import nnx
from jax2onnx.plugin_system import register_example


# ---------------------------------------------------------------------------
# STATELESS EXAMPLE: Two ReLU activations in a row.
# ---------------------------------------------------------------------------
double_relu = nnx.Sequential(
    nnx.relu,  # first ReLU
    nnx.relu,  # second ReLU
)

register_example(
    component="SequentialReLU",
    description="Two ReLU activations chained with nnx.Sequential (no parameters).",
    source="https://flax.readthedocs.io/en/latest/nnx/index.html",
    since="v0.7.1",
    context="examples.nnx",
    children=["nnx.Sequential", "nnx.relu"],
    testcases=[
        {
            "testcase": "sequential_double_relu",
            "callable": double_relu,
            "input_shapes": [(5,)],
        },
    ],
)


# ---------------------------------------------------------------------------
# FAILING STATEFUL EXAMPLE: A module that calls a Sequential submodule with a residual.
# This is the minimal test case that reproduces the TransformerDecoder bug.
# ---------------------------------------------------------------------------
class ComplexParentWithResidual(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.initial_op = nnx.Linear(in_features=16, out_features=16, rngs=rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(in_features=16, out_features=32, rngs=rngs),
            lambda x: nnx.relu(x),
            nnx.Linear(in_features=32, out_features=16, rngs=rngs),
        )
        self.layernorm = nnx.LayerNorm(num_features=16, rngs=rngs)

    def __call__(self, x):
        x_residual = self.initial_op(x)
        ffn_output = self.ffn(x_residual)
        # The residual connection around the nested Sequential is the key to reproducing the bug.
        output = self.layernorm(x_residual + ffn_output)
        return output


register_example(
    component="SequentialWithResidual",
    description="Tests nnx.Sequential nested inside a module with a residual connection.",
    source="Internal bug report",
    since="v0.7.1",
    context="examples.nnx",
    children=["nnx.Sequential", "nnx.Linear", "nnx.relu", "nnx.LayerNorm"],
    testcases=[
        {
            "testcase": "sequential_nested_with_residual",
            "callable": ComplexParentWithResidual(rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 16)],
        },
    ],
)
