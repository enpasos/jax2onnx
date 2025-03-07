# file: jax2onnx/plugins/linear.py

from flax import nnx
from obsolete.convert import Z, OnnxValue
from obsolete.typing_helpers import Supports2Onnx
from jax2onnx.plugins.flax.nnx.linear_general import build_linear_general_onnx_node


def to_onnx(self, x, **params):
    if params.get("use_jaxpr", False):
        return linear_to_onnx_via_jaxpr(self, x, **params)
    # Otherwise, fall back to existing implementation:
    # (e.g., call a pre-existing conversion function or run manual Gemm logic)
    return original_linear_to_onnx(self, x, **params)


def original_linear_to_onnx(self: Supports2Onnx, z: Z, **params) -> Z:
    """Convert an `nnx.Linear` layer into an ONNX `Gemm` node."""
    return build_linear_general_onnx_node(self, z, **params)


# Attach `to_onnx` method to `nnx.Linear`
nnx.Linear.to_onnx = to_onnx


import jax


def linear_to_onnx_via_jaxpr(self, x, **params):
    """New JAXPR-based conversion for nnx.Linear."""
    graph = params["graph"]  # ONNX graph context
    x_name = x.name  # ONNX name of input tensor

    # Add weight and bias as ONNX initializers
    W_val = self.weight  # weight array
    b_val = getattr(self, "bias", None)  # bias array (may not exist if use_bias=False)
    W_name = graph.add_initializer(W_val, name=f"{params.get('name', 'linear')}_W")
    b_name = None
    if b_val is not None:
        b_name = graph.add_initializer(b_val, name=f"{params.get('name', 'linear')}_b")

    # Trace JAXPR of linear computation
    in_shape = x.shape
    in_dtype = x.dtype
    jaxpr = jax.make_jaxpr(lambda inp: self(inp))(
        jax.ShapeDtypeStruct(in_shape, in_dtype)
    )

    onnx_var_map = {jaxpr.invars[0]: x_name}  # Map jaxpr input var to ONNX input name

    output_name = None
    eqns = jaxpr.eqns
    i = 0
    while i < len(eqns):
        eqn = eqns[i]
        prim = eqn.primitive
        # Map dot_general to MatMul or Gemm
        if prim.name == "dot_general":
            # Get ONNX names for operands
            a_name = onnx_var_map[eqn.invars[0]]
            # Weight constant is likely eqn.invars[1] or invars[1] might be tuple of (W,consts) depending on how JAXPR shows it
            W_onnx_name = W_name
            # Check next eqn for bias add fusion
            if (
                i + 1 < len(eqns)
                and eqns[i + 1].primitive.name == "add"
                and b_name is not None
            ):
                add_eqn = eqns[i + 1]
                # Ensure one of the add inputs is the dot output and the other is the bias constant
                if (
                    add_eqn.invars[0] == eqn.outvars[0]
                    or add_eqn.invars[1] == eqn.outvars[0]
                ):
                    # Use Gemm for dot+add
                    output_name = graph.generate_unique_name("linear_out")
                    graph.add_node(
                        "Gemm",
                        inputs=[a_name, W_onnx_name, b_name],
                        outputs=[output_name],
                        attributes={"alpha": 1.0, "beta": 1.0, "transB": 0},
                    )
                    # Map the add eqn output var to Gemm output
                    onnx_var_map[add_eqn.outvars[0]] = output_name
                    i += 2  # skip the next add eqn since we've handled it
                    continue
            # If not fused, just do MatMul
            matmul_out = graph.generate_unique_name("linear_matmul_out")
            graph.add_node("MatMul", inputs=[a_name, W_onnx_name], outputs=[matmul_out])
            onnx_var_map[eqn.outvars[0]] = matmul_out
        elif prim.name == "add":
            # Handle add if not fused: one input could be bias or some other addition
            a_name = onnx_var_map[eqn.invars[0]]
            b_name_onnx = onnx_var_map.get(eqn.invars[1], None)
            if b_name_onnx is None and b_name is not None:
                b_name_onnx = b_name  # bias as constant
            output_name = graph.generate_unique_name("linear_add_out")
            graph.add_node("Add", inputs=[a_name, b_name_onnx], outputs=[output_name])
            onnx_var_map[eqn.outvars[0]] = output_name
        else:
            # Other primitives (if any) can be handled here (e.g., type casts)
            # For simplicity, assume none for Linear.
            pass
        i += 1
    # The final output of the linear
    if output_name is None:
        # If bias was not used and only one eqn (dot_general) produced output
        output_name = onnx_var_map[jaxpr.outvars[0]]
    return OnnxValue(name=output_name)


def get_test_params() -> list:
    """Return test parameters for verifying the ONNX conversion of `nnx.Linear`."""
    return [
        {
            "jax_component": "flax.nnx.Linear",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Linear",
            "onnx": [
                {
                    "component": "Gemm",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html",
                },
                {
                    "component": "MatMul",
                    "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "linear",
                    "component": nnx.Linear(
                        in_features=128,
                        out_features=64,
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(32, 128)],
                    "params": {"use_jaxpr": True},
                },
                {
                    "testcase": "linear_2d",
                    "component": nnx.Linear(
                        in_features=128,
                        out_features=64,
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(32, 10, 128)],
                },
            ],
        }
    ]
