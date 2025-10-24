# tests/extra_tests/examples/test_eqx_dino_rotary_lowering.py

import jax
import jax.numpy as jnp

from jax2onnx import to_onnx
from jax2onnx.plugins.examples.eqx.dino import (
    Block,
    DinoRoPE,
    DinoRotaryProcessHeads,
    _dino_rope_inference_sincos,
)


def _count_nodes(ir_model, op_type: str) -> int:
    functions = getattr(ir_model, "functions", {})
    if isinstance(functions, dict):
        fn_values = functions.values()
    else:
        fn_values = functions
    total = 0
    for fn in fn_values:
        graph = getattr(fn, "graph", None)
        if graph is None:
            continue
        nodes = list(getattr(graph, "_nodes", []))
        total += sum(1 for node in nodes if getattr(node, "op_type", "") == op_type)
    return total


def test_dino_rotary_process_heads_emits_rotary_lowering():
    dim = 64
    num_heads = 4
    seq_len = 16

    block = Block(dim=dim, num_heads=num_heads, key=jax.random.PRNGKey(0))
    rope = DinoRoPE(dim=dim, num_heads=num_heads)
    sin, cos = _dino_rope_inference_sincos(rope, H=4, W=4)
    process_heads = DinoRotaryProcessHeads(sin=sin, cos=cos, prefix_tokens=0)

    def block_call(x):
        return block(x, process_heads=process_heads)

    inputs = [jax.ShapeDtypeStruct((1, seq_len, dim), jnp.float32)]
    ir_model = to_onnx(block_call, inputs, return_mode="ir")

    split_nodes = _count_nodes(ir_model, "Split")
    neg_nodes = _count_nodes(ir_model, "Neg")
    concat_nodes = _count_nodes(ir_model, "Concat")

    assert split_nodes >= 2, "expected rotary lowering to introduce Split nodes"
    assert neg_nodes >= 2, "expected rotary lowering to introduce Neg nodes"
    assert concat_nodes >= 2, "expected rotary lowering to introduce Concat nodes"
