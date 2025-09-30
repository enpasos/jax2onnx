from __future__ import annotations

import onnx

import jax
import jax.numpy as jnp

from jax2onnx.user_interface import to_onnx


def _walk_graphs(graph):
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH and attr.g is not None:
                yield from _walk_graphs(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS and attr.graphs:
                for g in attr.graphs:
                    yield from _walk_graphs(g)


def test_generated_graphs_have_names():
    @jax.jit
    def fn(x):
        return jnp.tanh(x) + 1.0

    model = to_onnx(
        fn,
        inputs=[("B", 4)],
        model_name="no_null_graph_ir",
        use_onnx_ir=True,
    )

    for graph in _walk_graphs(model.graph):
        assert (
            graph.name
        ), "Encountered unnamed GraphProto, would render as 'null' in tools"
