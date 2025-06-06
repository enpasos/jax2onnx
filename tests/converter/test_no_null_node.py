# tests/converter/test_no_empty_graph_name.py
import onnx
import glob
from typing import Iterable
import pytest


def _walk_graphs(graph) -> Iterable[onnx.GraphProto]:
    """Yield `graph` and every sub-graph reachable from it."""
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _walk_graphs(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    yield from _walk_graphs(g)


@pytest.mark.order(-1)  # run *after* the models have been produced
def test_no_empty_graph_names():
    onnx_files = glob.glob("docs/onnx/**/*.onnx", recursive=True)
    assert onnx_files, "No ONNX files found under docs/onnx/**"

    bad = []
    for path in onnx_files:
        model = onnx.load(path)
        for g in _walk_graphs(model.graph):
            if g.name == "":
                bad.append(f"{path}: contains unnamed GraphProto")

    if bad:
        raise AssertionError(
            "🔴 Empty graph names trigger a ‘null’ node in Netron:\n" + "\n".join(bad)
        )
