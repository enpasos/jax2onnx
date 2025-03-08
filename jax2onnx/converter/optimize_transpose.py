import onnx
from onnx import helper
from typing import Dict, List


def remove_redundant_transpose_pairs(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove consecutive Transpose nodes whose combined permutation is the identity.
    This function modifies the graph in place.
    """
    graph = onnx_model.graph
    # Build a mapping from output names to nodes that consume them.
    output_to_consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            output_to_consumers.setdefault(inp, []).append(node)

    nodes_to_remove: List[onnx.NodeProto] = []
    nodes_to_add: List[onnx.NodeProto] = (
        []
    )  # Not used in current code; could be removed.

    for node in graph.node:
        if node.op_type != "Transpose":
            continue
        # Get the permutation attribute of the first Transpose node.
        perm_attr = [attr for attr in node.attribute if attr.name == "perm"]
        if not perm_attr:
            continue
        perm1 = list(perm_attr[0].ints)
        # Check if the output of this node is exclusively consumed by a second Transpose.
        out_name = node.output[0]
        consumers = output_to_consumers.get(out_name, [])
        if len(consumers) != 1:
            continue
        consumer = consumers[0]
        if consumer.op_type != "Transpose":
            continue
        perm_attr2 = [attr for attr in consumer.attribute if attr.name == "perm"]
        if not perm_attr2:
            continue
        perm2 = list(perm_attr2[0].ints)
        # Compose the two permutations: composed[i] = perm1[perm2[i]]
        composed = [perm1[p] for p in perm2]
        # If the composed permutation is identity, mark both for removal.
        if composed == list(range(len(composed))):
            # Rewire: any node that consumes the output of the consumer will instead consume the input of the first Transpose.
            orig_input = node.input[0]
            consumer_out = consumer.output[0]
            for n in graph.node:
                for idx, inp in enumerate(n.input):
                    if inp == consumer_out:
                        n.input[idx] = orig_input
            # Also, if consumer_out is listed in graph outputs, replace it.
            for tensor in graph.output:
                if tensor.name == consumer_out:
                    tensor.name = orig_input
            nodes_to_remove.extend([node, consumer])

    # Remove marked nodes.
    new_nodes = [n for n in graph.node if n not in nodes_to_remove]
    del graph.node[:]
    graph.node.extend(new_nodes)
    return onnx_model
