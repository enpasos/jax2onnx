# file: jax2onnx/converter/optimize_transpose.py


import onnx
from onnx import helper
from typing import Dict, List

# Define the set of allowed elementwise operations.
ALLOWED_ELEMENTWISE_OPS = {"Elu", "Gelu", "Relu", "Sigmoid", "Tanh"}


def remove_redundant_transpose_pairs(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove Transpose pairs (possibly separated by elementwise-only nodes)
    whose combined permutation is the identity.

    This function looks for a chain:
      T1 -> [E1 -> E2 -> ... -> E_k] -> T2
    where T1 and T2 are Transpose nodes and E* are elementwise ops (from ALLOWED_ELEMENTWISE_OPS)
    that do not change the ordering of elements.

    When the composed permutation (i.e. T2∘T1) is the identity,
    T1 and T2 are removed and the graph is rewired:
      - For the first elementwise node (if any), its input is replaced with T1's input.
      - Consumers of T2's output are rewired to use the output of the last elementwise node,
        or T1's input if there is no intermediate elementwise op.

    The function modifies the graph in place and returns the modified model.
    """
    graph = onnx_model.graph

    # Build a mapping from tensor name to list of consumer nodes.
    output_to_consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            output_to_consumers.setdefault(inp, []).append(node)

    # Use a list to track nodes to remove.
    nodes_to_remove: List[onnx.NodeProto] = []

    # Iterate over a snapshot of nodes.
    for node in list(graph.node):
        if node in nodes_to_remove:
            continue
        if node.op_type != "Transpose":
            continue

        # Start building the chain with the first Transpose.
        chain = [node]
        current_node = node

        # Walk downstream along the unique-consumer chain.
        while True:
            out_name = current_node.output[0]
            consumers = output_to_consumers.get(out_name, [])
            if len(consumers) != 1:
                break  # Cannot extend chain uniquely.
            next_node = consumers[0]
            # If the next node is one of the allowed elementwise ops, add it to the chain.
            if next_node.op_type in ALLOWED_ELEMENTWISE_OPS:
                chain.append(next_node)
                current_node = next_node
                continue
            # Otherwise, if the next node is a Transpose, add it and stop.
            elif next_node.op_type == "Transpose":
                chain.append(next_node)
            break

        # Only remove if we have at least a pair (chain length >= 2).
        if len(chain) < 2:
            continue

        # At this point, chain = [T1, (E1,...,E_k)*, T2].
        T1 = chain[0]
        T2 = chain[-1]

        # Get permutation attributes from T1 and T2.
        perm_attr1 = [attr for attr in T1.attribute if attr.name == "perm"]
        perm_attr2 = [attr for attr in T2.attribute if attr.name == "perm"]
        if not perm_attr1 or not perm_attr2:
            continue
        perm1 = list(perm_attr1[0].ints)
        perm2 = list(perm_attr2[0].ints)

        # Compose the two permutations: composed[i] = perm1[perm2[i]]
        composed = [perm1[p] for p in perm2]
        # Check if the composed permutation is the identity.
        if composed != list(range(len(composed))):
            continue

        # --- Rewire the graph to bypass T1 and T2 ---
        # For the first node after T1 (if any elementwise op exists), replace its input.
        if len(chain) > 2:
            first_elem_node = chain[1]
            # Replace any occurrence of T1's output with T1's input.
            new_input = T1.input[0]
            for i in range(len(first_elem_node.input)):
                if first_elem_node.input[i] == T1.output[0]:
                    first_elem_node.input[i] = new_input

        # Determine the new tensor that should replace T2's output.
        # If there are elementwise nodes between, use the output of the last elementwise node.
        # Otherwise (direct T1->T2), use T1.input[0].
        if len(chain) > 2:
            new_output = chain[-2].output[0]
        else:
            new_output = T1.input[0]

        # Rewire all consumers of T2's output to use new_output.
        for n in graph.node:
            for i in range(len(n.input)):
                if n.input[i] == T2.output[0]:
                    n.input[i] = new_output
        # Also, update graph outputs if needed.
        for tensor in graph.output:
            if tensor.name == T2.output[0]:
                tensor.name = new_output

        # Mark T1 and T2 for removal.
        nodes_to_remove.extend([T1, T2])

    # Remove marked nodes.
    new_nodes = [n for n in graph.node if n not in nodes_to_remove]
    del graph.node[:]
    graph.node.extend(new_nodes)
    return onnx_model
