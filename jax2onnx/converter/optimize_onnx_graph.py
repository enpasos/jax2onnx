# file: jax2onnx/converter/optimize_transpose.py


import onnx
from onnx import shape_inference
from typing import Dict, List


def remove_redundant_casts(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove Cast nodes that cast a tensor to its own type.

    This function first runs shape inference to populate type information in the graph,
    then examines each Cast node. If the input tensor's element type is the same as
    the "to" attribute of the Cast node, the node is redundant and is removed,
    with its consumers rewired to the original input.

    Args:
        onnx_model: The input ONNX model.

    Returns:
        The optimized ONNX model with redundant Cast nodes removed.
    """
    # Run shape inference to obtain type information.
    inferred_model = shape_inference.infer_shapes(onnx_model)
    graph = inferred_model.graph

    # Build a mapping from tensor name to its element type.
    type_dict: Dict[str, int] = {}

    def update_type_info(values):
        for value in values:
            # value.type.tensor_type.elem_type is an int corresponding to TensorProto enum.
            type_dict[value.name] = value.type.tensor_type.elem_type

    update_type_info(graph.input)
    update_type_info(graph.value_info)
    update_type_info(graph.output)
    for init in graph.initializer:
        type_dict[init.name] = init.data_type

    nodes_to_remove: List[onnx.NodeProto] = []

    # Iterate over nodes to find redundant Casts.
    for node in graph.node:
        if node.op_type != "Cast":
            continue

        # Get the "to" attribute from the node.
        to_attr = None
        for attr in node.attribute:
            if attr.name == "to":
                to_attr = attr.i
                break
        if to_attr is None:
            continue

        # The Cast node should have one input.
        cast_inp = node.input[0]
        # Check if we know the element type for the input.
        if cast_inp not in type_dict:
            continue

        input_elem_type = type_dict[cast_inp]
        # If the target type equals the input's type, the cast is redundant.
        if input_elem_type != to_attr:
            continue

        # Rewire: for all nodes consuming the output of this Cast, replace with cast_inp.
        cast_out = node.output[0]
        for n in graph.node:
            for idx, inp in enumerate(n.input):
                if inp == cast_out:
                    n.input[idx] = cast_inp

        # Also update graph outputs if needed.
        for out in graph.output:
            if out.name == cast_out:
                out.name = cast_inp

        nodes_to_remove.append(node)

    # Remove the redundant Cast nodes.
    new_nodes = [n for n in graph.node if n not in nodes_to_remove]
    del graph.node[:]
    graph.node.extend(new_nodes)
    return inferred_model


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
