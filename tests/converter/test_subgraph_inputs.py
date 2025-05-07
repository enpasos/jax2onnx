import unittest
import numpy as np
from onnx import helper, TensorProto


from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.name_generator import UniqueNameGenerator


class TestOnnxBuilderSubgraphInputs(unittest.TestCase):

    def test_intermediate_tensor_is_not_subgraph_input(self):
        """
        Tests that an intermediate tensor, produced by a node within the
        (sub)graph and registered with add_value_info, does not incorrectly
        become a formal input to that (sub)graph.
        """
        name_gen = UniqueNameGenerator()
        # Use a reasonably current opset, e.g., 15 or 17.
        # The specific opset is not critical for this structural test.
        builder = OnnxBuilder(
            name_generator=name_gen, opset=15, model_name="subgraph_input_test_model"
        )

        # 1. Define a true graph input (analogous to 'iter64' in ForiLoopPlugin)
        graph_input_A_name = builder.name_generator.get("graph_input_A")
        builder.add_scalar_input(graph_input_A_name, TensorProto.INT64)

        # 2. Define an intermediate tensor (analogous to 'iter32' in ForiLoopPlugin)
        # This tensor will be produced by a Cast node.
        # It's crucial to call add_value_info for it, as done in ForiLoopPlugin.
        intermediate_B_name = builder.name_generator.get("intermediate_B_int32")
        builder.add_value_info(intermediate_B_name, (), np.int32)  # Scalar int32

        # 3. Add a node that produces the intermediate tensor from the true input
        # Cast: graph_input_A (int64) -> intermediate_B_int32 (int32)
        cast_node_name = builder.name_generator.get("node_cast_A_to_B")
        builder.add_node(
            helper.make_node(
                "Cast",
                inputs=[graph_input_A_name],
                outputs=[intermediate_B_name],
                to=TensorProto.INT32,
                name=cast_node_name,
            )
        )

        # 4. Add a node that consumes the intermediate tensor and produces a graph output
        # This makes intermediate_B_name a necessary part of the graph's internal logic.
        graph_output_C_name = builder.name_generator.get("graph_output_C")
        identity_node_name = builder.name_generator.get("node_identity_B_to_C")

        # Explicitly add value_info for the output tensor as well for clarity,
        # though builder.add_output would also handle creating ValueInfoProto.
        builder.add_value_info(graph_output_C_name, (), np.int32)
        builder.add_node(
            helper.make_node(
                "Identity",
                inputs=[intermediate_B_name],
                outputs=[graph_output_C_name],
                name=identity_node_name,
            )
        )
        # Define the graph output formally
        builder.add_output(graph_output_C_name, (), np.int32)

        # 5. Create the graph, specifying is_subgraph=True, as this is where
        # the issue was observed (in the context of a Loop body).
        subgraph_name = builder.get_unique_name("test_subgraph_structure")
        graph_proto = builder.create_graph(name=subgraph_name, is_subgraph=True)

        # 6. Perform Assertions
        subgraph_actual_input_names = [inp.name for inp in graph_proto.input]
        subgraph_actual_output_names = [out.name for out in graph_proto.output]

        # Assertion 1: The true graph input ('graph_input_A_name') must be present.
        self.assertIn(
            graph_input_A_name,
            subgraph_actual_input_names,
            f"The true graph input '{graph_input_A_name}' was not found in the subgraph's inputs. "
            f"Actual inputs: {subgraph_actual_input_names}",
        )

        # Assertion 2: CRITICAL - The intermediate tensor ('intermediate_B_name')
        # (produced by the Cast node) should NOT be a formal input to the subgraph.
        self.assertNotIn(
            intermediate_B_name,
            subgraph_actual_input_names,
            f"The intermediate tensor '{intermediate_B_name}' (produced by '{cast_node_name}') "
            f"was incorrectly listed as a subgraph input. Actual inputs: {subgraph_actual_input_names}",
        )

        # Assertion 3: The graph output ('graph_output_C_name') should be correctly listed.
        self.assertIn(
            graph_output_C_name,
            subgraph_actual_output_names,
            f"The graph output '{graph_output_C_name}' was not found in the subgraph's outputs. "
            f"Actual outputs: {subgraph_actual_output_names}",
        )

        # Assertion 4: The subgraph should have exactly one formal input in this setup.
        self.assertEqual(
            len(subgraph_actual_input_names),
            1,
            f"The subgraph should have had exactly 1 input ('{graph_input_A_name}'), "
            f"but found {len(subgraph_actual_input_names)}: {subgraph_actual_input_names}",
        )

        # Optional: Verify node connections for completeness
        node_names_in_graph = [node.name for node in graph_proto.node]
        self.assertIn(cast_node_name, node_names_in_graph)
        self.assertIn(identity_node_name, node_names_in_graph)

        cast_node_proto = next(n for n in graph_proto.node if n.name == cast_node_name)
        self.assertEqual(cast_node_proto.input[0], graph_input_A_name)
        self.assertEqual(cast_node_proto.output[0], intermediate_B_name)

        identity_node_proto = next(
            n for n in graph_proto.node if n.name == identity_node_name
        )
        self.assertEqual(identity_node_proto.input[0], intermediate_B_name)
        self.assertEqual(identity_node_proto.output[0], graph_output_C_name)


if __name__ == "__main__":
    unittest.main()
