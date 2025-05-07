# import unittest
# import jax
# from jax.core import ShapedArray
# from onnx import helper, TensorProto
# import numpy as np
# import logging


# from jax2onnx.converter.onnx_builder import OnnxBuilder
# from jax2onnx.converter.name_generator import UniqueNameGenerator
# from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# class TestJaxprConverterInteractionWithBuilder(unittest.TestCase):

# def test_jaxpr_uses_preexisting_intermediate_tensor_not_new_input(self):
#     """
#     Tests that Jaxpr2OnnxConverter correctly uses an intermediate tensor
#     produced by a node already in its builder (added before _process_jaxpr)
#     without making that intermediate tensor a new graph input.
#     Also inspects why an output tensor might become an input.
#     """
#     # 1. Create the OnnxBuilder instance (simulating 'body_builder')
#     name_gen = UniqueNameGenerator()
#     test_body_builder = OnnxBuilder(
#         name_generator=name_gen,
#         opset=15, # Or your target opset
#         model_name="jaxpr_interaction_test_model"
#     )

#     # 2. Add a true graph input 'A' (analogous to 'iter64')
#     true_subgraph_input_A = test_body_builder.name_generator.get("true_subgraph_input_A_int64")
#     test_body_builder.add_scalar_input(true_subgraph_input_A, TensorProto.INT64)

#     # 3. Add value_info for an intermediate tensor 'B' (analogous to 'iter32')
#     intermediate_tensor_B = test_body_builder.name_generator.get("intermediate_tensor_B_int32")
#     test_body_builder.add_value_info(intermediate_tensor_B, (), np.int32)

#     # 4. Add a Cast node to the builder: A (int64) -> B (int32)
#     cast_node_name = test_body_builder.name_generator.get("preexisting_cast_A_to_B")
#     test_body_builder.add_node(
#         helper.make_node(
#             "Cast",
#             inputs=[true_subgraph_input_A],
#             outputs=[intermediate_tensor_B],
#             to=TensorProto.INT32,
#             name=cast_node_name
#         )
#     )

#     # 5. Create a simple JAX function and its JAXPR
#     def identity_func(x_arg):
#         return x_arg

#     dummy_arg_for_jaxpr = np.int32(0)
#     closed_jaxpr = jax.make_jaxpr(identity_func)(dummy_arg_for_jaxpr)
#     jaxpr_to_process = closed_jaxpr.jaxpr
#     jaxpr_consts = closed_jaxpr.consts

#     self.assertEqual(len(jaxpr_to_process.invars), 1, "JAXPR should have one invar")
#     j_invar_for_B = jaxpr_to_process.invars[0]

#     # 6. Instantiate Jaxpr2OnnxConverter with the builder
#     test_body_converter = Jaxpr2OnnxConverter(test_body_builder)

#     # 7. Map the JAXPR invar to 'B' and outvar to 'C'
#     test_body_converter.var_to_name[j_invar_for_B] = intermediate_tensor_B

#     jaxpr_outvar = jaxpr_to_process.outvars[0]
#     final_graph_output_C = test_body_builder.name_generator.get("final_output_C")
#     test_body_converter.var_to_name[jaxpr_outvar] = final_graph_output_C

#     # 8. Call _process_jaxpr
#     test_body_converter._process_jaxpr(jaxpr_to_process, jaxpr_consts)

#     test_body_builder.add_output(final_graph_output_C, (), np.int32)

#     # --- Start Concrete Inspection ---
#     print("\n\n--- State of test_body_builder BEFORE create_graph ---")
#     print(f"Builder Instance ID: {id(test_body_builder)}")
#     print(f"Builder Name (model_name): {test_body_builder.model_name}")

#     print("\n  test_body_builder.inputs (ValueInfoProto list for formal graph inputs):")
#     if not test_body_builder.inputs:
#         print("    <empty>")
#     for i, inp_vi in enumerate(test_body_builder.inputs):
#         tensor_type_str = "N/A"
#         if inp_vi.type.HasField("tensor_type") and inp_vi.type.tensor_type: # Check HasField for safety
#              tensor_type_str = TensorProto.DataType.Name(inp_vi.type.tensor_type.elem_type)
#         print(f"    Input {i}: Name='{inp_vi.name}', ONNX Type='{tensor_type_str}'")

#     print("\n  test_body_builder.outputs (ValueInfoProto list for formal graph outputs):")
#     if not test_body_builder.outputs:
#         print("    <empty>")
#     for i, out_vi in enumerate(test_body_builder.outputs):
#         tensor_type_str = "N/A"
#         if out_vi.type.HasField("tensor_type") and out_vi.type.tensor_type:
#              tensor_type_str = TensorProto.DataType.Name(out_vi.type.tensor_type.elem_type)
#         print(f"    Output {i}: Name='{out_vi.name}', ONNX Type='{tensor_type_str}'")

#     print("\n  test_body_builder.value_info (All known tensors with type/shape info):")
#     key_tensors_for_vi_check = {
#         true_subgraph_input_A: "Input A (true_subgraph_input_A)",
#         intermediate_tensor_B: "Intermediate B (intermediate_tensor_B)",
#         final_graph_output_C: "Output C (final_graph_output_C)"
#     }
#     found_vi_details = []
#     # Make a defensive copy if iterating and potentially modifying, though here it's just reading
#     value_info_list = list(test_body_builder.value_info)
#     for vi in value_info_list:
#         if vi.name in key_tensors_for_vi_check:
#             tensor_type_str = "N/A"
#             if vi.type.HasField("tensor_type") and vi.type.tensor_type:
#                 tensor_type_str = TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
#             found_vi_details.append(f"Name='{vi.name}' ({key_tensors_for_vi_check[vi.name]}), ONNX Type='{tensor_type_str}'")
#     if found_vi_details:
#         for detail in found_vi_details:
#              print(f"    ValueInfo: {detail}")
#     else:
#         print("    <no specific value_info found for key tracking tensors>")
#     print(f"    Total value_info items: {len(value_info_list)}")


#     print("\n  test_body_builder.nodes (Internal operations):")
#     if not test_body_builder.nodes:
#         print("    <empty>")
#     for i, node_proto in enumerate(test_body_builder.nodes):
#         node_inputs_str = ", ".join(node_proto.input) if node_proto.input else "None"
#         node_outputs_str = ", ".join(node_proto.output) if node_proto.output else "None"
#         print(f"    Node {i}: Name='{node_proto.name}', OpType='{node_proto.op_type}', "
#               f"Inputs=[{node_inputs_str}], Outputs=[{node_outputs_str}]")
#     print("--- End of State Inspection ---\n\n")
#     # --- End Concrete Inspection ---

#     # 9. Create the graph
#     graph_proto = test_body_builder.create_graph(
#         name=test_body_builder.get_unique_name("test_subgraph_processed_by_jaxpr_conv"),
#         is_subgraph=True
#     )

#     # 10. Assertions
#     actual_graph_inputs = [inp.name for inp in graph_proto.input]

#     self.assertIn(
#         true_subgraph_input_A,
#         actual_graph_inputs,
#         f"The true subgraph input '{true_subgraph_input_A}' was not found. "
#         f"Actual inputs: {actual_graph_inputs}"
#     )
#     self.assertNotIn(
#         intermediate_tensor_B,
#         actual_graph_inputs,
#         f"The intermediate tensor '{intermediate_tensor_B}' (produced by pre-existing Cast "
#         f"and used by JAXPR) should NOT be a subgraph input. Actual inputs: {actual_graph_inputs}"
#     )
#     self.assertEqual(
#         len(actual_graph_inputs),
#         1,
#         f"Subgraph should have had exactly 1 input ('{true_subgraph_input_A}'), "
#         f"but found {len(actual_graph_inputs)}: {actual_graph_inputs}"
#     )

# if __name__ == '__main__':
#     unittest.main()
