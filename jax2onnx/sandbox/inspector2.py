import onnx

model = onnx.load("docs/onnx/examples/onnx_functions/011_vit_conv_embedding.onnx")

print("\n[🌳] Top-Level Outputs:")
for output in model.graph.output:
    print(
        f" - {output.name}, shape={[d.dim_value for d in output.type.tensor_type.shape.dim]}"
    )

print("\n[🔧] Top-Level Nodes:")
for node in model.graph.node:
    print(
        f" - {node.name}: {node.op_type}, inputs={node.input}, outputs={node.output}, domain={node.domain}"
    )

print(f"\n[🧠] Number of functions: {len(model.functions)}")
for f in model.functions:
    print(
        f" - {f.name} ({len(f.node)} ops), inputs={list(f.input)}, outputs={list(f.output)}"
    )
