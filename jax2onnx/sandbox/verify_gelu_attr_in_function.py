import os
import onnx_ir as ir
import onnx

OUT_DIR = "tmp_verify_gelu"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- robust STRING attribute maker (works across ir variants) ---------------
def make_string_attr(name: str, value: str):
    Attr = getattr(ir, "Attr", getattr(ir, "Attribute", None))
    if Attr is None:
        raise RuntimeError("onnx_ir.Attr/Attribute not found")

    # Preferred newer API
    if hasattr(Attr, "s"):
        return Attr.s(name, value)

    # Enum-based constructor
    AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
    if AttrType is not None and hasattr(AttrType, "STRING"):
        return Attr(name, AttrType.STRING, value)

    # Last resort: some builds accept (name, value)
    return Attr(name, value)

def build_function_model(func_default_opset: int | None, out_name: str) -> str:
    # --- Function body: Gelu(approximate="tanh") over (3,)
    fx = ir.Value(name="fx", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,)))
    fy = ir.Value(name="fy", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,)))
    gelu_fn = ir.Node(
        op_type="Gelu",
        domain="",
        inputs=[fx],
        outputs=[fy],
        name="FuncGelu",
        attributes=[make_string_attr("approximate", "tanh")],
    )

    # Create the function graph; set default-domain opset ONLY if requested
    if func_default_opset is None:
        f_graph = ir.Graph(inputs=[fx], outputs=[fy], nodes=[gelu_fn], name="FGraph")
    else:
        f_graph = ir.Graph(
            inputs=[fx],
            outputs=[fy],
            nodes=[gelu_fn],
            name="FGraph",
            opset_imports={"": int(func_default_opset)},
        )

    func = ir.Function(domain="custom", name="F", graph=f_graph, attributes=[])

    # --- Top graph: call the function F(custom)
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,)))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,)))
    call = ir.node(op_type="F", domain="custom", inputs=[x], outputs=[y])

    # IMPORTANT: set default model graph opset and custom domain here
    main_graph = ir.Graph(
        inputs=[x],
        outputs=[y],
        nodes=[call],
        name="Main",
        opset_imports={"": 21, "custom": 1},
    )

    # Model: attach the function; do not assign Model.opset_imports (read-only in some builds)
    model = ir.Model(main_graph, ir_version=10, functions=[func])

    path = os.path.join(OUT_DIR, out_name)
    ir.save(model, path)
    return path

def dump(path: str):
    print(f"\n=== Inspect: {path} ===")
    m = onnx.load_model(path)

    # Model-level opsets
    print("Model opset_imports:", {imp.domain: imp.version for imp in m.opset_import})

    # Function-level opsets and attributes
    if not getattr(m, "functions", None):
        print("No functions found in model.")
        return

    for f in m.functions:
        f_imports = {imp.domain: imp.version for imp in f.opset_import}
        print(f"Function: domain='{f.domain}', name='{f.name}', opsets:", f_imports)
        for n in f.node:
            if n.op_type == "Gelu":
                # Gather attributes (STRING → a.s)
                attrs = {}
                for a in n.attribute:
                    if a.type == onnx.AttributeProto.AttributeType.STRING:
                        val = a.s.decode() if isinstance(a.s, (bytes, bytearray)) else str(a.s)
                        attrs[a.name] = val
                print("  Gelu node attrs:", attrs)

# A: function graph WITHOUT default domain opset
pA = build_function_model(func_default_opset=None, out_name="func_gelu_no_func_opset.onnx")
dump(pA)

# B: function graph WITH default domain opset 21
pB = build_function_model(func_default_opset=21, out_name="func_gelu_func_opset_21.onnx")
dump(pB)

print("\nInterpretation:")
print(" - If A shows Gelu with attrs {}, and B shows {'approximate': 'tanh'},")
print("   the attribute was dropped due to missing default-domain opset on the function graph.")
