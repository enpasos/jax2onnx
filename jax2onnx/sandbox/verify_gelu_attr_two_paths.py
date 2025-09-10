# jax2onnx/sandbox/verify_gelu_attr_two_paths.py
import os
import onnx_ir as ir
import onnx

OUT = "tmp_verify_two_paths"
os.makedirs(OUT, exist_ok=True)

def _make_string_attr(name: str, value: str):
    # Robust STRING attribute creator across ir builds
    Attr = getattr(ir, "Attr", getattr(ir, "Attribute", None))
    AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
    if hasattr(Attr, "s"):
        return Attr.s(name, value)
    if AttrType is not None and hasattr(AttrType, "STRING"):
        return Attr(name, AttrType.STRING, value)
    # Accepts (name, value) in some builds
    return Attr(name, value)

def build_and_dump(path: str):
    # --- Function graph: two Gelu nodes, attrs built two different ways ---
    fx = ir.Value(name="fx",
                  type=ir.TensorType(ir.DataType.FLOAT),
                  shape=ir.Shape((3,)))
    fy  = ir.Value(name="fy",
                   type=ir.TensorType(ir.DataType.FLOAT),
                   shape=ir.Shape((3,)))
    fy2 = ir.Value(name="fy2",
                   type=ir.TensorType(ir.DataType.FLOAT),
                   shape=ir.Shape((3,)))

    # G1: enum-ctor style for STRING (potentially lossy in some builds)
    AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
    g1_attrs = []
    if AttrType is not None and hasattr(AttrType, "STRING"):
        g1_attrs = [ir.Attr("approximate", AttrType.STRING, "tanh")]
    else:
        g1_attrs = [_make_string_attr("approximate", "tanh")]  # fallback

    g1 = ir.Node(op_type="Gelu",
                 domain="",
                 inputs=[fx],
                 outputs=[fy],
                 name="G1",
                 attributes=g1_attrs)

    # G2: robust helper path for STRING
    g2 = ir.Node(op_type="Gelu",
                 domain="",
                 inputs=[fx],
                 outputs=[fy2],
                 name="G2",
                 attributes=[_make_string_attr("approximate", "tanh")])

    fgraph = ir.Graph(inputs=[fx],
                      outputs=[fy, fy2],
                      nodes=[g1, g2],
                      name="FGraph",
                      opset_imports={"": 21})  # default domain for body

    func = ir.Function(domain="custom", name="F", graph=fgraph, attributes=[])

    # --- Main graph: call F twice, produce 2 outputs so we keep both nodes in body ---
    x  = ir.Value(name="x",
                  type=ir.TensorType(ir.DataType.FLOAT),
                  shape=ir.Shape((3,)))
    y1 = ir.Value(name="y1",
                  type=ir.TensorType(ir.DataType.FLOAT),
                  shape=ir.Shape((3,)))
    y2 = ir.Value(name="y2",
                  type=ir.TensorType(ir.DataType.FLOAT),
                  shape=ir.Shape((3,)))

    call1 = ir.node(op_type="F", domain="custom", inputs=[x], outputs=[y1])
    call2 = ir.node(op_type="F", domain="custom", inputs=[x], outputs=[y2])

    main = ir.Graph(inputs=[x],
                    outputs=[y1, y2],
                    nodes=[call1, call2],
                    name="Main",
                    opset_imports={"": 21, "custom": 1})

    model = ir.Model(main, ir_version=10, functions=[func])
    ir.save(model, path)

    # --- Inspect with onnx to see what survived inside the FunctionProto ---
    m = onnx.load_model(path)
    print(f"\n=== Inspect: {path} ===")
    for f in m.functions:
        print("function opsets:", {imp.domain: imp.version for imp in f.opset_import})
        for n in f.node:
            if n.op_type == "Gelu":
                attrs = {}
                for a in n.attribute:
                    if a.name == "approximate" and a.type == onnx.AttributeProto.STRING:
                        attrs["approximate"] = a.s.decode()
                print(f" {n.name}: attrs={attrs}")

p = os.path.join(OUT, "gelu_two_paths.onnx")
build_and_dump(p)
print("\nInterpretation:")
print(" - If G1 shows approximate missing/None while G2 has 'tanh',")
print("   the enum-ctor STRING path is lossy in function bodies for your onnx_ir build.")
