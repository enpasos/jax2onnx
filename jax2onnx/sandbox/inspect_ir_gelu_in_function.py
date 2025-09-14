# jax2onnx/sandbox/inspect_ir_gelu_in_function.py

from jax2onnx.converter2.conversion_api import to_onnx as to_ir  # returns IR model

# import your SuperBlock (the same function as the failing test):
from jax2onnx.plugins2.examples2.onnx_functions.onnx_functions_000 import SuperBlock


def dump_gelu_attrs_in_ir(ir_model):
    # Top graph
    print("TOP GRAPH:")
    nodes = getattr(ir_model.graph, "nodes", None) or getattr(
        ir_model.graph, "_nodes", []
    )
    for n in nodes:
        if getattr(n, "op_type", "") == "Gelu":
            attrs = getattr(n, "attributes", None) or getattr(n, "_attributes", None)
            print(
                "  Gelu attrs (top):",
                {
                    getattr(a, "name", getattr(a, "key", None)): getattr(
                        a, "value", None
                    )
                    for a in (attrs or [])
                },
            )

    # Functions (dict or list)
    fstore = getattr(ir_model, "functions", None) or []
    fvalues = fstore.values() if isinstance(fstore, dict) else fstore
    for fn in fvalues:
        print(
            f"FUNCTION: domain='{getattr(fn, 'domain', '')}', name='{getattr(fn, 'name', '')}'"
        )
        fn_nodes = getattr(fn.graph, "nodes", None) or getattr(fn.graph, "_nodes", [])
        for n in fn_nodes:
            if getattr(n, "op_type", "") == "Gelu":
                attrs = getattr(n, "attributes", None) or getattr(
                    n, "_attributes", None
                )
                print(
                    "  Gelu attrs (fn):",
                    {
                        getattr(a, "name", getattr(a, "key", None)): getattr(
                            a, "value", None
                        )
                        for a in (attrs or [])
                    },
                )


def main():
    # Build the *same* callable your test uses
    # SuperBlock signature depends on your module; adapt if needed.
    def callable_(x):
        sb = SuperBlock()
        return sb(x)

    # IR conversion (opset must match your tests)
    ir_model = to_ir(
        fn=callable_,
        inputs=[("B", 10, 3)],
        input_params=None,
        model_name="probe_superblock_ir",
        opset=21,
        enable_double_precision=False,
        loosen_internal_shapes=False,
        record_primitive_calls_file=None,
    )
    dump_gelu_attrs_in_ir(ir_model)


if __name__ == "__main__":
    main()
