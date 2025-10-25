# scripts/inline_layer_scale.py

import onnx
from onnx import helper
from copy import deepcopy
import sys


def inline_layer_scale(model: onnx.ModelProto) -> onnx.ModelProto:
    domain_to_func = {f.domain: f for f in model.functions}
    vt = domain_to_func["custom.VisionTransformer.1"]

    def clone_value_info(src_vi, new_name):
        clone = deepcopy(src_vi)
        clone.name = new_name
        return clone

    block_funcs = {
        f.domain: f for f in model.functions if f.domain.startswith("custom.Block.")
    }
    layer_scale_funcs = {
        f.domain: f
        for f in model.functions
        if f.domain.startswith("custom.LayerScale.")
    }

    # Build mapping from block domain to corresponding layer scale domains used in that function
    block_layer_scale_domains = {}
    for domain, func in block_funcs.items():
        domains = []
        for node in func.node:
            if node.domain and node.domain.startswith("custom.LayerScale."):
                domains.append(node.domain)
        block_layer_scale_domains[domain] = domains

    vt_value_info_map = {vi.name: vi for vi in vt.value_info}

    new_nodes = []
    for node in vt.node:
        block_domains = block_layer_scale_domains.get(node.domain)
        if node.op_type == "Block" and block_domains:
            block_func = block_funcs[node.domain]
            block_value_info = {vi.name: vi for vi in block_func.value_info}
            mapping = {}
            # Map block inputs and outputs
            mapping[block_func.input[0]] = node.input[0]
            mapping[block_func.output[0]] = node.output[0]

            # Ensure output value info exists
            if node.output[0] not in vt_value_info_map:
                vt.value_info.append(
                    clone_value_info(
                        block_value_info[block_func.output[0]], node.output[0]
                    )
                )
                vt_value_info_map[node.output[0]] = vt.value_info[-1]

            for inner in block_func.node:
                if inner.domain and inner.domain.startswith("custom.LayerScale."):
                    # Inline LayerScale function
                    ls_func = layer_scale_funcs[inner.domain]
                    ls_vi_map = {vi.name: vi for vi in ls_func.value_info}
                    ls_input = inner.input[0]
                    ls_output = inner.output[0]
                    mapping.setdefault(ls_input, mapping.get(ls_input, ls_input))
                    mapping.setdefault(ls_output, mapping.get(ls_output, ls_output))
                    if ls_output not in vt_value_info_map:
                        vt.value_info.append(
                            clone_value_info(block_value_info[ls_output], ls_output)
                        )
                        vt_value_info_map[ls_output] = vt.value_info[-1]
                    for vi_name, vi in ls_vi_map.items():
                        if vi_name in (ls_func.input[0], ls_func.output[0]):
                            continue
                        new_name = f"{inner.name}_{vi_name}"
                        mapping[vi_name] = new_name
                        clone = clone_value_info(vi, new_name)
                        vt.value_info.append(clone)
                    for ls_node in ls_func.node:
                        new_inputs = [mapping.get(inp, inp) for inp in ls_node.input]
                        new_outputs = [mapping.get(out, out) for out in ls_node.output]
                        attrs = {
                            attr.name: helper.get_attribute_value(attr)
                            for attr in ls_node.attribute
                        }
                        new_nodes.append(
                            helper.make_node(
                                ls_node.op_type,
                                new_inputs,
                                new_outputs,
                                name=f"{node.name}_{ls_node.name}",
                                domain=ls_node.domain,
                                **attrs,
                            )
                        )
                else:
                    new_inputs = [mapping.get(inp, inp) for inp in inner.input]
                    new_outputs = [mapping.get(out, out) for out in inner.output]
                    attrs = {
                        attr.name: helper.get_attribute_value(attr)
                        for attr in inner.attribute
                    }
                    new_nodes.append(
                        helper.make_node(
                            inner.op_type,
                            new_inputs,
                            new_outputs,
                            name=f"{node.name}_{inner.name}",
                            domain=inner.domain,
                            **attrs,
                        )
                    )
        else:
            new_nodes.append(node)

    vt.ClearField("node")
    vt.node.extend(new_nodes)
    return model


def main():
    if len(sys.argv) != 3:
        print("Usage: python inline_layer_scale.py <input.onnx> <output.onnx>")
        return
    input_path, output_path = sys.argv[1:3]
    model = onnx.load(input_path)
    model = inline_layer_scale(model)
    onnx.save(model, output_path)


if __name__ == "__main__":
    main()
