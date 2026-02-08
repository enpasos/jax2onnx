# tests/extra_tests/test_issue_173_nested_residual_groups.py

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx import to_onnx

pix = pytest.importorskip("dm_pix", reason="dm_pix is required for issue #173 repro")

filters: int = 64
kernel_size: tuple[int, int] = (3, 3)
blocks: int = 4
groups: int = 4
spatial_size: int = 16


class DepthToSpace(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.block_size = 2

    def __call__(self, input):
        return pix.depth_to_space(input, self.block_size)


class ResBlock(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)
        self.conv1 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)

    def __call__(self, input):
        x = nnx.silu(self.conv0(input))
        x = self.conv1(x)
        return x + input


class ResGroup(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.res_blocks = nnx.List(ResBlock(rngs=rngs) for _ in range(blocks))
        self.conv0 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)

    def __call__(self, input):
        x = input
        for block in self.res_blocks:
            x = block(x)
        x = self.conv0(x)
        return x + input


class ResGroupModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(1, filters, kernel_size=kernel_size, rngs=rngs)
        self.res_groups = nnx.List(ResGroup(rngs=rngs) for _ in range(groups))
        self.conv1 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)
        self.feats_conv = nnx.Conv(filters, 4, kernel_size=kernel_size, rngs=rngs)
        self.depth_to_space = DepthToSpace(rngs=rngs)

    def __call__(self, input):
        stem = self.conv0(input)
        x = stem
        for group in self.res_groups:
            x = group(x)
        x = self.conv1(x) + stem
        x = self.feats_conv(x)
        x = self.depth_to_space(x)
        return jnp.clip(x, 0.0, 1.0)


def _add_nodes(model):
    return [node for node in model.graph.node if node.op_type == "Add"]


def _assert_repro_wiring(model, *, blocks_count: int, groups_count: int) -> None:
    adds = _add_nodes(model)
    add_outputs = {node.output[0] for node in adds}
    adds_per_group = blocks_count + 1

    assert len(adds) == groups_count * adds_per_group + 1

    for g in range(groups_count):
        base = g * adds_per_group
        first_add = adds[base]
        tail_add = adds[base + blocks_count]

        if g > 0:
            prev_tail = adds[(g - 1) * adds_per_group + blocks_count].output[0]
            assert (
                first_add.input[1] == prev_tail
            ), f"group {g}: first block skip should equal {prev_tail}"
            assert (
                tail_add.input[1] == prev_tail
            ), f"group {g}: tail skip should equal {prev_tail}"

        for i in range(1, blocks_count):
            prev_out = adds[base + i - 1].output[0]
            cur_skip = adds[base + i].input[1]
            assert (
                cur_skip == prev_out
            ), f"group {g}, block {i}: expected skip {prev_out}, got {cur_skip}"

        block_add_outputs = {adds[base + i].output[0] for i in range(blocks_count)}
        assert (
            tail_add.input[1] not in block_add_outputs
        ), f"group {g}: tail skip unexpectedly reads an inner block add output"

    stem_skip = adds[0].input[1]
    final_add = adds[-1]
    assert stem_skip in final_add.input
    other_final_inputs = [inp for inp in final_add.input if inp != stem_skip]
    assert len(other_final_inputs) == 1
    other_final_input = other_final_inputs[0]
    assert other_final_input not in add_outputs


def _assert_clip_present(model) -> None:
    op_types = {node.op_type for node in model.graph.node}
    assert "Clip" in op_types or {"Min", "Max"}.issubset(op_types)


def _assert_no_dangling_transposes(model) -> None:
    graph_output_names = {out.name for out in model.graph.output if out.name}
    consumers_by_input_name: dict[str, int] = {}
    for node in model.graph.node:
        for inp_name in node.input:
            if inp_name:
                consumers_by_input_name[inp_name] = (
                    consumers_by_input_name.get(inp_name, 0) + 1
                )

    for node in model.graph.node:
        if node.op_type != "Transpose":
            continue
        for out_name in node.output:
            if not out_name:
                continue
            is_graph_output = out_name in graph_output_names
            has_consumer = consumers_by_input_name.get(out_name, 0) > 0
            assert (
                is_graph_output or has_consumer
            ), f"Dangling Transpose output detected: '{out_name}'"


def _assert_no_transpose_inputs_to_adds(model) -> None:
    producers = {
        out_name: node
        for node in model.graph.node
        for out_name in node.output
        if out_name
    }
    for node in model.graph.node:
        if node.op_type != "Add":
            continue
        for inp_name in node.input:
            if not inp_name:
                continue
            producer = producers.get(inp_name)
            if producer is None:
                continue
            assert producer.op_type != "Transpose", (
                f"Add '{node.name}' still receives transposed input '{inp_name}' "
                f"from '{producer.name}'."
            )


def _assert_ort_matches_jax_nchw(model, fn, x_nhwc: jax.Array) -> None:
    ort = pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required for issue #173 regression tests"
    )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    x_nchw = np.transpose(np.asarray(x_nhwc, dtype=np.float32), (0, 3, 1, 2))
    y_jax_nchw = np.transpose(np.asarray(fn(x_nhwc)), (0, 3, 1, 2))
    (y_onnx,) = session.run(None, {input_name: x_nchw})
    np.testing.assert_allclose(y_onnx, y_jax_nchw, rtol=1e-4, atol=1e-4)


def test_issue_173_repro_nested_residual_groups_with_depth_to_space() -> None:
    model_obj = ResGroupModel(rngs=nnx.Rngs(0))

    def fn(x):
        return model_obj(x)

    input_shape = (1, spatial_size, spatial_size, 1)
    model = to_onnx(
        fn,
        inputs=[jax.ShapeDtypeStruct(input_shape, jnp.float32)],
        model_name="issue173_nested_residual_groups_repro",
        inputs_as_nchw=[0],
        outputs_as_nchw=[0],
    )

    _assert_repro_wiring(model, blocks_count=blocks, groups_count=groups)
    _assert_clip_present(model)
    _assert_no_dangling_transposes(model)
    _assert_no_transpose_inputs_to_adds(model)

    output_dims = [
        dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim
    ]
    assert output_dims == [1, 1, spatial_size * 2, spatial_size * 2]

    x = jax.random.normal(jax.random.PRNGKey(123), input_shape, jnp.float32)
    _assert_ort_matches_jax_nchw(model, fn, x)
