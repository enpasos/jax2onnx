# tests/extra_tests/framework/test_internal_improvements.py

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

import jax2onnx
from jax2onnx.converter.conversion_api import _to_ir_dtype_from_np, to_onnx
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.optimizer_graph_utils import (
    _consumer_nodes,
    _node_in_sequence,
    _producer_node,
)
from jax2onnx.converter.output_binding import _value_is_graph_connected
from jax2onnx.ir_utils import numpy_dtype_to_ir_with_float_policy
from jax2onnx.plugins.plugin_system import (
    INSTANCE_MAP2,
    PLUGIN_REGISTRY,
    FunctionPlugin,
    _IN_FUNCTION_BUILD,
    _ONNX_FN_HITS,
    _import_tree,
    apply_monkey_patches,
    onnx_function,
)
from jax2onnx.user_interface import _is_floating_dtype, _run_allclose, _to_numpy_input


class _DummyVar:
    def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.aval = SimpleNamespace(shape=shape, dtype=dtype)


def test_bfloat16_and_float16_ir_dtype_helpers() -> None:
    bf16_dtype = jnp.dtype("bfloat16")
    f16_dtype = np.dtype(np.float16)

    assert _to_ir_dtype_from_np(bf16_dtype) == ir.DataType.BFLOAT16
    assert _to_ir_dtype_from_np(f16_dtype) == ir.DataType.FLOAT

    assert (
        numpy_dtype_to_ir_with_float_policy(bf16_dtype, enable_double_precision=False)
        == ir.DataType.BFLOAT16
    )
    assert (
        numpy_dtype_to_ir_with_float_policy(bf16_dtype, enable_double_precision=True)
        == ir.DataType.BFLOAT16
    )
    assert (
        numpy_dtype_to_ir_with_float_policy(f16_dtype, enable_double_precision=False)
        == ir.DataType.FLOAT16
    )


def test_inputs_as_nchw_preserves_low_precision_dtypes() -> None:
    for dtype, expected_ir_dtype in (
        (np.float16, ir.DataType.FLOAT16),
        (jnp.bfloat16, ir.DataType.BFLOAT16),
    ):
        ir_model = to_onnx(
            fn=lambda x, _dtype=dtype: x + jnp.asarray(1, dtype=_dtype),
            inputs=[jax.ShapeDtypeStruct((1, 4, 4, 3), dtype)],
            input_params={},
            model_name="nchw_low_precision",
            enable_double_precision=False,
            opset=21,
            record_primitive_calls_file=None,
            inputs_as_nchw=[0],
            outputs_as_nchw=[0],
        )
        assert ir_model.graph.inputs[0].type is not None
        assert ir_model.graph.inputs[0].type.dtype == expected_ir_dtype
        assert ir_model.graph.outputs[0].type is not None
        assert ir_model.graph.outputs[0].type.dtype == expected_ir_dtype


def test_user_interface_bfloat16_and_unsigned_int_helpers() -> None:
    bf16_arr = np.asarray([1.0, 2.0], dtype=jnp.dtype("bfloat16"))
    assert _is_floating_dtype(bf16_arr)

    meta_bf16 = SimpleNamespace(type="tensor(bfloat16)", shape=(2,))
    meta_bf16_alias = SimpleNamespace(type="tensor(bf16)", shape=(2,))
    meta_bf16_complex = SimpleNamespace(type="tensor(bfloat16)", shape=(2, 2))
    meta_u16 = SimpleNamespace(type="tensor(uint16)", shape=(2,))
    meta_u32 = SimpleNamespace(type="tensor(uint32)", shape=(2,))
    meta_u64 = SimpleNamespace(type="tensor(uint64)", shape=(2,))

    out_bf16 = _to_numpy_input(np.asarray([1.0, 2.0], dtype=np.float32), meta_bf16)
    out_bf16_alias = _to_numpy_input(
        np.asarray([1.0, 2.0], dtype=np.float32), meta_bf16_alias
    )
    assert out_bf16.dtype.name == "bfloat16"
    assert out_bf16_alias.dtype.name == "bfloat16"

    packed_complex_bf16 = _to_numpy_input(
        np.asarray([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64),
        meta_bf16_complex,
    )
    assert packed_complex_bf16.dtype.name == "bfloat16"
    assert packed_complex_bf16.shape == (2, 2)

    assert _to_numpy_input(np.asarray([1, 2]), meta_u16).dtype == np.uint16
    assert _to_numpy_input(np.asarray([1, 2]), meta_u32).dtype == np.uint32
    assert _to_numpy_input(np.asarray([1, 2]), meta_u64).dtype == np.uint64


def test_run_allclose_complex_and_bfloat16_preserves_imaginary_part(
    monkeypatch: Any,
) -> None:
    bf16_dtype = jnp.dtype("bfloat16")

    class _FakeSession:
        def __init__(self, outputs: list[np.ndarray]) -> None:
            self._outputs = outputs

        def get_inputs(self) -> list[Any]:
            return [SimpleNamespace(name="x", type="tensor(float)", shape=(1,))]

        def run(self, _output_names: Any, _ort_inputs: Any) -> list[np.ndarray]:
            return self._outputs

    class _FakeORT:
        class GraphOptimizationLevel:
            ORT_DISABLE_ALL = 0

        class SessionOptions:
            def __init__(self) -> None:
                self.graph_optimization_level = 0
                self.enable_mem_pattern = True
                self.intra_op_num_threads = 1
                self.inter_op_num_threads = 1

        current_outputs: list[np.ndarray] = []

        @classmethod
        def InferenceSession(cls, *args: Any, **kwargs: Any) -> _FakeSession:
            return _FakeSession(cls.current_outputs)

    monkeypatch.setattr(
        "jax2onnx.user_interface.importlib.import_module",
        lambda name: _FakeORT if name == "onnxruntime" else __import__(name),
    )

    # 1) Same-shape bfloat16 output must NOT discard imaginary part of complex JAX output.
    _FakeORT.current_outputs = [np.asarray([1.0], dtype=bf16_dtype)]
    ok, msg = _run_allclose(
        lambda x: jnp.asarray([1.0 + 2.0j], dtype=jnp.complex64),
        "dummy.onnx",
        [np.asarray([1.0], dtype=np.float32)],
        {},
        rtol=1e-2,
        atol=1e-2,
    )
    assert not ok
    assert "mismatch" in msg

    # 2) Packed (..., 2) bfloat16 output must unpack to complex and match.
    _FakeORT.current_outputs = [np.asarray([[1.0, 2.0]], dtype=bf16_dtype)]
    ok_packed, _ = _run_allclose(
        lambda x: jnp.asarray([1.0 + 2.0j], dtype=jnp.complex64),
        "dummy.onnx",
        [np.asarray([1.0], dtype=np.float32)],
        {},
        rtol=1e-2,
        atol=1e-2,
    )
    assert ok_packed


def test_graph_connectivity_and_optimizer_fast_paths() -> None:
    ctx = IRContext(opset=21, enable_double_precision=False)
    x = ctx.add_input_for_invar(_DummyVar((2,), np.dtype(np.float32)), 0)
    const_val = ctx.bind_const_for_var(
        object(), np.asarray([1.0, 2.0], dtype=np.float32)
    )
    out_val = ctx.builder.Add(x, const_val, _outputs=["add_out"])
    orphan = ir.Value(
        name="orphan",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape((2,)),
    )

    assert _value_is_graph_connected(ctx, x)
    assert _value_is_graph_connected(ctx, const_val)
    assert _value_is_graph_connected(ctx, out_val)
    assert not _value_is_graph_connected(ctx, orphan)

    # Review edge case #2: a graph output whose producer node was removed from the
    # graph must NOT be treated as connected even if value.graph is still set.
    ctx.builder.add_graph_output_value(out_val)
    add_node = out_val.producer()
    assert add_node is not None
    ctx.builder.graph.remove(add_node)
    assert not _value_is_graph_connected(ctx, out_val)

    # Re-add node for subsequent checks
    out_val2 = ctx.builder.Add(x, const_val, _outputs=["add_out2"])
    mid_val = ctx.builder.Neg(out_val2, _outputs=["neg_out"])
    end_val = ctx.builder.Relu(mid_val, _outputs=["relu_out"])

    # Review edge case #1: stale list snapshot after removing and inserting a middle
    # node at equal graph size must NOT falsely match the newly inserted node.
    stale_nodes = list(ctx.builder.nodes)
    neg_node = mid_val.producer()
    relu_node = end_val.producer()
    assert neg_node is not None and relu_node is not None
    ctx.builder.graph.remove(neg_node)
    replacement_node = ir.Node("", "Abs", inputs=[out_val2], num_outputs=1)
    ctx.builder.graph.insert_before(relu_node, replacement_node)
    assert len(stale_nodes) == len(ctx.builder.graph)
    assert stale_nodes[0] is ctx.builder.graph[0]
    assert stale_nodes[-1] is ctx.builder.graph[-1]
    assert _node_in_sequence(replacement_node, ctx.builder.graph)
    assert not _node_in_sequence(replacement_node, stale_nodes)
    assert _producer_node(stale_nodes, replacement_node.outputs[0]) is None
    assert (
        _producer_node(ctx.builder.graph, replacement_node.outputs[0])
        is replacement_node
    )

    consumers = _consumer_nodes(ctx.builder.graph, x)
    assert len(consumers) == 1 and consumers[0].op_type == "Add"


def test_to_onnx_restores_x64_when_user_fn_mutates_config() -> None:
    original_x64 = bool(jax.config.read("jax_enable_x64"))
    try:
        jax.config.update("jax_enable_x64", False)

        def mutating_fn(x: jax.Array) -> jax.Array:
            jax.config.update("jax_enable_x64", True)
            return x + 1.0

        jax2onnx.to_onnx(
            mutating_fn,
            [(2,)],
            enable_double_precision=False,
            return_mode="ir",
        )
        assert bool(jax.config.read("jax_enable_x64")) is False
    finally:
        jax.config.update("jax_enable_x64", original_x64)


def test_import_tree_retries_failed_module_in_second_pass(
    tmp_path: Path, monkeypatch: Any
) -> None:
    pkg_dir = tmp_path / "retry_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        "READY = False\nATTEMPTS = 0\n", encoding="utf-8"
    )
    (pkg_dir / "a_consumer.py").write_text(
        "import retry_pkg\n"
        "retry_pkg.ATTEMPTS += 1\n"
        "if not retry_pkg.READY:\n"
        "    raise RuntimeError('z_provider not ready yet')\n"
        "LOADED = True\n",
        encoding="utf-8",
    )
    (pkg_dir / "z_provider.py").write_text(
        "import retry_pkg\nretry_pkg.READY = True\n", encoding="utf-8"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        _import_tree(pkg_dir, "retry_pkg")
        assert sys.modules["retry_pkg"].ATTEMPTS == 2
        assert "retry_pkg.a_consumer" in sys.modules
        assert getattr(sys.modules["retry_pkg.a_consumer"], "LOADED", False) is True
    finally:
        for mod in list(sys.modules):
            if mod == "retry_pkg" or mod.startswith("retry_pkg."):
                sys.modules.pop(mod, None)


def test_contextvar_defaults_are_immutable_frozensets() -> None:
    assert isinstance(_ONNX_FN_HITS.get(), frozenset)
    assert isinstance(_IN_FUNCTION_BUILD.get(), frozenset)


def test_function_plugin_resolves_bound_method_per_instance_key() -> None:
    @onnx_function
    class TinyScaleBlock(nnx.Module):
        def __init__(self, factor: float):
            self.factor = factor

        def __call__(self, x):
            return x * self.factor

    block_a = TinyScaleBlock(2.0)
    block_b = TinyScaleBlock(5.0)

    with apply_monkey_patches():
        y_a = block_a(jnp.asarray([1.0, 3.0], dtype=jnp.float32))
        y_b = block_b(jnp.asarray([1.0, 3.0], dtype=jnp.float32))
        np.testing.assert_allclose(np.asarray(y_a), np.asarray([2.0, 6.0]))
        np.testing.assert_allclose(np.asarray(y_b), np.asarray([5.0, 15.0]))

        plugin = next(
            p
            for p in PLUGIN_REGISTRY.values()
            if isinstance(p, FunctionPlugin) and p.target is TinyScaleBlock
        )
        key_a = next(k for k, inst in INSTANCE_MAP2.items() if inst is block_a)
        key_b = next(k for k, inst in INSTANCE_MAP2.items() if inst is block_b)
        fn_a = plugin._resolve_orig_fn(key_a)
        fn_b = plugin._resolve_orig_fn(key_b)
        assert getattr(fn_a, "__self__", None) is block_a
        assert getattr(fn_b, "__self__", None) is block_b


def test_function_plugin_abstract_eval_uses_requested_instance() -> None:
    @onnx_function
    class TinySliceBlock(nnx.Module):
        def __init__(self, width: int) -> None:
            self.width = width

        def __call__(self, x: jax.Array) -> jax.Array:
            return x[: self.width]

    block_a = TinySliceBlock(1)
    block_b = TinySliceBlock(3)
    sample = jnp.arange(4, dtype=jnp.float32)

    with apply_monkey_patches():
        block_a(sample)
        block_b(sample)
        plugin = next(
            p
            for p in PLUGIN_REGISTRY.values()
            if isinstance(p, FunctionPlugin) and p.target is TinySliceBlock
        )
        spec = jax.ShapeDtypeStruct(sample.shape, sample.dtype)
        # The plugin was last called on block_b. Tracing a previously captured
        # call for block_a must still infer block_a's output shape.
        shape_a = jax.eval_shape(
            lambda x: plugin.primitive.bind(x, instance_key=id(block_a)), spec
        )
        shape_b = jax.eval_shape(
            lambda x: plugin.primitive.bind(x, instance_key=id(block_b)), spec
        )

    assert shape_a.shape == (1,)
    assert shape_b.shape == (3,)
    assert shape_a.dtype == shape_b.dtype == np.dtype(np.float32)


def test_function_plugin_batching_uses_requested_instance() -> None:
    @onnx_function
    class TinyBatchScaleBlock(nnx.Module):
        def __init__(self, factor: float) -> None:
            self.factor = factor

        def __call__(self, x: jax.Array) -> jax.Array:
            return x * self.factor

    block_a = TinyBatchScaleBlock(2.0)
    block_b = TinyBatchScaleBlock(5.0)
    sample = jnp.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=jnp.float32)

    with apply_monkey_patches():
        block_a(sample[0])
        block_b(sample[0])
        plugin = next(
            p
            for p in PLUGIN_REGISTRY.values()
            if isinstance(p, FunctionPlugin) and p.target is TinyBatchScaleBlock
        )
        # Batch an earlier call after another instance updated the plugin's
        # most recent callable. Each captured key must keep its own weights.
        batched_a = jax.vmap(
            lambda x: plugin.primitive.bind(x, instance_key=id(block_a))
        )(sample)
        batched_b = jax.vmap(
            lambda x: plugin.primitive.bind(x, instance_key=id(block_b))
        )(sample)

    np.testing.assert_allclose(np.asarray(batched_a), np.asarray(sample) * 2.0)
    np.testing.assert_allclose(np.asarray(batched_b), np.asarray(sample) * 5.0)
