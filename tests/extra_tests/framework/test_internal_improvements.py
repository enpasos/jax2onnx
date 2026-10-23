# tests/extra_tests/framework/test_internal_improvements.py

from __future__ import annotations

from pathlib import Path
import gc
import inspect
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import weakref

from flax import nnx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
import pytest

import jax2onnx
import jax2onnx.plugins.plugin_system as plugin_system
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
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec, apply_patches
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
        key_a = next(k for k, inst in INSTANCE_MAP2.items() if inst is block_a)
        key_b = next(k for k, inst in INSTANCE_MAP2.items() if inst is block_b)
        # The plugin was last called on block_b. Tracing a previously captured
        # call for block_a must still infer block_a's output shape.
        shape_a = jax.eval_shape(
            lambda x: plugin.primitive.bind(x, instance_key=key_a), spec
        )
        shape_b = jax.eval_shape(
            lambda x: plugin.primitive.bind(x, instance_key=key_b), spec
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
        key_a = next(k for k, inst in INSTANCE_MAP2.items() if inst is block_a)
        key_b = next(k for k, inst in INSTANCE_MAP2.items() if inst is block_b)
        batched_a = jax.vmap(lambda x: plugin.primitive.bind(x, instance_key=key_a))(
            sample
        )
        batched_b = jax.vmap(lambda x: plugin.primitive.bind(x, instance_key=key_b))(
            sample
        )

    np.testing.assert_allclose(np.asarray(batched_a), np.asarray(sample) * 2.0)
    np.testing.assert_allclose(np.asarray(batched_b), np.asarray(sample) * 5.0)


def test_function_plugin_recycled_address_uses_new_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WidthDriven:
        def __init__(self, width: int) -> None:
            self.width = width

        def __call__(self, x: jax.Array) -> jax.Array:
            return jnp.zeros((*x.shape[:-1], self.width), dtype=x.dtype)

    plugin = FunctionPlugin("onnx_fn::test.RecycledAddress", WidthDriven)
    wrapped = plugin._make_patch_fn(plugin.primitive, is_class=True)(
        WidthDriven.__call__
    )
    spec = jax.ShapeDtypeStruct((3, 257, 768), jnp.float32)
    new_instance = WidthDriven(3072)
    expected = jax.eval_shape(new_instance, spec)
    assert expected.shape == (3, 257, 3072)
    current_instance = WidthDriven(192)
    old_ref = weakref.ref(current_instance)
    reused_address = id(current_instance)
    actual_id = id

    def controlled_id(value: object) -> int:
        if isinstance(value, WidthDriven) and value.width != 64:
            return reused_address
        return actual_id(value)

    monkeypatch.setattr(plugin_system, "id", controlled_id, raising=False)
    first = jax.make_jaxpr(lambda x: wrapped(current_instance, x))(spec)
    old_key = first.eqns[0].params["instance_key"]
    assert first.out_avals[0].shape == (3, 257, 192)
    assert INSTANCE_MAP2[old_key] is current_instance

    # Move the existing last-bound fallback off the old instance before GC.
    current_instance = WidthDriven(64)
    intervening = jax.make_jaxpr(lambda x: wrapped(current_instance, x))(spec)
    assert intervening.out_avals[0].shape == (3, 257, 64)
    gc.collect()
    assert old_ref() is None
    assert old_key not in INSTANCE_MAP2

    # Exercise the normal wrapper with a simulated recycled address. Never
    # reinject an old private token: registration must distinguish lifetimes.
    current_instance = new_instance
    actual = jax.make_jaxpr(lambda x: wrapped(current_instance, x))(spec)
    assert actual.out_avals[0].shape == expected.shape
    new_key = actual.eqns[0].params["instance_key"]
    assert new_key != old_key
    assert INSTANCE_MAP2[new_key] is new_instance


def test_function_plugin_live_instance_reuses_key_and_abstract_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SliceBlock:
        def __call__(self, x: jax.Array) -> jax.Array:
            return x[:2]

    plugin = FunctionPlugin("onnx_fn::test.StableLifetime", SliceBlock)
    wrapped = plugin._make_patch_fn(plugin.primitive, is_class=True)(
        SliceBlock.__call__
    )
    original_abstract_eval = plugin.primitive.abstract_eval
    calls = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["instance_key"])
        return original_abstract_eval(*args, **kwargs)

    monkeypatch.setattr(plugin.primitive, "abstract_eval", observe)
    block = SliceBlock()
    spec = jax.ShapeDtypeStruct((4,), jnp.float32)
    first = jax.make_jaxpr(lambda x: wrapped(block, x))(spec)
    second = jax.make_jaxpr(lambda x: wrapped(block, x))(spec)
    first_key = first.eqns[0].params["instance_key"]
    assert first_key == second.eqns[0].params["instance_key"]
    assert plugin_system._register_function_instance(block) == first_key
    assert calls == [first_key]
    assert first.out_avals[0].shape == second.out_avals[0].shape == (2,)
    assert INSTANCE_MAP2[first_key] is block


def test_function_instance_keys_distinguish_equal_unhashable_objects() -> None:
    comparisons = []

    class EqualBlock:
        __hash__ = None

        def __eq__(self, other: object) -> bool:
            comparisons.append(other)
            return isinstance(other, EqualBlock)

    first = EqualBlock()
    second = EqualBlock()
    assert first == second
    comparisons.clear()
    first_key = plugin_system._register_function_instance(first)
    second_key = plugin_system._register_function_instance(second)
    assert isinstance(first_key, int) and isinstance(second_key, int)
    assert second_key > first_key
    assert plugin_system._register_function_instance(first) == first_key
    assert plugin_system._register_function_instance(second) == second_key
    assert INSTANCE_MAP2[first_key] is first
    assert INSTANCE_MAP2[second_key] is second
    assert comparisons == []


def test_function_instance_keys_support_frozen_modules_without_hashing() -> None:
    class FrozenBlock(eqx.Module):
        width: int

        def __hash__(self) -> int:
            raise AssertionError("instance registration must not hash modules")

    first = FrozenBlock(2)
    second = FrozenBlock(2)
    assert first == second
    before = (vars(first).copy(), vars(second).copy())
    first_key = plugin_system._register_function_instance(first)
    second_key = plugin_system._register_function_instance(second)
    assert second_key > first_key
    assert plugin_system._register_function_instance(first) == first_key
    assert INSTANCE_MAP2[first_key] is first
    assert INSTANCE_MAP2[second_key] is second
    assert (vars(first), vars(second)) == before


def test_function_instance_registration_does_not_retain_objects() -> None:
    class Block:
        pass

    block = Block()
    identity = id(block)
    reference = weakref.ref(block)
    key = plugin_system._register_function_instance(block)
    assert plugin_system._FUNCTION_INSTANCE_IDENTITIES[identity][0]() is block
    del block
    gc.collect()
    assert reference() is None
    assert key not in INSTANCE_MAP2
    assert identity not in plugin_system._FUNCTION_INSTANCE_IDENTITIES


def test_function_instance_stale_callback_preserves_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Block:
        pass

    old = Block()
    new = Block()
    reused_address = id(old)
    actual_id = id
    monkeypatch.setattr(
        plugin_system,
        "id",
        lambda obj: reused_address if isinstance(obj, Block) else actual_id(obj),
        raising=False,
    )
    old_key = plugin_system._register_function_instance(old)
    old_entry = plugin_system._FUNCTION_INSTANCE_IDENTITIES[reused_address]
    old_reference = old_entry[0]
    stale_callback = old_reference.__callback__
    assert stale_callback is not None
    # The existing entry is live but belongs to a different object. It must
    # never be accepted merely because the synthetic address matches.
    new_key = plugin_system._register_function_instance(new)
    successor = plugin_system._FUNCTION_INSTANCE_IDENTITIES[reused_address]
    assert old_reference() is old
    assert new_key > old_key
    assert successor[0]() is new
    assert INSTANCE_MAP2[old_key] is old
    assert INSTANCE_MAP2[new_key] is new
    del old
    gc.collect()
    assert old_reference() is None
    assert old_key not in INSTANCE_MAP2
    assert plugin_system._FUNCTION_INSTANCE_IDENTITIES[reused_address] is successor
    stale_callback(old_reference)
    assert plugin_system._FUNCTION_INSTANCE_IDENTITIES[reused_address] is successor
    assert plugin_system._register_function_instance(new) == new_key
    assert INSTANCE_MAP2[new_key] is new


def _restoration_target(kind: str) -> tuple[Any, str]:
    def original(*args: Any) -> None:
        pass

    class Parent:
        target = original

    class Target(Parent):
        pass

    if kind == "inherited":
        return Target, "target"
    if kind == "inherited-classmethod":
        Parent.target = classmethod(original)
        return Target, "target"
    if kind == "metaclass":

        class Meta(type):
            __call__ = original

        class MetaTarget(metaclass=Meta):
            pass

        return MetaTarget, "__call__"
    if kind == "ordinary":
        Target.target = original
    elif kind == "staticmethod":
        Target.target = staticmethod(original)
    elif kind == "classmethod":
        Target.target = classmethod(original)
    elif kind == "module":
        module = ModuleType("restoration_target")
        module.target = original
        return module, "target"
    else:
        raise AssertionError(kind)
    return Target, "target"


def _assert_restored_attribute(
    owner: Any, attr: str, raw: Any, local: bool, resolved: Any
) -> None:
    assert (attr in vars(owner)) is local
    assert inspect.getattr_static(owner, attr) is raw
    current = getattr(owner, attr)
    assert current is resolved or (
        inspect.ismethod(current)
        and inspect.ismethod(resolved)
        and current.__func__ is resolved.__func__
        and current.__self__ is resolved.__self__
    )


@pytest.mark.parametrize("context", ["function", "leaf"])
@pytest.mark.parametrize(
    "kind",
    [
        "inherited",
        "inherited-classmethod",
        "metaclass",
        "ordinary",
        "staticmethod",
        "classmethod",
        "module",
    ],
)
def test_patch_context_restores_exact_namespace_and_descriptor_after_nested_exception(
    monkeypatch: pytest.MonkeyPatch, context: str, kind: str
) -> None:
    owner, attr = _restoration_target(kind)
    raw, local, resolved = (
        inspect.getattr_static(owner, attr),
        attr in vars(owner),
        getattr(owner, attr),
    )
    parent = (
        type(owner)
        if kind == "metaclass"
        else (owner.__bases__[0] if inspect.isclass(owner) else None)
    )
    parent_namespace = dict(vars(parent)) if parent is not None else None
    calls = []
    replacement = object()

    def factory(original: Any) -> Any:
        calls.append(original)
        return replacement

    if context == "function":
        assert not plugin_system._PATCH_STATE
        monkeypatch.setattr(
            plugin_system, "_iter_patch_specs", lambda: iter([(factory, [owner], attr)])
        )
        activate = apply_monkey_patches
    else:

        def activate():
            return apply_patches([MonkeyPatchSpec(owner, attr, factory)])

    with pytest.raises(RuntimeError, match="body failure"):
        with activate():
            assert getattr(owner, attr) is replacement
            with activate():
                assert getattr(owner, attr) is replacement
                if context == "function":
                    assert plugin_system._PATCH_STATE[(owner, attr)]["count"] == 2
            assert getattr(owner, attr) is replacement
            raise RuntimeError("body failure")
    assert len(calls) == (1 if context == "function" else 2)
    _assert_restored_attribute(owner, attr, raw, local, resolved)
    if parent_namespace is not None:
        assert set(vars(parent)) == set(parent_namespace)
        assert all(
            vars(parent)[key] is value for key, value in parent_namespace.items()
        )
    assert not plugin_system._PATCH_STATE


@pytest.mark.parametrize(
    "kind", ["instance", "slot", "missing-instance", "missing-module"]
)
@pytest.mark.parametrize("spec_kind", ["assign", "factory"])
@pytest.mark.parametrize("delete_if_missing", [False, True])
def test_leaf_patch_preserves_instance_slot_and_missing_attribute_semantics(
    kind: str, spec_kind: str, delete_if_missing: bool
) -> None:
    original, replacement = object(), object()
    if kind == "slot":

        class Target:
            __slots__ = ("target",)

        owner = Target()
        owner.target = original
    elif kind == "missing-module":
        owner = ModuleType("missing_restoration_target")
    else:
        owner = SimpleNamespace()
        if kind == "instance":
            owner.target = original
    missing = kind.startswith("missing")
    calls = []

    def factory(value: Any) -> Any:
        calls.append(value)
        assert value is (None if missing else original)
        return replacement

    spec = (
        AssignSpec(owner, "target", replacement, delete_if_missing)
        if spec_kind == "assign"
        else MonkeyPatchSpec(owner, "target", factory, delete_if_missing)
    )
    with apply_patches([spec]):
        assert owner.target is replacement
    assert len(calls) == (0 if spec_kind == "assign" else 1)
    if missing:
        assert not hasattr(owner, "target")
    else:
        assert owner.target is original


@pytest.mark.parametrize("context", ["function", "leaf"])
def test_patch_context_unwinds_earlier_targets_when_later_factory_raises(
    monkeypatch: pytest.MonkeyPatch, context: str
) -> None:
    first, attr = _restoration_target("inherited")
    second, _ = _restoration_target("classmethod")
    snapshots = [
        (
            owner,
            inspect.getattr_static(owner, attr),
            attr in vars(owner),
            getattr(owner, attr),
        )
        for owner in (first, second)
    ]
    calls = []
    replacement = object()

    def first_factory(original: Any) -> Any:
        calls.append("first")
        return replacement

    def failing_factory(original: Any) -> Any:
        calls.append("second")
        assert getattr(first, attr) is replacement
        raise RuntimeError("entry failure")

    if context == "function":
        monkeypatch.setattr(
            plugin_system,
            "_iter_patch_specs",
            lambda: iter(
                [(first_factory, [first], attr), (failing_factory, [second], attr)]
            ),
        )
        active = apply_monkey_patches()
    else:
        active = apply_patches(
            [
                MonkeyPatchSpec(first, attr, first_factory),
                MonkeyPatchSpec(second, attr, failing_factory),
            ]
        )
    with pytest.raises(RuntimeError, match="entry failure"):
        with active:
            raise AssertionError("Entry failure must occur before yielding")
    assert calls == ["first", "second"]
    for owner, raw, local, resolved in snapshots:
        _assert_restored_attribute(owner, attr, raw, local, resolved)
    assert not plugin_system._PATCH_STATE


def test_leaf_duplicate_target_stack_restores_original_in_reverse_order() -> None:
    original, first, second = object(), object(), object()
    owner = SimpleNamespace(target=original)
    calls = []

    def factory(value: Any) -> Any:
        calls.append(value)
        assert value is first
        return second

    with pytest.raises(RuntimeError, match="duplicate body failure"):
        with apply_patches(
            [
                AssignSpec(owner, "target", first),
                MonkeyPatchSpec(owner, "target", factory),
            ]
        ):
            assert owner.target is second
            raise RuntimeError("duplicate body failure")
    assert len(calls) == 1 and calls[0] is first
    assert owner.target is original


@pytest.mark.parametrize("context", ["function", "leaf"])
def test_patch_context_propagates_inherited_shadow_restoration_failure(
    monkeypatch: pytest.MonkeyPatch, context: str
) -> None:
    rejection = {"active": False}

    class Meta(type):
        def __delattr__(cls, name: str) -> None:
            if name == "target" and rejection["active"]:
                raise RuntimeError("restore failure")
            super().__delattr__(name)

    class Parent:
        target = object()

    class Target(Parent, metaclass=Meta):
        pass

    def factory(original: Any) -> Any:
        return object()

    if context == "function":
        monkeypatch.setattr(
            plugin_system,
            "_iter_patch_specs",
            lambda: iter([(factory, [Target], "target")]),
        )
        active = apply_monkey_patches()
    else:
        active = apply_patches([MonkeyPatchSpec(Target, "target", factory)])
    try:
        with pytest.raises(RuntimeError, match="restore failure"):
            with active:
                rejection["active"] = True
    finally:
        rejection["active"] = False
        if "target" in vars(Target):
            type.__delattr__(Target, "target")
    assert not plugin_system._PATCH_STATE


def test_real_patch_targets_keep_exact_namespaces_after_activation_and_legacy_export() -> (
    None
):
    plugin_system.import_all_plugins()
    from flax import linen
    from jax2onnx.plugins.examples.eqx.dino import DinoRoPE as EqxDinoRoPE
    from jax2onnx.plugins.examples.nnx.dinov3 import DinoRoPE as NnxDinoRoPE

    targets = [
        EqxDinoRoPE,
        NnxDinoRoPE,
        linen.Conv,
        linen.ConvLocal,
        linen.MultiHeadAttention,
    ]
    snapshots = [
        (
            owner,
            inspect.getattr_static(owner, "__call__"),
            "__call__" in vars(owner),
            getattr(owner, "__call__"),
        )
        for owner in targets
    ]
    assert all(not local for _, _, local, _ in snapshots)
    with apply_monkey_patches():
        assert all("__call__" in vars(owner) for owner in targets[:3])
    for owner, raw, local, resolved in snapshots:
        _assert_restored_attribute(owner, "__call__", raw, local, resolved)
    model = jax2onnx.to_onnx(
        lambda x: x + jnp.asarray(1, dtype=jnp.float32),
        inputs=[jax.ShapeDtypeStruct((2, 3), np.float32)],
        opset=23,
    )
    assert len(model.graph.input) == len(model.graph.output) == 1
    assert any(node.op_type == "Add" for node in model.graph.node)
    for owner, raw, local, resolved in snapshots:
        _assert_restored_attribute(owner, "__call__", raw, local, resolved)
