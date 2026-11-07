# tests/extra_tests/framework/test_initializer_metadata_rope.py

"""A caller-owned freeze policy survives export of a small RoPE attention model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort  # type: ignore[import-untyped]
import optax

from jax2onnx import to_onnx


_PATH = "pkg.rope_fixture.path"
_TRAINABLE = "pkg.rope_fixture.trainable"


class _RoPE(eqx.Module):
    rotations: jax.Array

    def __init__(self) -> None:
        angles = jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 3
        self.rotations = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

    def __call__(self, x: jax.Array) -> jax.Array:
        table = jax.lax.stop_gradient(self.rotations)
        pairs = x.reshape(3, 2, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        c, s = table[..., 0], table[..., 1]
        return jnp.stack([a * c - b * s, a * s + b * c], axis=-1).reshape(3, 4)


class _Attention(eqx.Module):
    qkv: jax.Array
    rope: _RoPE
    proj: jax.Array

    def __init__(self, *, key: jax.Array) -> None:
        qkv_key, proj_key = jax.random.split(key)
        self.qkv = jax.random.normal(qkv_key, (4, 12), dtype=jnp.float32) / 5
        self.rope = _RoPE()
        self.proj = jax.random.normal(proj_key, (4, 4), dtype=jnp.float32) / 5

    def __call__(self, x: jax.Array) -> jax.Array:
        q, k, v = jnp.split(x @ self.qkv, 3, axis=-1)
        logits = self.rope(q) @ self.rope(k).T / 2
        return (jax.nn.softmax(logits, axis=-1) @ v) @ self.proj


def _path(key_path: Any) -> str:
    return str(jtu.keystr(key_path)).lstrip(".")


def _loss(model: _Attention, x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.mean((model(x) - y) ** 2)


def _adamw_step(model: _Attention, mask: _Attention, x: Any, y: Any) -> _Attention:
    params, frozen = eqx.partition(model, mask)
    optimizer = optax.adamw(1e-2, weight_decay=0.1)
    grads = eqx.filter_grad(lambda p: _loss(eqx.combine(p, frozen), x, y))(params)
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    return eqx.combine(eqx.apply_updates(params, updates), frozen)


def _restore_template(
    proto: onnx.ModelProto, template: _Attention
) -> tuple[_Attention, _Attention]:
    """A small downstream consumer: the template supplies the architecture.

    Only annotated tensors supply parameters; exporter-generated shape/index
    initializers do not participate in this consumer's training policy.
    """
    parameters = {}
    for tensor in proto.graph.initializer:
        props = {entry.key: entry.value for entry in tensor.metadata_props}
        if _PATH in props:
            assert props[_TRAINABLE] in ("true", "false")
            assert props[_PATH] not in parameters
            parameters[props[_PATH]] = (
                numpy_helper.to_array(tensor),
                props[_TRAINABLE] == "true",
            )
    assert set(parameters) == {
        _path(path) for path, _ in jtu.tree_leaves_with_path(template)
    }

    def load(path: Any, leaf: Any) -> jax.Array:
        value, _ = parameters[_path(path)]
        assert value.shape == leaf.shape
        assert value.dtype == leaf.dtype
        return jnp.asarray(value)

    restored = jtu.tree_map_with_path(load, template)
    mask = jtu.tree_map_with_path(lambda path, _: parameters[_path(path)][1], template)
    return restored, mask


def test_rope_freeze_policy_survives_export_and_adamw(tmp_path: Path) -> None:
    model = _Attention(key=jax.random.PRNGKey(0))
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10
    y = jnp.flip(x, axis=0)
    source_grad = eqx.filter_grad(_loss)(model, x, y)
    np.testing.assert_array_equal(source_grad.rope.rotations, 0)
    assert jnp.any(source_grad.qkv)

    # stop_gradient is not a freeze policy: decoupled weight decay still moves
    # an unmasked floating buffer whose mathematical gradient is zero.
    all_trainable = jtu.tree_map(lambda _: True, model)
    naive = _adamw_step(model, all_trainable, x, y)
    assert not jnp.array_equal(naive.rope.rotations, model.rope.rotations)

    # The framework explicitly selects this policy; the exporter does not
    # infer it from stop_gradient or interpret these application-owned keys.
    mask = jtu.tree_map_with_path(
        lambda path, _: _path(path) != "rope.rotations", model
    )
    by_identity = {
        id(leaf): {
            _PATH: _path(path),
            _TRAINABLE: "true" if trainable else "false",
        }
        for (path, leaf), trainable in zip(
            jtu.tree_leaves_with_path(model), jtu.tree_leaves(mask), strict=True
        )
    }
    proto = to_onnx(
        model,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        opset=18,
        input_names=["tokens"],
        output_names=["out"],
        initializer_metadata=lambda array: by_identity.get(id(array)),
    )
    destination = tmp_path / "rope.onnx"
    onnx.save_model(proto, destination)
    reloaded = onnx.load_model(destination)
    onnx.checker.check_model(reloaded, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        reloaded.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(["out"], {"tokens": np.asarray(x)})[0]
    np.testing.assert_allclose(actual, model(x), rtol=1e-5, atol=1e-6)

    template = eqx.filter_eval_shape(_Attention, key=jax.random.PRNGKey(1))
    imported, imported_mask = _restore_template(reloaded, template)
    assert imported_mask.qkv and imported_mask.proj
    assert not imported_mask.rope.rotations
    for source, restored in zip(
        jtu.tree_leaves(model), jtu.tree_leaves(imported), strict=True
    ):
        np.testing.assert_array_equal(restored, source)
    np.testing.assert_allclose(imported(x), actual, rtol=1e-5, atol=1e-6)
    imported_grad = eqx.filter_grad(_loss)(imported, x, y)
    for source, restored in zip(
        jtu.tree_leaves(source_grad), jtu.tree_leaves(imported_grad), strict=True
    ):
        np.testing.assert_allclose(restored, source, rtol=1e-5, atol=1e-6)
    trained = _adamw_step(imported, imported_mask, x, y)
    np.testing.assert_array_equal(trained.rope.rotations, imported.rope.rotations)
    assert not jnp.array_equal(trained.qkv, imported.qkv)
    assert not jnp.array_equal(trained.proj, imported.proj)
