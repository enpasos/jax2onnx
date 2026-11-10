# tests/extra_tests/test_state_inputs_contract.py

"""Executable form of the state-input contract in docs/user_guide/state_inputs.md.

Tests are named after the clauses they enforce (``test_K3_...``). The generic
``check_contract`` covers the file-level clauses K6-K10 for one stateful
callable and runs over a corpus of programs.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
import contextlib
import copy
from pathlib import Path
import re
from typing import Any

import equinox as eqx
from flax import linen, nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import StateCaptureError, StateInput, onnx_function, to_onnx

tu = jax.tree_util
REPOSITORY = Path(__file__).resolve().parents[2]
CHECKER_CLAUSES = {"K6", "K7", "K8", "K9", "K10"}
BF16 = np.dtype(jnp.bfloat16)


# --------------------------------------------------------------------------- helpers
def W(*shape: int, dtype: Any = jnp.float32, seed: int = 0) -> jax.Array:
    values = np.random.default_rng(seed).standard_normal(shape) * 0.5 + 0.1
    return jnp.asarray(values, dtype)


def X(*shape: int, dtype: Any = jnp.float32) -> jax.Array:
    return jnp.asarray(np.random.default_rng(7).standard_normal(shape) * 0.5, dtype)


def all_state(fn: Any) -> list[StateInput]:
    """Declare every distinct array leaf of the callable, in leaf order."""
    seen: set[int] = set()
    entries: list[StateInput] = []
    for leaf in tu.tree_leaves(fn):
        if isinstance(leaf, jax.Array) and id(leaf) not in seen:
            seen.add(id(leaf))
            index = len(entries)
            entries.append(
                StateInput(leaf, f"s{index}", {"pkg.demo.path": f"leaf{index}"})
            )
    return entries


def _specs(xs: list[jax.Array]) -> list[jax.ShapeDtypeStruct]:
    return [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in xs]


def _session(proto: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(
        proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _run(proto: onnx.ModelProto, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = _session(proto)
    if all(array.dtype != BF16 for array in feed.values()):
        return list(session.run(None, feed))
    # ONNX Runtime's NumPy interface rejects BF16 arrays, so bind their bits.
    binding = session.io_binding()
    for name, array in feed.items():
        if array.dtype == BF16:
            bits = np.ascontiguousarray(array).view(np.uint16)
            binding.bind_ortvalue_input(
                name,
                ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                    bits, onnx.TensorProto.BFLOAT16
                ),
            )
        else:
            binding.bind_cpu_input(name, array)
    for output in session.get_outputs():
        binding.bind_output(output.name)
    session.run_with_iobinding(binding)
    return list(binding.copy_outputs_to_cpu())


def _flat(out: Any) -> list[np.ndarray]:
    return [np.asarray(o) for o in tu.tree_leaves(out)]


def _widened(array: np.ndarray) -> np.ndarray:
    """BF16 values as FP32, which represents them exactly."""
    return array.astype(np.float32) if array.dtype == BF16 else array


def _close(got: list[np.ndarray], ref: list[np.ndarray], rtol: float) -> bool:
    return len(got) == len(ref) and all(
        g.shape == r.shape
        and g.dtype == r.dtype
        and np.allclose(_widened(g), _widened(r), rtol=rtol, atol=1e-4)
        for g, r in zip(got, ref)
    )


def _perturb(array: jax.Array) -> jax.Array:
    """Another value of the same shape, dtype and weak type."""
    if jnp.issubdtype(array.dtype, jnp.floating):
        return array * 1.07 + 0.03
    if array.dtype == jnp.bool_:
        return jnp.logical_not(array)
    return array + 1


def _metadata(tensor: onnx.TensorProto) -> dict[str, str]:
    return {prop.key: prop.value for prop in tensor.metadata_props}


def _without_state_payloads(proto: onnx.ModelProto, names: set[str]) -> bytes:
    stripped = copy.deepcopy(proto)
    for tensor in stripped.graph.initializer:
        if tensor.name in names:
            for payload in ("raw_data", "float_data", "double_data", "int32_data"):
                tensor.ClearField(payload)
    return stripped.SerializeToString(deterministic=True)


def check_contract(
    fn: Callable[..., Any],
    xs: list[jax.Array],
    entries: list[StateInput],
    *,
    unused: tuple[str, ...] = (),
    rtol: float = 1e-3,
    specs: list[Any] | None = None,
    runtime: bool = True,
    **export_options: Any,
) -> onnx.ModelProto:
    """Check the file-level clauses K6-K10 for one stateful callable.

    ``specs`` replaces the input specifications derived from ``xs``, for
    example to use symbolic dimensions. ``runtime=False`` stops after K9, for
    programs that ONNX Runtime has no kernels for.
    """
    specs = _specs(xs) if specs is None else specs
    leaves, treedef = tu.tree_flatten(fn)
    options: dict[str, Any] = {"allow_unused_state": bool(unused), **export_options}
    proto = to_onnx(fn, specs, state=entries, **options)
    onnx.checker.check_model(proto, full_check=True)
    bound = [entry for entry in entries if entry.name not in unused]
    inputs = list(proto.graph.input)
    initializers = {tensor.name: tensor for tensor in proto.graph.initializer}

    # K6: data inputs first, then overridable state in declaration order.
    assert [value.name for value in inputs[: len(xs)]] == [
        f"in_{index}" for index in range(len(xs))
    ]
    assert [value.name for value in inputs[len(xs) :]] == [
        entry.name for entry in bound if entry.overridable
    ]
    assert len(initializers) == len(proto.graph.initializer)
    assert not set(unused) & (set(initializers) | {value.name for value in inputs})
    for entry in bound:
        source = np.asarray(entry.array)
        tensor = initializers[entry.name]
        stored = onnx.numpy_helper.to_array(tensor)
        # K7: the array's bits, dtype and shape, without a cast.
        assert stored.dtype == source.dtype and stored.shape == source.shape
        assert stored.tobytes() == source.tobytes()
        # K8: exactly the declared metadata.
        assert _metadata(tensor) == dict(entry.metadata)
        if entry.overridable:
            info = next(v for v in inputs if v.name == entry.name).type.tensor_type
            assert info.elem_type == tensor.data_type
            assert [dim.dim_value for dim in info.shape.dim] == list(source.shape)

    # K10 with the saved defaults.
    feed = {f"in_{index}": np.asarray(x) for index, x in enumerate(xs)}
    reference = _flat(fn(*xs))
    if runtime:
        assert _close(_run(proto, feed), reference, rtol)

    # The same program with other state values of the same type.
    new = {id(entry.array): _perturb(entry.array) for entry in entries}
    assert all(
        (other.dtype, other.weak_type) == (entry.array.dtype, entry.array.weak_type)
        for entry, other in zip(entries, new.values())
    ), "the perturbation must keep the type"
    fn2 = tu.tree_unflatten(treedef, [new.get(id(leaf), leaf) for leaf in leaves])
    entries2 = [
        StateInput(new[id(e.array)], e.name, e.metadata, e.overridable) for e in entries
    ]
    reference2 = _flat(fn2(*xs))
    assert not _close(reference2, reference, rtol), "the perturbation must show"
    proto2 = to_onnx(fn2, specs, state=entries2, **options)

    # K9: only the state payloads may differ between the two files.
    names = {entry.name for entry in entries}
    assert _without_state_payloads(proto, names) == _without_state_payloads(
        proto2, names
    )
    if not runtime:
        return proto
    assert _close(_run(proto2, feed), reference2, rtol)

    # K10 under override: file A fed with B's values equals program B.
    overrides = {
        entry.name: np.asarray(new[id(entry.array)])
        for entry in bound
        if entry.overridable
    }
    if len(overrides) == len(bound):
        assert _close(_run(proto, {**feed, **overrides}), reference2, rtol)
    return proto


# --------------------------------------------------------------------------- programs
class Direct(eqx.Module):
    w: jax.Array
    b: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.w + self.b


class Scale(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * self.w + self.w


class Pair(eqx.Module):
    a: jax.Array
    c: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * self.a + jnp.tanh(self.c)


class Views(eqx.Module):
    w: jax.Array  # (2, 4)

    def __call__(self, x: jax.Array) -> jax.Array:
        w = self.w
        tiled = jnp.broadcast_to(w[0], (3, 4)).sum(0)
        return x + w.T.reshape(-1)[:4] + tiled + w[1] * 2.0


class Conv(eqx.Module):
    conv: eqx.nn.Conv2d

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


class Cond(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.cond(x.sum() > 0, lambda v: v * self.w, lambda v: v - self.w, x)


class Scan(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> Any:
        return jax.lax.scan(lambda c, _: (c * self.w + 1.0, c.sum()), x, None, length=3)


class While(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.while_loop(
            lambda c: c[1] < 3, lambda c: (c[0] * self.w + 1.0, c[1] + 1), (x, 0)
        )[0]


class WhilePredicate(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        limit = jnp.abs(self.w).sum() + 4.0
        return jax.lax.while_loop(lambda c: c.sum() < limit, lambda c: c + 0.01, x)


class Fori(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.fori_loop(0, 3, lambda i, c: c * self.w + 1.0, x)


class NestedJit(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.jit(lambda v: jax.jit(lambda u: u * self.w)(v) + 1.0)(x)


class JitInCondInScan(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> Any:
        def step(carry: jax.Array, _: Any) -> tuple[jax.Array, jax.Array]:
            scaled = jax.lax.cond(
                carry.sum() > 0,
                lambda v: jax.jit(lambda u: u * self.w)(v),
                lambda v: v + self.w,
                carry,
            )
            return scaled, scaled.sum()

        return jax.lax.scan(step, x, None, length=2)


class Checkpoint(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.checkpoint(lambda v: jnp.sin(v) * self.w)(x)


@jax.custom_jvp
def _scaled_sine(v: jax.Array, w: jax.Array) -> jax.Array:
    return jnp.sin(v) * w


_scaled_sine.defjvps(
    lambda tangent, out, v, w: tangent * jnp.cos(v) * w,
    lambda tangent, out, v, w: tangent * jnp.sin(v),
)


class CustomJvp(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return _scaled_sine(x, self.w)


@jax.custom_vjp
def _scaled_cosine(v: jax.Array, w: jax.Array) -> jax.Array:
    return jnp.cos(v) * w


_scaled_cosine.defvjp(
    lambda v, w: (jnp.cos(v) * w, (v, w)),
    lambda residual, g: (
        -g * jnp.sin(residual[0]) * residual[1],
        g * jnp.cos(residual[0]),
    ),
)


class CustomVjp(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return _scaled_cosine(x, self.w)


class Vmap(eqx.Module):
    w: jax.Array  # (4, 2)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.vmap(lambda row: row @ self.w)(x)


class CompileTimeEval(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        with jax.ensure_compile_time_eval():
            table = jnp.exp(self.w) * jnp.arange(4.0)
        return x + table


class Mask(eqx.Module):
    m: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.where(self.m, x, -x)


class Unused(eqx.Module):
    w: jax.Array
    dead: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * self.w


class CastAtUse(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * self.w.astype(x.dtype)


class NormBlock(eqx.Module):
    linear: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.softmax(jax.nn.gelu(self.norm(self.linear(x))))


class SelfAttention(eqx.Module):
    attention: eqx.nn.MultiheadAttention

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.attention(x, x, x)


class PatchEmbedding(eqx.Module):
    conv: eqx.nn.Conv2d
    cls: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        tokens = self.conv(x).reshape(self.cls.shape[0], -1).T
        return jnp.concatenate([self.cls[None, :], tokens], axis=0)


class Returned(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return x * self.w, self.w


class PassedThrough(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return x + 1.0, self.w


class NoData(eqx.Module):
    w: jax.Array

    def __call__(self) -> jax.Array:
        return self.w * 2.0


class OnlyState(eqx.Module):
    w: jax.Array

    def __call__(self) -> jax.Array:
        return self.w


class CallArguments(eqx.Module):
    w: jax.Array

    def __call__(
        self, x: jax.Array, *, key: Any = None, inference: bool = False
    ) -> jax.Array:
        if inference:
            return x * self.w
        return x * self.w * jax.random.bernoulli(key, 0.5, x.shape)


class Untraceable(eqx.Module):
    a: jax.Array
    c: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        raise AssertionError("violations of K1 and K2 must raise before tracing")


class Eager(eqx.Module):
    w: jax.Array
    s: jax.Array
    mode: str = eqx.field(static=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.mode == "float":
            return x * self.w + float(self.s)
        if self.mode == "numpy":
            return x * self.w + np.asarray(self.s) * 2.0
        if self.mode == "shape":
            return jnp.reshape(x, (int(self.s), -1)) * self.w
        return x * self.s + float(self.s)  # traced and eager read of one array


_GLOBAL_W = W(4, seed=3)


class LeakAtRoot(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * self.w + _GLOBAL_W


class LeakInBranch(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        branch = jax.lax.cond(x.sum() > 0, lambda v: v * _GLOBAL_W, lambda v: v, x)
        return branch * self.w


class LeakInLoop(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        looped = jax.lax.fori_loop(0, 2, lambda i, c: c + _GLOBAL_W, x)
        return looped * self.w


@onnx_function
class FunctionBlock(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.tanh(x * self.w)


@onnx_function
def _scaled_by_global(x: jax.Array) -> jax.Array:
    return x * _GLOBAL_W


@onnx_function
def _scaled_by_argument(x: jax.Array, w: jax.Array) -> jax.Array:
    return jnp.tanh(x * w)


class LeakInFunction(eqx.Module):
    w: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return _scaled_by_global(x) + self.w


class StateAsFunctionArgument(eqx.Module):
    w: jax.Array
    head: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return _scaled_by_argument(x, self.w) + self.head


class OuterWithFunctionBlock(eqx.Module):
    head: jax.Array
    block: FunctionBlock

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.block(x) + self.head


_DENSE = linen.Dense(2)


def _apply_dense(variables: Any, x: jax.Array) -> jax.Array:
    return _DENSE.apply(variables, x)


def _linen_partial() -> tuple[Any, list[jax.Array]]:
    variables = _DENSE.init(jax.random.PRNGKey(0), jnp.ones((1, 4)))
    return tu.Partial(_apply_dense, variables), [X(3, 4)]


def _tied() -> tuple[Any, list[jax.Array]]:
    shared = W(4)
    return Pair(shared, shared), [X(4)]


def _twins() -> tuple[Any, list[jax.Array]]:
    first = W(4)
    return Pair(first, jnp.array(first)), [X(4)]


def _named(overridable: bool) -> Callable[[Any], list[StateInput]]:
    def entries(fn: Any) -> list[StateInput]:
        return [
            StateInput(fn.w, "weight", {"pkg.demo.trainable": "true"}),
            StateInput(
                fn.b, "bias", {"pkg.demo.trainable": "false"}, overridable=overridable
            ),
        ]

    return entries


def _head_only(fn: Any) -> list[StateInput]:
    return [StateInput(fn.head, "head")]


def _fixed(fn: Any) -> list[StateInput]:
    return [StateInput(fn.w, "w", overridable=False)]


def _as_bfloat16(module: Any) -> Any:
    """The same module with every floating array converted to BF16."""
    return tu.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        module,
    )


def _bfloat16_block() -> tuple[Any, list[jax.Array]]:
    key = jax.random.PRNGKey(0)
    block = NormBlock(eqx.nn.Linear(8, 8, key=key), eqx.nn.LayerNorm(8))
    return _as_bfloat16(block), [X(8, dtype=jnp.bfloat16)]


def _bfloat16_attention() -> tuple[Any, list[jax.Array]]:
    attention = eqx.nn.MultiheadAttention(2, 8, key=jax.random.PRNGKey(0))
    return _as_bfloat16(SelfAttention(attention)), [X(5, 8, dtype=jnp.bfloat16)]


def _bfloat16_patches() -> tuple[Any, list[jax.Array]]:
    conv = eqx.nn.Conv2d(3, 4, 4, stride=4, key=jax.random.PRNGKey(0))
    return _as_bfloat16(PatchEmbedding(conv, W(4))), [X(3, 8, 8, dtype=jnp.bfloat16)]


def _inference_call(model: CallArguments, x: jax.Array) -> jax.Array:
    return model(x, inference=True)


Case = tuple[str, Callable[[], tuple[Any, list[jax.Array]]], dict[str, Any]]
CASES: list[Case] = [
    ("direct", lambda: (Direct(W(4, 2), W(2)), [X(3, 4)]), {}),
    ("tied_object_is_one_input", _tied, {"n_state": 1}),
    ("equal_values_stay_separate", _twins, {"n_state": 2}),
    ("derived_views", lambda: (Views(W(2, 4)), [X(4)]), {}),
    (
        "eqx_conv",
        lambda: (
            Conv(eqx.nn.Conv2d(3, 2, 3, key=jax.random.PRNGKey(0), dtype=jnp.float32)),
            [X(3, 8, 8)],
        ),
        {},
    ),
    ("cond_true_branch", lambda: (Cond(W(4)), [jnp.abs(X(4))]), {}),
    ("cond_false_branch", lambda: (Cond(W(4)), [-jnp.abs(X(4))]), {}),
    ("scan", lambda: (Scan(W(4)), [X(4)]), {}),
    ("while_loop", lambda: (While(W(4)), [X(4)]), {}),
    ("while_predicate", lambda: (WhilePredicate(W(4)), [X(4)]), {}),
    ("fori_loop", lambda: (Fori(W(4)), [X(4)]), {}),
    ("nested_jit", lambda: (NestedJit(W(4)), [X(4)]), {}),
    ("jit_in_cond_in_scan", lambda: (JitInCondInScan(W(4)), [X(4)]), {}),
    ("checkpoint", lambda: (Checkpoint(W(4)), [X(4)]), {}),
    ("custom_jvp", lambda: (CustomJvp(W(4)), [X(4)]), {}),
    ("custom_vjp", lambda: (CustomVjp(W(4)), [X(4)]), {}),
    ("vmap", lambda: (Vmap(W(4, 2)), [X(3, 4)]), {}),
    ("compile_time_eval_stays_in_graph", lambda: (CompileTimeEval(W(4)), [X(4)]), {}),
    ("nnx_linear", lambda: (nnx.Linear(4, 2, rngs=nnx.Rngs(0)), [X(3, 4)]), {}),
    (
        "nnx_linear_general",
        lambda: (nnx.LinearGeneral(4, (2, 2), rngs=nnx.Rngs(0)), [X(3, 4)]),
        {},
    ),
    (
        "nnx_conv",
        lambda: (nnx.Conv(2, 3, kernel_size=(1, 1), rngs=nnx.Rngs(0)), [X(1, 4, 4, 2)]),
        {},
    ),
    (
        "nnx_batch_norm",
        lambda: (
            nnx.BatchNorm(3, use_running_average=True, rngs=nnx.Rngs(0)),
            [X(2, 3)],
        ),
        {},
    ),
    ("nnx_layer_norm", lambda: (nnx.LayerNorm(3, rngs=nnx.Rngs(0)), [X(2, 3)]), {}),
    ("linen_through_partial", _linen_partial, {}),
    (
        "eqx_call_arguments_through_partial",
        lambda: (tu.Partial(_inference_call, CallArguments(W(4))), [X(4)]),
        {},
    ),
    (
        "float16",
        lambda: (Scale(W(4, dtype=jnp.float16)), [X(4, dtype=jnp.float16)]),
        {"rtol": 2e-2},
    ),
    (
        "bfloat16_at_file_level",
        lambda: (Scale(W(4, dtype=jnp.bfloat16)), [X(4, dtype=jnp.bfloat16)]),
        {"runtime": False},
    ),
    ("bfloat16_linear_norm_gelu_softmax", _bfloat16_block, {"runtime": False}),
    ("bfloat16_attention", _bfloat16_attention, {"runtime": False}),
    ("bfloat16_patch_embedding", _bfloat16_patches, {"runtime": False}),
    ("bfloat16_cast_at_use", lambda: (CastAtUse(W(4, dtype=jnp.bfloat16)), [X(4)]), {}),
    (
        "float64_under_x64",
        lambda: (Scale(W(4, dtype=jnp.float64)), [X(4, dtype=jnp.float64)]),
        {"x64": True},
    ),
    (
        "int32",
        lambda: (
            Scale(jnp.arange(1, 5, dtype=jnp.int32)),
            [jnp.arange(4, dtype=jnp.int32)],
        ),
        {},
    ),
    ("bool", lambda: (Mask(jnp.asarray([True, False, True, False])), [X(4)]), {}),
    ("scalar", lambda: (Scale(jnp.asarray(0.5)), [X(4)]), {}),
    (
        "weak_typed_scalar",
        lambda: (eqx.nn.PReLU(), [jnp.asarray([-1.0, 2.0, -3.0, 4.0], jnp.float32)]),
        {},
    ),
    (
        "weak_scalar_with_float16_data",
        lambda: (Scale(jnp.asarray(0.5)), [X(4, dtype=jnp.float16)]),
        {"rtol": 2e-2},
    ),
    (
        "named_entries",
        lambda: (Direct(W(4, 2), W(2)), [X(3, 4)]),
        {"entries": _named(True)},
    ),
    (
        "non_overridable",
        lambda: (Direct(W(4, 2), W(2)), [X(3, 4)]),
        {"entries": _named(False)},
    ),
    (
        "unused_with_opt_in",
        lambda: (Unused(W(4), W(4, seed=1)), [X(4)]),
        {"unused": ("s1",)},
    ),
    ("state_used_and_returned", lambda: (Returned(W(4)), [X(4)]), {}),
    ("state_only_returned", lambda: (PassedThrough(W(4)), [X(4)]), {}),
    (
        "state_only_returned_non_overridable",
        lambda: (PassedThrough(W(4)), [X(4)]),
        {"entries": _fixed},
    ),
    ("no_data_input", lambda: (NoData(W(4)), []), {}),
    ("state_is_the_only_output", lambda: (OnlyState(W(4)), []), {}),
    (
        "state_as_function_argument",
        lambda: (StateAsFunctionArgument(W(4), W(4, seed=1)), [X(4)]),
        {},
    ),
    (
        "undeclared_arrays_in_function_module",
        lambda: (OuterWithFunctionBlock(W(4), FunctionBlock(W(4, seed=1))), [X(4)]),
        {"entries": _head_only},
    ),
]


@pytest.mark.parametrize("name,build,options", CASES, ids=[case[0] for case in CASES])
def test_K6_to_K10_hold_for(
    name: str, build: Callable[..., Any], options: dict[str, Any]
) -> None:
    with jax.enable_x64() if options.get("x64") else contextlib.nullcontext():
        fn, xs = build()
        entries = options.get("entries", all_state)(fn)
        if "n_state" in options:
            assert len(entries) == options["n_state"]
        check_contract(
            fn,
            xs,
            entries,
            unused=options.get("unused", ()),
            rtol=options.get("rtol", 1e-3),
            runtime=options.get("runtime", True),
        )


# --------------------------------------------------------------------------- K1, K2: declaration
def _export(fn: Any, entries: list[StateInput], **kwargs: Any) -> onnx.ModelProto:
    return to_onnx(fn, _specs([X(4)]), state=entries, **kwargs)


def test_K1_numpy_leaves_and_tracers_are_rejected() -> None:
    with pytest.raises(TypeError, match="concrete jax.Array"):
        StateInput(np.ones(4, np.float32), "w")
    with pytest.raises(TypeError, match="concrete jax.Array"):
        StateInput(1.0, "w")
    with pytest.raises(TypeError, match="concrete jax.Array"):
        jax.make_jaxpr(lambda w: StateInput(w, "w").array)(jnp.ones(4))


@pytest.mark.parametrize("name", ["", "layers.0.weight", "0w", "w-1"])
def test_K1_names_are_c90_identifiers(name: str) -> None:
    with pytest.raises(ValueError, match="C90 identifier"):
        StateInput(W(4), name)


def test_K1_complex_and_empty_state_are_rejected() -> None:
    with pytest.raises(TypeError, match="complex"):
        StateInput(jnp.ones(4, jnp.complex64), "w")
    with pytest.raises(ValueError, match="empty"):
        StateInput(jnp.ones((0, 4)), "w")


def test_K1_names_and_objects_are_unique() -> None:
    model = Untraceable(W(4), W(4, seed=1))
    with pytest.raises(ValueError, match="unique"):
        _export(model, [StateInput(model.a, "w"), StateInput(model.c, "w")])
    with pytest.raises(ValueError, match="already declares"):
        _export(model, [StateInput(model.a, "a"), StateInput(model.a, "b")])


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"input_names": ["taken"]},
        {"output_names": ["taken"]},
        {"input_params": {"taken": True}},
    ],
    ids=["default_input", "input_names", "output_names", "input_params"],
)
def test_K1_names_differ_from_other_interface_names(options: dict[str, Any]) -> None:
    model = Untraceable(W(4), W(4, seed=1))
    name = "taken" if options else "in_0"
    with pytest.raises(ValueError, match="collide"):
        _export(model, [StateInput(model.a, name)], **options)


def test_K1_state_must_be_a_sequence_of_entries() -> None:
    model = Untraceable(W(4), W(4, seed=1))
    with pytest.raises(TypeError, match="sequence of StateInput"):
        _export(model, StateInput(model.a, "w"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be StateInput"):
        _export(model, [model.a])  # type: ignore[list-item]


def test_K2_state_must_be_a_leaf_of_the_callable() -> None:
    model = Untraceable(W(4), W(4, seed=1))
    with pytest.raises(StateCaptureError, match="pytree leaves"):
        _export(lambda x: model(x), [StateInput(model.a, "w")])


# --------------------------------------------------------------------------- K3, K4, K5: tracing
@pytest.mark.parametrize("mode", ["float", "numpy", "shape", "mixed"])
def test_K3_eager_reads_fail_instead_of_baking(mode: str) -> None:
    model = Eager(W(4), jnp.asarray(1 if mode == "shape" else 0.5), mode)
    entries = [StateInput(model.s, "s")] if mode == "mixed" else all_state(model)
    with pytest.raises(
        (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError)
    ) as raised:
        _export(model, entries)
    assert any("state=" in note for note in raised.value.__notes__)


@pytest.mark.parametrize(
    "leaky",
    [LeakAtRoot, LeakInBranch, LeakInLoop, LeakInFunction],
    ids=["root", "cond_branch", "loop_body", "onnx_function_body"],
)
def test_K4_a_second_route_as_constant_is_rejected(leaky: Callable[..., Any]) -> None:
    model = leaky(_GLOBAL_W)
    with pytest.raises(StateCaptureError, match="also captured as a constant"):
        _export(model, [StateInput(model.w, "w")])


def test_K5_unused_state_needs_an_opt_in() -> None:
    model = Unused(W(4), W(4, seed=1))
    with pytest.raises(StateCaptureError, match="not read"):
        _export(model, all_state(model))
    with pytest.raises(TypeError, match="must be a bool"):
        _export(model, all_state(model), allow_unused_state=1)


# --------------------------------------------------------------------------- K6, K7, K10: interface
@pytest.mark.parametrize("opset", [13, 17, 23])
def test_K6_onnx_runtime_lists_state_as_overridable_initializers(opset: int) -> None:
    model = Direct(W(4, 2), W(2))
    proto = check_contract(model, [X(3, 4)], _named(True)(model), opset=opset)
    assert {entry.domain: entry.version for entry in proto.opset_import}[""] == opset
    session = _session(proto)
    assert [value.name for value in session.get_inputs()] == ["in_0"]
    assert {value.name for value in session.get_overridable_initializers()} == {
        "weight",
        "bias",
    }


@pytest.mark.parametrize("x64", [False, True])
def test_K6_weak_typed_state_keeps_float16_data(x64: bool) -> None:
    def weak_fp32(value: float) -> jax.Array:
        with jax.enable_x64(False):  # whatever the caller configured
            return jnp.asarray(value)

    with jax.enable_x64(x64):
        model = Scale(weak_fp32(0.5))
        assert model.w.weak_type and model.w.dtype == jnp.float32
        x = jnp.asarray([-2.0, 0.0, 3.0], jnp.float16)
        proto = to_onnx(model, _specs([x]), state=[StateInput(model.w, "scale")])
        default = next(t for t in proto.graph.initializer if t.name == "scale")
        assert default.data_type == onnx.TensorProto.FLOAT and not default.dims
        for value, feed in [
            (0.5, {"in_0": np.asarray(x)}),
            (0.25, {"in_0": np.asarray(x), "scale": np.asarray(0.25, np.float32)}),
        ]:
            expected = np.asarray(Scale(weak_fp32(value))(x))
            assert expected.dtype == np.float16
            (got,) = _run(proto, feed)
            assert got.dtype == np.float16
            np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(
    "spec",
    [("B", 4), lambda: jax.ShapeDtypeStruct(jax.export.symbolic_shape("B, 4"), "f4")],
    ids=["tuple", "shape_dtype_struct"],
)
def test_K10_symbolic_data_dimensions_stay_dynamic(spec: Any) -> None:
    model = Direct(W(4, 2), W(2))
    spec = spec() if callable(spec) else spec
    proto = check_contract(model, [X(3, 4)], _named(True)(model), specs=[spec])
    dims = proto.graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_param == "B" and dims[1].dim_value == 4
    other = Direct(W(4, 2, seed=1), W(2, seed=2))
    for batch in (1, 5):
        x = X(batch, 4)
        feed = {
            "in_0": np.asarray(x),
            "weight": np.asarray(other.w),
            "bias": np.asarray(other.b),
        }
        assert _close(_run(proto, feed), _flat(other(x)), 1e-3)


def test_K7_defaults_keep_signed_zeros_and_nan_payloads() -> None:
    bits = np.array(
        [0, 0x80000000, 0x7FC00011, 0xFFC00022, 0x3F800000, 0xBF800000], np.uint32
    )
    model = Scale(jnp.asarray(bits.view(np.float32)))
    proto = to_onnx(model, _specs([X(6)]), state=[StateInput(model.w, "w")])
    default = next(t for t in proto.graph.initializer if t.name == "w")
    assert onnx.numpy_helper.to_array(default).tobytes() == bits.tobytes()


def test_K7_state_is_never_narrowed() -> None:
    with jax.enable_x64(True):
        model = Scale(W(4, dtype=jnp.float64))
    with jax.enable_x64(False), pytest.raises(ValueError, match="narrowed"):
        _export(model, [StateInput(model.w, "w")])


# --------------------------------------------------------------------------- K8, K11, K12
@pytest.mark.parametrize(
    "metadata, error, message",
    [
        ({"pkg.jax2onnx.x": "1"}, ValueError, "reserved"),
        ({"k": 1}, TypeError, "strings"),
        ({1: "v"}, TypeError, "strings"),
        ({"": "v"}, TypeError, "strings"),
        ([("k", "v")], TypeError, "mapping"),
        ({"\ud800": "v"}, ValueError, "UTF-8"),
        ({"k": "\ud800"}, ValueError, "UTF-8"),
    ],
    ids=[
        "reserved_prefix",
        "value_type",
        "key_type",
        "empty_key",
        "not_a_mapping",
        "key_encoding",
        "value_encoding",
    ],
)
def test_K8_metadata_keys_and_values_are_validated(
    metadata: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        StateInput(W(4), "w", metadata)


def test_K8_metadata_values_may_be_empty_or_non_ascii() -> None:
    model = Scale(W(4))
    labels = {"pkg.demo.path": "blocks/π/weight", "pkg.demo.note": ""}
    proto = _export(model, [StateInput(model.w, "w", labels)])
    restored = onnx.load_model_from_string(proto.SerializeToString())
    default = next(t for t in restored.graph.initializer if t.name == "w")
    assert _metadata(default) == labels


@pytest.mark.parametrize("export_mode", ["standard", "web"])
def test_K8_file_exports_keep_defaults_and_metadata(
    tmp_path: Path, export_mode: str
) -> None:
    model = Direct(W(600, 600), W(600))  # the weight exceeds the external-data limit
    entries = _named(False)(model)
    path = to_onnx(
        model,
        _specs([X(2, 600)]),
        state=entries,
        return_mode="file",
        output_path=tmp_path / "model.onnx",
        export_mode=export_mode,  # type: ignore[arg-type]
    )
    raw = {
        t.name: t for t in onnx.load(path, load_external_data=False).graph.initializer
    }
    is_external = raw["weight"].data_location == onnx.TensorProto.EXTERNAL
    assert is_external == (export_mode == "standard")
    loaded = onnx.load(path)
    tensors = {tensor.name: tensor for tensor in loaded.graph.initializer}
    assert _metadata(tensors["weight"]) == {"pkg.demo.trainable": "true"}
    assert _metadata(tensors["bias"]) == {"pkg.demo.trainable": "false"}
    stored = onnx.numpy_helper.to_array(tensors["weight"])
    assert stored.tobytes() == np.asarray(model.w).tobytes()
    assert [value.name for value in loaded.graph.input] == ["in_0", "weight"]


def test_K8_ir_models_carry_the_defaults() -> None:
    model = Direct(W(4, 2), W(2))
    ir_model = to_onnx(
        model, _specs([X(3, 4)]), state=_named(True)(model), return_mode="ir"
    )
    weight = ir_model.graph.initializers["weight"]
    assert weight.is_graph_input() and weight.const_value is not None
    assert dict(weight.const_value.metadata_props) == {"pkg.demo.trainable": "true"}


def test_K11_declared_state_inside_a_function_module_fails() -> None:
    model = FunctionBlock(W(4))
    with pytest.raises(StateCaptureError, match="ONNX function body"):
        _export(model, [StateInput(model.w, "w")])


def test_K11_layout_options_cannot_address_state() -> None:
    model = Scale(W(4))
    with pytest.raises(ValueError, match="data inputs"):
        _export(model, [StateInput(model.w, "w")], inputs_as_nchw=[1])


def test_K12_exports_without_state_are_unchanged_and_exports_repeat() -> None:
    model = Direct(W(4, 2), W(2))
    specs = _specs([X(3, 4)])

    def exported(**kwargs: Any) -> bytes:
        return to_onnx(model, specs, **kwargs).SerializeToString(deterministic=True)

    assert exported() == exported(state=None) == exported(state=())
    assert exported(allow_unused_state=True) == exported()
    assert exported(state=all_state(model)) == exported(state=all_state(model))


@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"]
)
def test_K10_api_matches_the_manual_export_with_explicit_state(dtype: Any) -> None:
    """The definition: a state export is the ordinary export of g(*data, *state)."""
    model = Direct(W(4, 2, dtype=dtype), W(2, dtype=dtype))
    specs = _specs([X(3, 4, dtype=dtype)])
    api = to_onnx(model, specs, state=_named(True)(model))

    def explicit(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
        return Direct(weight, bias)(x)

    manual = to_onnx(
        explicit,
        [*specs, *_specs([model.w, model.b])],
        input_names=["in_0", "weight", "bias"],
    )
    without_defaults = copy.deepcopy(api)
    kept = [t for t in api.graph.initializer if t.name not in {"weight", "bias"}]
    del without_defaults.graph.initializer[:]
    without_defaults.graph.initializer.extend(kept)
    assert without_defaults.SerializeToString(
        deterministic=True
    ) == manual.SerializeToString(deterministic=True)


# --------------------------------------------------------------------------- contract hygiene
def test_every_clause_in_the_guide_has_a_test_and_vice_versa() -> None:
    guide = (REPOSITORY / "docs/user_guide/state_inputs.md").read_text()
    documented = set(re.findall(r"^\| (K\d+) ", guide, flags=re.MULTILINE))
    tested = set(CHECKER_CLAUSES)
    for name in globals():
        if name.startswith("test_K"):
            span = re.fullmatch(r"test_K(\d+)_to_K(\d+)_.*", name)
            if span is not None:
                first, last = int(span.group(1)), int(span.group(2))
                tested.update(f"K{number}" for number in range(first, last + 1))
            else:
                tested.update(re.findall(r"K\d+", name))
    assert documented == tested


def test_guide_examples_run_as_one_session() -> None:
    guide = REPOSITORY / "docs/user_guide/state_inputs.md"
    blocks = re.findall(
        r"^```python\n(.*?)^```", guide.read_text(), flags=re.MULTILINE | re.DOTALL
    )
    assert len(blocks) >= 5
    with jax.enable_x64(False):  # the examples assume JAX's default precision
        exec(compile("\n".join(blocks), str(guide), "exec"), {"__name__": "guide"})


def test_captured_constants_have_one_binding_point() -> None:
    """K4 relies on every scope binding its captures through the context."""
    offenders: list[str] = []
    for root in ("jax2onnx/converter", "jax2onnx/plugins"):
        for path in sorted((REPOSITORY / root).rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.For):
                    continue
                if "constvars" not in ast.unparse(node.iter):
                    continue
                calls = {
                    call.func.attr
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                }
                if "bind_const_for_var" in calls:
                    offenders.append(f"{path.relative_to(REPOSITORY)}:{node.lineno}")
    assert not offenders, (
        "Bind a traced JAXPR's constants with ctx.bind_captured_arrays(...): "
        + ", ".join(offenders)
    )
