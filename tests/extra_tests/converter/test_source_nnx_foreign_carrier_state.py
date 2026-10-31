# tests/extra_tests/converter/test_source_nnx_foreign_carrier_state.py

"""Foreign dataclass numeric fields are explicit SOURCE Function state."""

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import FunctionType
from typing import Any, ClassVar

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import Tracer
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins.flax.nnx._source_function_state import (
    NNXFunctionState,
    prepare_nnx_function_state,
)
from tests.extra_tests.converter.test_source_callable_preparation import (
    _preserve_settings as _preserve_settings,
)


@dataclass
class _ForeignTables:
    cos: Any
    sin: Any
    width: int = 3
    base: float = 8.0
    constructions: ClassVar[int] = 0

    def __post_init__(self) -> None:
        type(self).constructions += 1


@onnx_function
class _ForeignOwner(nnx.Module):
    def __init__(self, shared: bool = False, equal_distinct: bool = False) -> None:
        self.cos0 = jnp.asarray(np.arange(6).reshape(2, 3) / 2, np.float32)
        self.cos1 = self.cos2 = self.cos0
        self.sin0 = jnp.asarray(np.arange(6).reshape(2, 3) + 2, np.float32)
        self.sin1 = self.sin2 = self.sin0
        self.rope0 = _ForeignTables(self.cos0, self.sin0)
        self.rope1 = self.rope0 if shared else _ForeignTables(self.cos0, self.sin0)
        if equal_distinct:
            self.rope1 = _ForeignTables(jnp.array(self.cos0), jnp.array(self.sin0))
        self.counter = nnx.Variable(jnp.asarray(0, np.float32))

    def __call__(self, x: Any, *, advance: bool = False, fault: str = "") -> Any:
        assert self.rope0.cos is self.cos0 is self.cos1 is self.cos2
        assert self.rope0.sin is self.sin0 is self.sin1 is self.sin2
        if advance:
            self.counter[...] = self.counter[...] + np.float32(1)
        if fault == "numeric":
            self.cos0 = self.cos1 = self.cos2 = self.rope0.cos = self.rope1.cos = (
                self.cos0 + np.float32(1)
            )
        elif fault == "metadata":
            self.rope0.width += 1
        elif fault == "alias":
            self.rope0.cos = jnp.array(self.cos0)
        elif fault == "holder":
            replacement = object.__new__(type(self.rope0))
            for name in ("cos", "sin", "width", "base"):
                object.__setattr__(replacement, name, getattr(self.rope0, name))
            self.rope0 = replacement
        elif fault in {"shape", "dtype"}:
            replacement = (
                self.cos0[:, :2] if fault == "shape" else self.cos0.astype(jnp.float16)
            )
            self.cos0 = self.cos1 = self.cos2 = self.rope0.cos = self.rope1.cos = (
                replacement
            )
        elif fault == "signed-zero":
            self.rope0.base = -0.0
        return x if fault in {"shape", "dtype"} else x * self.rope0.cos + self.rope0.sin


def _prepared(owner: Any) -> tuple[NNXFunctionState, tuple[Any, ...]]:
    result = prepare_nnx_function_state(owner, type(owner).__call__, 17)
    assert result is not None
    return result


def _payload_free(value: Any, original: tuple[Any, ...], visited: set[int]) -> None:
    if id(value) in visited:
        return
    visited.add(id(value))
    assert all(value is not carrier for carrier in original)
    assert not isinstance(value, (jax.Array, Tracer, np.ndarray, np.generic))
    if isinstance(value, (type, FunctionType)):
        return
    if is_dataclass(value):
        for field in fields(value):
            _payload_free(getattr(value, field.name), original, visited)
    elif isinstance(value, dict):
        for key, item in value.items():
            _payload_free(key, original, visited)
            _payload_free(item, original, visited)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _payload_free(item, original, visited)


@pytest.mark.parametrize(
    "shared,equal_distinct", [(False, False), (True, False), (False, True)]
)
def test_foreign_carrier_reconstruction_preserves_actual_class_and_alias_layout(
    shared: bool, equal_distinct: bool
) -> None:
    owner = _ForeignOwner(shared, equal_distinct)
    original = (owner.rope0, owner.rope1)
    constructions = _ForeignTables.constructions
    adapter, state = _prepared(owner)
    _payload_free(adapter, original, set())
    hash(adapter.signature)
    clone = adapter.reconstruct(state)
    assert _ForeignTables.constructions == constructions
    assert clone is not owner
    assert type(clone.rope0) is type(clone.rope1) is _ForeignTables
    assert all(
        clone.rope0 is not carrier and clone.rope1 is not carrier
        for carrier in original
    )
    assert (clone.rope0 is clone.rope1) is shared
    assert clone.rope0.cos is clone.cos0 is clone.cos1 is clone.cos2
    assert clone.rope0.sin is clone.sin0 is clone.sin1 is clone.sin2
    assert (clone.rope0.cos is clone.rope1.cos) is (not equal_distinct)
    assert (clone.rope0.sin is clone.rope1.sin) is (not equal_distinct)
    replacement = tuple(jnp.asarray(value) + np.float32(4) for value in state)
    second = adapter.reconstruct(replacement)
    assert second.rope0 is not clone.rope0
    assert second.rope0.cos is second.cos0 is second.cos1 is second.cos2
    assert second.rope0.sin is second.sin0 is second.sin1 is second.sin2
    assert any(second.rope0.cos is value for value in replacement)
    assert any(second.rope0.sin is value for value in replacement)
    assert owner.rope0 is original[0] and owner.rope1 is original[1]
    assert owner.rope0.cos is owner.cos0 and owner.rope0.sin is owner.sin0


@pytest.mark.parametrize(
    "fault", ["numeric", "metadata", "alias", "holder", "shape", "dtype", "signed-zero"]
)
def test_foreign_carrier_mutation_cannot_apply_unrelated_updates_to_caller(
    fault: str,
) -> None:
    owner = _ForeignOwner()
    if fault == "signed-zero":
        owner.rope0.base = owner.rope1.base = 0.0
    adapter, state = _prepared(owner)
    identities = (
        owner.counter,
        owner.counter.get_raw_value(),
        owner.rope0,
        owner.rope1,
        owner.cos0,
        owner.sin0,
    )
    with pytest.raises(
        SourcePrecisionError, match="foreign numeric carrier support remains unfinished"
    ) as failure:
        adapter.invoke(
            (jnp.ones((2, 3), np.float32),), {"advance": True, "fault": fault}, state
        )
    assert failure.value.category == "lowering-defect"
    if fault == "signed-zero":
        assert not np.signbit(owner.rope0.base)
    assert (
        owner.counter is identities[0]
        and owner.counter.get_raw_value() is identities[1]
    )
    assert owner.rope0 is identities[2] and owner.rope1 is identities[3]
    assert owner.cos0 is identities[4] and owner.sin0 is identities[5]
    assert owner.rope0.width == 3 and owner.rope0.cos is owner.cos0
    assert all(a is b for a, b in zip(adapter._owner_state(owner), state, strict=True))


def test_unrelated_variable_effects_still_update_without_replacing_carriers() -> None:
    owner = _ForeignOwner()
    adapter, state = _prepared(owner)
    original = (owner.counter, owner.rope0, owner.rope1, owner.cos0, owner.sin0)
    _, updated, effects = adapter.invoke(
        (jnp.ones((2, 3), np.float32),), {"advance": True}, state
    )
    assert effects
    checked = adapter.with_effects(effects)
    checked.update_owner(owner, updated)
    assert owner.counter is original[0] and np.asarray(
        owner.counter[...]
    ) == np.float32(1)
    assert owner.rope0 is original[1] and owner.rope1 is original[2]
    assert owner.cos0 is original[3] and owner.sin0 is original[4]


@pytest.mark.parametrize("failure_call", [1, 2])
def test_unrelated_update_failure_preserves_foreign_carriers_and_state(
    failure_call: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _ForeignOwner()
    adapter, state = _prepared(owner)
    _, updated, effects = adapter.invoke(
        (jnp.ones((2, 3), np.float32),), {"advance": True}, state
    )
    checked = adapter.with_effects(effects)
    original = (
        owner.counter,
        owner.counter.get_raw_value(),
        owner.rope0,
        owner.rope1,
        owner.cos0,
        owner.sin0,
    )
    native_update = nnx.update
    calls = 0

    def fail_after_update(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        result = native_update(*args, **kwargs)
        if calls == failure_call:
            raise RuntimeError("injected unrelated state update")
        return result

    with monkeypatch.context() as context:
        context.setattr(nnx, "update", fail_after_update)
        with pytest.raises(RuntimeError, match="injected unrelated state update"):
            checked.update_owner(owner, updated)
    assert owner.counter is original[0] and owner.counter.get_raw_value() is original[1]
    assert owner.rope0 is original[2] and owner.rope1 is original[3]
    assert owner.cos0 is original[4] and owner.sin0 is original[5]
    assert owner.rope0.cos is owner.rope1.cos is owner.cos0 is owner.cos1 is owner.cos2
    assert all(a is b for a, b in zip(checked._owner_state(owner), state, strict=True))


def test_foreign_only_numeric_fields_have_explicit_ports() -> None:
    class Owner(nnx.Module):
        def __init__(self) -> None:
            self.rope = _ForeignTables(
                jnp.asarray(np.arange(6).reshape(2, 3), np.float32),
                jnp.asarray(np.ones((2, 3)), np.float32),
            )

        def __call__(self, x: Any) -> Any:
            return x * self.rope.cos + self.rope.sin

    owner = Owner()
    adapter, state = _prepared(owner)
    assert adapter.state_count == 2 and len(adapter.carriers) == 1
    assert adapter.entries == ()
    replacements = tuple(value + np.float32(2) for value in state)
    clone = adapter.reconstruct(replacements)
    assert clone.rope.cos is replacements[0] and clone.rope.sin is replacements[1]
    assert clone.rope is not owner.rope


@pytest.mark.parametrize("optimization", ["disabled", "all"])
def test_foreign_carrier_state_is_dynamic_in_trace_named_function_and_cpu(
    optimization: str, tmp_path: Path
) -> None:
    template = _ForeignOwner()
    adapter, state = _prepared(template)

    def call(x: Any, *carried: Any) -> Any:
        owner = adapter.reconstruct(tuple(carried))
        return owner(x)

    x = jnp.asarray(np.arange(6).reshape(2, 3), np.float32)
    replaced = tuple(value + np.float32(2) for value in state)
    fresh = jax.jit(call)
    first = fresh(x, *state)
    changed = fresh(x, *replaced)
    eager = call(x, *replaced)
    rebuilt = adapter.reconstruct(replaced)
    expected = np.asarray(x) * np.asarray(rebuilt.cos0) + np.asarray(rebuilt.sin0)
    assert not np.array_equal(np.asarray(first), expected)
    np.testing.assert_array_equal(np.asarray(changed), expected)
    np.testing.assert_array_equal(np.asarray(eager), expected)
    trace = jax.make_jaxpr(call)(x, *replaced)
    assert not trace.consts
    model = ui._to_onnx(
        call,
        [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in (x, *replaced)],
        precision_policy=PrecisionPolicy.SOURCE,
        opset=23,
    )
    path = tmp_path / "foreign-carried.onnx"
    path.write_bytes(model.SerializeToString())
    loaded = onnx.load_model(path)
    onnx.checker.check_model(loaded, full_check=True)
    assert any("_ForeignOwner" in function.name for function in loaded.functions)
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if optimization == "disabled"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    feeds = {
        port.name: np.asarray(value)
        for port, value in zip(session.get_inputs(), (x, *replaced), strict=True)
    }
    returned = session.run(None, feeds)
    np.savez(
        tmp_path / "actual-returned.npz",
        **{f"leaf_{i}": value for i, value in enumerate(returned)},
    )
    assert len(returned) == 1 and returned[0].dtype == np.float32
    np.testing.assert_array_equal(returned[0], expected)
    assert template.rope0.cos is template.cos0 and template.rope0.sin is template.sin0


@pytest.mark.parametrize("ingress", ["original", "reconstruct", "invoke"])
def test_foreign_mutable_numpy_buffers_fail_before_body_or_caller_write(
    ingress: str,
) -> None:
    owner = _ForeignOwner()
    if ingress == "original":
        owner.rope0.cos = np.asarray(owner.cos0).copy()
        original_buffer = owner.rope0.cos.copy()
        with pytest.raises(
            SourcePrecisionError, match="mutable foreign-linked buffers"
        ):
            _prepared(owner)
        np.testing.assert_array_equal(owner.rope0.cos, original_buffer)
        return
    calls: list[object] = []

    def operation(receiver: Any, x: Any) -> Any:
        calls.append(receiver)
        receiver.rope0.cos[...] = np.float32(0)
        return x

    result = prepare_nnx_function_state(owner, operation, 19)
    assert result is not None
    adapter, state = result
    port = next(
        field.port for field in adapter.carriers[0].fields if field.name == "cos"
    )
    assert port is not None
    buffer = np.asarray(state[port]).copy()
    original_buffer = buffer.copy()
    changed = list(state)
    changed[port] = buffer
    with pytest.raises(SourcePrecisionError, match="mutable foreign-linked buffers"):
        if ingress == "reconstruct":
            adapter.reconstruct(tuple(changed))
        else:
            adapter.invoke((object(),), {}, tuple(changed))
    assert calls == []
    np.testing.assert_array_equal(buffer, original_buffer)
    assert owner.rope0.cos is owner.cos0
