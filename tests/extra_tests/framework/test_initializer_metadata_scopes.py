# tests/extra_tests/framework/test_initializer_metadata_scopes.py

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

from flax import linen, nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.plugins.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
)
from tests.extra_tests.framework.test_initializer_metadata_contract import (
    _assert_storage,
    _properties,
    _run,
    _tensors,
)


@pytest.fixture(autouse=True)
def _clear_local_functions() -> Iterator[None]:
    yield
    for qualified in tuple(ONNX_FUNCTION_PLUGIN_REGISTRY):
        if qualified.startswith(f"{__name__}."):
            PLUGIN_REGISTRY.pop(f"onnx_fn::{qualified}", None)
            ONNX_FUNCTION_PLUGIN_REGISTRY.pop(qualified, None)


def _assert_complete_inventory(
    model: onnx.ModelProto,
    expected: dict[str, jax.Array],
    *,
    allow_repeated: bool = False,
) -> None:
    carriers = [tensor for tensor in _tensors(model) if _properties(tensor)]
    inventory = Counter(_properties(tensor)["source"] for tensor in carriers)
    assert set(inventory) == set(expected)
    if not allow_repeated:
        assert inventory == Counter(expected.keys())
    for tensor in carriers:
        name = _properties(tensor)["source"]
        assert _properties(tensor) == {"source": name, "opaque": ""}
        _assert_storage(tensor, np.asarray(expected[name]))


def test_root_jit_and_two_nested_functions_have_complete_capture_inventory() -> None:
    sources = {
        name: jnp.asarray([index + 1.0, index + 2.0], dtype=jnp.float32)
        for index, name in enumerate(("root", "jit", "outer", "inner"))
    }
    seen: set[str] = set()

    def annotate(source: object) -> dict[str, str] | None:
        for name, value in sources.items():
            if source is value:
                seen.add(name)
                return {"source": name, "opaque": ""}
        pytest.fail("inputs, literals, and generated helpers are not captures")

    @onnx_function
    class Inner:
        def __call__(self, x: jax.Array) -> jax.Array:
            return x + sources["inner"]

    inner = Inner()

    @onnx_function
    class Outer:
        def __call__(self, x: jax.Array) -> jax.Array:
            return inner(x) * sources["outer"]

    outer = Outer()
    jitted = jax.jit(lambda x: x + sources["jit"])

    def apply(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return outer(x) + sources["root"], jitted(x) + 1.0

    model = to_onnx(apply, [(2,)], initializer_metadata=annotate)
    assert seen == set(sources)
    assert len(model.functions) == 2
    _assert_complete_inventory(model, sources)
    onnx.checker.check_model(model, full_check=True)
    x = np.asarray([0.5, 1.5], dtype=np.float32)
    for actual, expected in zip(_run(model, x), apply(jnp.asarray(x))):
        np.testing.assert_allclose(actual, expected)


def test_jit_multioutput_captures_keep_disjoint_sources_and_direct_constant_output() -> (
    None
):
    sources = {
        "add": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        "multiply": jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        "direct": jnp.asarray([5.0, 6.0], dtype=jnp.float32),
    }
    seen: set[str] = set()

    def annotate(source: object) -> dict[str, str] | None:
        for name, value in sources.items():
            if source is value:
                seen.add(name)
                return {"source": name, "opaque": ""}
        return None

    target = jax.jit(
        lambda x: (x + sources["add"], x * sources["multiply"], sources["direct"])
    )
    model = to_onnx(target, [(2,)], initializer_metadata=annotate)
    assert seen == set(sources)
    _assert_complete_inventory(model, sources)
    x = np.asarray([0.25, 0.75], dtype=np.float32)
    actual = _run(model, x)
    assert len(actual) == 3
    onnx.checker.check_model(model, full_check=True)
    for result, expected in zip(actual, target(jnp.asarray(x))):
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "kind",
    [
        "cond",
        "constant_cond",
        "scan",
        "zero_scan",
        "while",
        "zero_while",
        "fori",
        "zero_fori",
    ],
)
def test_control_flow_capture_inventory_includes_all_structurally_reachable_bodies(
    kind: str,
) -> None:
    root = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    body = jnp.asarray([3.0, 4.0], dtype=jnp.float32)
    other = jnp.asarray([5.0, 6.0], dtype=jnp.float32)
    sources = {"root": root, "body": body}
    if "cond" in kind:
        sources["other"] = other
    seen: set[str] = set()

    def annotate(source: object) -> dict[str, str] | None:
        for name, value in sources.items():
            if source is value:
                seen.add(name)
                return {"source": name, "opaque": ""}
        return None

    def apply(x: jax.Array) -> jax.Array:
        count = 0 if kind.startswith("zero_") else 2
        if "cond" in kind:
            predicate = True if kind == "constant_cond" else x[0] > 0
            result = jax.lax.cond(predicate, lambda y: y + body, lambda y: y - other, x)
        elif "scan" in kind:
            result = jax.lax.scan(
                lambda carry, _: (carry + body, None), x, None, length=count
            )[0]
        elif "while" in kind:
            result = jax.lax.while_loop(
                lambda state: state[1] < count,
                lambda state: (state[0] + body, state[1] + 1),
                (x, 0),
            )[0]
        else:
            result = jax.lax.fori_loop(0, count, lambda _, carry: carry + body, x)
        return result + root

    model = to_onnx(apply, [(2,)], initializer_metadata=annotate)
    assert seen == set(sources)
    _assert_complete_inventory(model, sources)
    onnx.checker.check_model(model, full_check=True)
    for x in (np.asarray([0.5, 1.5], np.float32), np.asarray([-0.5, 1.5], np.float32)):
        np.testing.assert_allclose(_run(model, x)[0], np.asarray(apply(jnp.asarray(x))))


def test_linen_dense_preserves_both_parameter_sources_and_runtime() -> None:
    module = linen.Dense(3, dtype=jnp.float32, param_dtype=jnp.float32)
    variables = module.init(jax.random.key(19), jnp.ones((1, 2), dtype=jnp.float32))
    sources = {f"dense.{name}": value for name, value in variables["params"].items()}
    seen: set[str] = set()

    def annotate(source: object) -> dict[str, str] | None:
        for name, value in sources.items():
            if source is value:
                seen.add(name)
                return {"source": name, "opaque": ""}
        return None

    model = to_onnx(
        lambda x: module.apply(variables, x), [(1, 2)], initializer_metadata=annotate
    )
    assert seen == {"dense.kernel", "dense.bias"}
    _assert_complete_inventory(model, sources)
    x = np.asarray([[0.25, 0.75]], dtype=np.float32)
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_allclose(
        _run(model, x)[0], module.apply(variables, jnp.asarray(x)), rtol=1e-6, atol=1e-6
    )


def test_reused_nnx_linear_general_preserves_complete_inventory_and_shared_uses() -> (
    None
):
    module = nnx.LinearGeneral(
        3, 3, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nnx.Rngs(23)
    )
    sources = {
        "shared.kernel": module.kernel.get_value(),
        "shared.bias": module.bias.get_value(),
    }

    def annotate(source: object) -> dict[str, str] | None:
        for name, value in sources.items():
            if source is value:
                return {"source": name, "opaque": ""}
        return None

    model = to_onnx(
        lambda x: module(module(x)), [(1, 3)], initializer_metadata=annotate
    )
    _assert_complete_inventory(model, sources, allow_repeated=True)
    kernels = {
        tensor.name
        for tensor in _tensors(model)
        if _properties(tensor).get("source") == "shared.kernel"
    }
    gemms = [node for node in model.graph.node if node.op_type == "Gemm"]
    assert len(gemms) == 2
    assert all(node.input[1] in kernels for node in gemms)
    x = np.asarray([[0.25, 0.5, 0.75]], dtype=np.float32)
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_allclose(
        _run(model, x)[0], module(module(jnp.asarray(x))), rtol=1e-6, atol=1e-6
    )
