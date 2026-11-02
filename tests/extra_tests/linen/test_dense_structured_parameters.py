# tests/extra_tests/linen/test_dense_structured_parameters.py

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.linen.dtypes import promote_dtype

from jax2onnx.plugins.flax.linen.dense import DensePlugin


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _StructuredParameter:
    payload: np.ndarray
    scale: np.ndarray

    def tree_flatten(self) -> tuple[tuple[np.ndarray, np.ndarray], None]:
        return (self.payload, self.scale), None

    @classmethod
    def tree_unflatten(
        cls, auxiliary: None, children: tuple[np.ndarray, np.ndarray]
    ) -> _StructuredParameter:
        del auxiliary
        return cls(*children)


@pytest.fixture
def patch(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Callable[..., Any], Mock, Mock]:
    original = Mock(return_value=object())
    bind = Mock(return_value=object())
    monkeypatch.setattr(DensePlugin, "_PRIM", SimpleNamespace(bind=bind))
    monkeypatch.setattr(DensePlugin, "_ORIGINAL_DENSE_CALL", None)
    return DensePlugin.get_monkey_patch(original), original, bind


def _module(params: dict[str, Any], *, use_bias: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        scope=SimpleNamespace(variables=lambda: {"params": params}),
        use_bias=use_bias,
        features=3,
        dtype=None,
        promote_dtype=promote_dtype,
    )


def _assert_original(
    patched: Callable[..., Any],
    original: Mock,
    bind: Mock,
    module: Any,
    inputs: np.ndarray,
) -> None:
    result = patched(module, inputs)
    assert result is original.return_value
    original.assert_called_once()
    assert original.call_args.args[0] is module
    assert original.call_args.args[1] is inputs
    assert original.call_args.kwargs == {}
    bind.assert_not_called()


@pytest.mark.parametrize(
    "container", ["mapping", "single_leaf_mapping", "list", "tuple", "registered"]
)
def test_structured_kernel_uses_original_call_unchanged(
    patch: tuple[Callable[..., Any], Mock, Mock], container: str
) -> None:
    kernel = np.ones((4, 3), dtype=np.float32)
    scale = np.asarray(0.25, dtype=np.float32)
    structured = {
        "mapping": {"payload": kernel, "scale": scale},
        "single_leaf_mapping": {"payload": kernel},
        "list": [kernel, scale],
        "tuple": (kernel, scale),
        "registered": _StructuredParameter(kernel, scale),
    }[container]
    module = _module({"kernel": structured, "bias": np.zeros(3, dtype=np.float32)})
    inputs = np.ones((2, 4), dtype=np.float32)
    _assert_original(*patch, module, inputs)
    assert module.scope.variables()["params"]["kernel"] is structured


def test_used_structured_bias_uses_original_call_unchanged(
    patch: tuple[Callable[..., Any], Mock, Mock],
) -> None:
    bias = _StructuredParameter(
        np.ones(3, dtype=np.float32), np.asarray(0.25, dtype=np.float32)
    )
    module = _module({"kernel": np.ones((4, 3), dtype=np.float32), "bias": bias})
    _assert_original(*patch, module, np.ones((2, 4), dtype=np.float32))
    assert module.scope.variables()["params"]["bias"] is bias


@pytest.mark.parametrize("bias_kind", ["array", "missing", "unused_structured"])
def test_array_kernel_retains_shortcut(
    patch: tuple[Callable[..., Any], Mock, Mock], bias_kind: str
) -> None:
    patched, original, bind = patch
    kernel = jnp.ones((4, 3), dtype=jnp.float32)
    params: dict[str, Any] = {"kernel": kernel}
    if bias_kind == "array":
        params["bias"] = jnp.arange(3, dtype=jnp.float32)
    elif bias_kind == "unused_structured":
        params["bias"] = {"payload": np.ones(3, dtype=np.float32)}
    module = _module(params, use_bias=bias_kind != "unused_structured")
    inputs = jnp.ones((2, 4), dtype=jnp.float32)

    assert patched(module, inputs) is bind.return_value

    original.assert_not_called()
    bind.assert_called_once()
    assert bind.call_args.args[0] is inputs
    assert bind.call_args.args[1] is kernel
    if bias_kind == "array":
        assert bind.call_args.args[2] is params["bias"]
    else:
        np.testing.assert_array_equal(bind.call_args.args[2], np.zeros(3, np.float32))
    assert bind.call_args.kwargs == {
        "use_bias": module.use_bias,
        "dimension_numbers": (((1,), (0,)), ((), ())),
    }


@pytest.mark.parametrize(
    "scope_kind", ["absent", "without_variables", "empty", "no_kernel"]
)
def test_uninitialized_or_missing_kernel_retains_original_fallback(
    patch: tuple[Callable[..., Any], Mock, Mock], scope_kind: str
) -> None:
    scope = {
        "absent": None,
        "without_variables": SimpleNamespace(),
        "empty": SimpleNamespace(variables=lambda: {}),
        "no_kernel": SimpleNamespace(
            variables=lambda: {"params": {"bias": np.ones(3, dtype=np.float32)}}
        ),
    }[scope_kind]
    module = SimpleNamespace(scope=scope, use_bias=True, features=3)
    _assert_original(*patch, module, np.ones((2, 4), dtype=np.float32))
