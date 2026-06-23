# jax2onnx/plugins/examples/rl/policies.py

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


CONTINUOUS_OBS_DIM: int = 17
CONTINUOUS_HIDDEN_DIM: int = 32
CONTINUOUS_ACTION_DIM: int = 6

DISCRETE_OBS_DIM: int = 8
DISCRETE_HIDDEN_DIM: int = 16
DISCRETE_NUM_ACTIONS: int = 4

_CONTINUOUS_W1: npt.NDArray[np.float32] = np.linspace(
    -0.2,
    0.2,
    CONTINUOUS_OBS_DIM * CONTINUOUS_HIDDEN_DIM,
    dtype=np.float32,
).reshape(CONTINUOUS_OBS_DIM, CONTINUOUS_HIDDEN_DIM)
_CONTINUOUS_B1: npt.NDArray[np.float32] = np.linspace(
    -0.05,
    0.05,
    CONTINUOUS_HIDDEN_DIM,
    dtype=np.float32,
)
_CONTINUOUS_W2: npt.NDArray[np.float32] = np.linspace(
    -0.15,
    0.15,
    CONTINUOUS_HIDDEN_DIM * CONTINUOUS_ACTION_DIM,
    dtype=np.float32,
).reshape(CONTINUOUS_HIDDEN_DIM, CONTINUOUS_ACTION_DIM)
_CONTINUOUS_B2: npt.NDArray[np.float32] = np.linspace(
    -0.03,
    0.03,
    CONTINUOUS_ACTION_DIM,
    dtype=np.float32,
)

_DISCRETE_W1: npt.NDArray[np.float32] = np.linspace(
    -0.25,
    0.25,
    DISCRETE_OBS_DIM * DISCRETE_HIDDEN_DIM,
    dtype=np.float32,
).reshape(DISCRETE_OBS_DIM, DISCRETE_HIDDEN_DIM)
_DISCRETE_B1: npt.NDArray[np.float32] = np.linspace(
    -0.02,
    0.02,
    DISCRETE_HIDDEN_DIM,
    dtype=np.float32,
)
_DISCRETE_W2: npt.NDArray[np.float32] = np.linspace(
    -0.1,
    0.1,
    DISCRETE_HIDDEN_DIM * DISCRETE_NUM_ACTIONS,
    dtype=np.float32,
).reshape(DISCRETE_HIDDEN_DIM, DISCRETE_NUM_ACTIONS)
_DISCRETE_B2: npt.NDArray[np.float32] = np.linspace(
    -0.01,
    0.01,
    DISCRETE_NUM_ACTIONS,
    dtype=np.float32,
)

_FORBIDDEN_POLICY_OPS: tuple[str, ...] = (
    "Dropout",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
)
_FORBIDDEN_INTERFACE_NAMES: frozenset[str] = frozenset(
    {
        "deterministic",
        "key",
        "mutable",
        "params",
        "rng",
        "train_state",
        "training",
    }
)


def representative_continuous_obs(batch_size: int = 4) -> npt.NDArray[np.float32]:
    values = np.linspace(
        -2.0,
        2.0,
        batch_size * CONTINUOUS_OBS_DIM,
        dtype=np.float32,
    )
    return cast(
        npt.NDArray[np.float32],
        values.reshape(batch_size, CONTINUOUS_OBS_DIM),
    )


def representative_discrete_obs(batch_size: int = 5) -> npt.NDArray[np.float32]:
    values = np.linspace(
        -1.0,
        1.0,
        batch_size * DISCRETE_OBS_DIM,
        dtype=np.float32,
    )
    return cast(
        npt.NDArray[np.float32],
        values.reshape(batch_size, DISCRETE_OBS_DIM),
    )


def continuous_tanh_policy(obs: jax.Array) -> jax.Array:
    h = jnp.tanh(obs @ _CONTINUOUS_W1 + _CONTINUOUS_B1)
    mean_action = h @ _CONTINUOUS_W2 + _CONTINUOUS_B2
    return jnp.tanh(mean_action)


def discrete_argmax_policy(obs: jax.Array) -> jax.Array:
    h = jnp.tanh(obs @ _DISCRETE_W1 + _DISCRETE_B1)
    logits = h @ _DISCRETE_W2 + _DISCRETE_B2
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def _value_names(values: Iterable[Any]) -> list[str]:
    return [str(getattr(value, "name", "")) for value in values]


def _public_io_contract(model: Any) -> bool:
    graph = getattr(model, "graph", None)
    if graph is None:
        return False

    input_names = _value_names(getattr(graph, "input", ()))
    output_names = _value_names(getattr(graph, "output", ()))
    if input_names != ["obs"] or output_names != ["action"]:
        return False

    lowered_inputs = [name.lower() for name in input_names]
    return not any(
        forbidden in input_name
        for input_name in lowered_inputs
        for forbidden in _FORBIDDEN_INTERFACE_NAMES
    )


def _policy_contract(
    path_specs: Sequence[str],
) -> Callable[[Any], bool]:
    graph_check = EG(
        path_specs,
        symbols={"B": None},
        must_absent=_FORBIDDEN_POLICY_OPS,
        no_unused_inputs=True,
    )

    def _check(model: Any) -> bool:
        return _public_io_contract(model) and graph_check(model)

    return _check


def _runtime_obs_inputs(
    obs_factory: Callable[[int], npt.NDArray[np.float32]],
) -> list[list[npt.NDArray[np.float32]]]:
    return [[obs_factory(batch_size)] for batch_size in (1, 2, 7)]


def _runtime_batch_size(inputs: Sequence[Any]) -> int:
    if len(inputs) != 1:
        raise AssertionError(f"Expected one runtime input, got {len(inputs)}.")
    obs = np.asarray(inputs[0])
    if obs.ndim != 2:
        raise AssertionError(f"Expected rank-2 obs input, got shape {obs.shape}.")
    return int(obs.shape[0])


def _single_runtime_output(outputs: Sequence[Any]) -> npt.NDArray[Any]:
    if len(outputs) != 1:
        raise AssertionError(f"Expected one runtime output, got {len(outputs)}.")
    return np.asarray(outputs[0])


def _continuous_runtime_contract(
    *,
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    **_: Any,
) -> bool:
    batch_size = _runtime_batch_size(inputs)
    action = _single_runtime_output(outputs)
    expected_shape = (batch_size, CONTINUOUS_ACTION_DIM)
    if action.shape != expected_shape:
        raise AssertionError(
            f"Continuous policy action shape mismatch: expected {expected_shape}, "
            f"got {action.shape}."
        )
    if action.dtype != np.float32:
        raise AssertionError(
            f"Continuous policy action dtype mismatch: expected float32, "
            f"got {action.dtype}."
        )
    if not np.all(np.isfinite(action)):
        raise AssertionError("Continuous policy produced non-finite actions.")
    if np.min(action) < -1.0 - 1e-6 or np.max(action) > 1.0 + 1e-6:
        raise AssertionError("Continuous policy actions escaped tanh range [-1, 1].")
    return True


def _discrete_runtime_contract(
    *,
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    **_: Any,
) -> bool:
    batch_size = _runtime_batch_size(inputs)
    action = _single_runtime_output(outputs)
    expected_shape = (batch_size,)
    if action.shape != expected_shape:
        raise AssertionError(
            f"Discrete policy action shape mismatch: expected {expected_shape}, "
            f"got {action.shape}."
        )
    if action.dtype != np.int32:
        raise AssertionError(
            f"Discrete policy action dtype mismatch: expected int32, got {action.dtype}."
        )
    if np.any(action < 0) or np.any(action >= DISCRETE_NUM_ACTIONS):
        raise AssertionError(
            "Discrete policy produced action indices outside "
            f"[0, {DISCRETE_NUM_ACTIONS})."
        )
    return True


register_example(
    component="ContinuousTanhPolicy",
    description="Deterministic continuous-control RL policy: obs -> tanh(mean_action).",
    since="0.14.2",
    context="examples.rl",
    children=["jnp.matmul", "jnp.add", "jnp.tanh"],
    testcases=[
        {
            "testcase": "continuous_tanh_policy",
            "callable": continuous_tanh_policy,
            "input_shapes": [("B", CONTINUOUS_OBS_DIM)],
            "input_dtypes": [jnp.float32],
            "input_values": [representative_continuous_obs()],
            "input_names": ["obs"],
            "output_names": ["action"],
            "expected_output_shapes": [("B", CONTINUOUS_ACTION_DIM)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_dynamic": True,
            "run_only_f32_variant": True,
            "check_onnx_load": True,
            "check_onnx_shape_inference": True,
            "post_check_onnx_graph": _policy_contract(
                [
                    "Gemm:Bx32 -> Tanh:Bx32 -> Gemm:Bx6 -> Tanh:Bx6",
                ],
            ),
            "runtime_input_values": _runtime_obs_inputs(representative_continuous_obs),
            "post_check_onnx_runtime": _continuous_runtime_contract,
        },
    ],
)

register_example(
    component="DiscreteArgmaxPolicy",
    description="Deterministic discrete-control RL policy: obs -> argmax(logits).",
    since="0.14.2",
    context="examples.rl",
    children=["jnp.matmul", "jnp.add", "jnp.tanh", "jnp.argmax"],
    testcases=[
        {
            "testcase": "discrete_argmax_policy",
            "callable": discrete_argmax_policy,
            "input_shapes": [("B", DISCRETE_OBS_DIM)],
            "input_dtypes": [jnp.float32],
            "input_values": [representative_discrete_obs()],
            "input_names": ["obs"],
            "output_names": ["action"],
            "expected_output_shapes": [("B",)],
            "expected_output_dtypes": [jnp.int32],
            "run_only_dynamic": True,
            "run_only_f32_variant": True,
            "check_onnx_load": True,
            "check_onnx_shape_inference": True,
            "post_check_onnx_graph": _policy_contract(
                [
                    "Gemm:Bx16 -> Tanh:Bx16 -> Gemm:Bx4 -> ArgMax:B",
                ],
            ),
            "runtime_input_values": _runtime_obs_inputs(representative_discrete_obs),
            "post_check_onnx_runtime": _discrete_runtime_contract,
        },
    ],
)
