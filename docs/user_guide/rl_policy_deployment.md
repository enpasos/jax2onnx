# RL Policy Deployment

Use this pattern when exporting a trained reinforcement-learning policy for
inference. The deployment callable should be a small deterministic actor with a
plain public interface:

```text
obs -> action
```

Keep training state, optimizer state, PRNG keys, rollout loops, and environment
state outside the exported callable. Freeze or close over policy parameters so
they become constants or initializers in the ONNX model, not runtime inputs.

The reference examples are listed in [Examples](examples.md):

- `ContinuousTanhPolicy`: `obs -> tanh(mean_action)`
- `DiscreteArgmaxPolicy`: `obs -> argmax(logits)`

Both are registered as `examples.rl` exports and run through the standard
generated example-test path, including ONNX checker, shape inference, runtime
parity, dynamic-batch execution, and action-contract assertions.

## Export Contract

A deployment-ready RL policy should make these choices explicit:

- Input name: `obs`
- Output name: `action`
- Dynamic batch: use a symbolic leading dimension such as `"B"`
- No runtime inputs named `key`, `rng`, `params`, `train_state`,
  `deterministic`, `training`, or `mutable`
- No stochastic sampling in the exported graph
- No `Dropout` or ONNX random operators in the exported graph

For continuous-control policies, bound the final action if the runtime expects
an actuator range:

```python
import jax.numpy as jnp


def continuous_policy(obs):
    h = jnp.tanh(obs @ w1 + b1)
    mean_action = h @ w2 + b2
    return jnp.tanh(mean_action)
```

For discrete-control policies, choose the action-index dtype intentionally and
validate it against the target runtime:

```python
import jax.numpy as jnp


def discrete_policy(obs):
    h = jnp.tanh(obs @ w1 + b1)
    logits = h @ w2 + b2
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)
```

## Export

Use `input_names` and `output_names` so the ONNX interface stays stable for the
serving code:

```python
from pathlib import Path

from jax2onnx import to_onnx

OBS_DIM = 17
model_path = Path("policy.onnx")

to_onnx(
    continuous_policy,
    inputs=[("B", OBS_DIM)],
    input_names=["obs"],
    output_names=["action"],
    return_mode="file",
    output_path=str(model_path),
)
```

The callable may be a function, a bound module, or a small wrapper around a
trained actor. The important part is that the exported callable has already been
reduced to inference behavior.

## Validate

Start with the general deployment checks from
[Validation & Deployment Readiness](validation.md), then add RL-specific
assertions for the public interface and action contract.

```python
import numpy as np
import onnx
import onnxruntime as ort

from jax2onnx import allclose

onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
onnx.shape_inference.infer_shapes(onnx_model)

session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"],
)

assert [input_.name for input_ in session.get_inputs()] == ["obs"]
assert [output.name for output in session.get_outputs()] == ["action"]

obs = np.linspace(-2.0, 2.0, 4 * OBS_DIM, dtype=np.float32).reshape(4, OBS_DIM)
passed, message = allclose(
    continuous_policy,
    str(model_path),
    inputs=[obs],
    rtol=1e-5,
    atol=1e-5,
)
assert passed, message
```

Run at least a few concrete batch sizes through ONNX Runtime when the export
uses a symbolic batch dimension:

```python
for batch_size in (1, 2, 7):
    obs = np.linspace(
        -1.5,
        1.5,
        batch_size * OBS_DIM,
        dtype=np.float32,
    ).reshape(batch_size, OBS_DIM)

    action = session.run(["action"], {"obs": obs})[0]
    assert action.shape[0] == batch_size
```

For a continuous actor with final `tanh`, also validate the action range:

```python
action = session.run(["action"], {"obs": obs})[0]
assert np.max(action) <= 1.0 + 1e-6
assert np.min(action) >= -1.0 - 1e-6
```

For a discrete actor, validate that action indices stay in range:

```python
action = session.run(["action"], {"obs": obs})[0]
assert action.ndim == 1
assert np.all(action >= 0)
assert np.all(action < NUM_ACTIONS)
```

## Scope

This pattern is policy-only. It is not intended to export:

- `env.step` or simulator dynamics
- Brax, Gymnax, MJX, or other environment loops
- PPO/SAC `train_step` functions
- optimizer state or replay buffers
- stochastic sampling branches that require a runtime PRNG key
- recurrent policy state; model that as an explicit interface such as
  `obs, h -> action, h_next` in a separate design

If a training actor needs flags such as `deterministic=True` or
`use_running_average=True`, bind those choices before export so the ONNX model
represents the exact inference contract.
