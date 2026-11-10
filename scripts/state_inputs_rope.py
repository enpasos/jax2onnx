# scripts/state_inputs_rope.py

"""Small Equinox attention/RoPE model exported with its state as named inputs.

Inspired by Clément's RoPE use case; this is not an Equimo/DINO adapter. Run::

    python -m scripts.state_inputs_rope --output-dir /tmp/state-inputs-rope

``to_onnx(..., state=...)`` exports four arrays of the model as named inputs
with saved defaults and labels. The script then saves and reloads the file,
runs it with the defaults and with replacement arrays, rebuilds the Python
model from the file and takes one AdamW step.

Modules stay undecorated. The free ONNX Function receives its weight as an
argument. Both frozen projections hold one array object and therefore become
one input; the two trainable weights start with equal values but are separate
objects and stay separate inputs.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import optax

from jax2onnx import StateInput, onnx_function, to_onnx


STATE_NAMES = (
    "shared_weight",
    "independent_weight",
    "rope_rotations",
    "shared_frozen",
)
PATH = "pkg.example.path"
TRAINABLE = "pkg.example.trainable"


class StoredState(NamedTuple):
    """One state array as read from a model file, with its labels."""

    name: str
    value: np.ndarray | jax.Array
    metadata: dict[str, str]


@onnx_function
def project(x: jax.Array, weight: jax.Array) -> jax.Array:
    return x @ weight


class RoPE(eqx.Module):
    rotations: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        pairs = x.reshape(x.shape[0], x.shape[1] // 2, 2)
        rotations = jax.lax.stop_gradient(self.rotations)
        cosine, sine = rotations[..., 0], rotations[..., 1]
        left, right = pairs[..., 0], pairs[..., 1]
        return jnp.stack(
            (left * cosine - right * sine, left * sine + right * cosine), axis=-1
        ).reshape(x.shape)


class FrozenProjection(eqx.Module):
    matrix: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ jax.lax.stop_gradient(self.matrix)


class SmallAttention(eqx.Module):
    shared_weight: jax.Array
    independent_weight: jax.Array
    rope: RoPE
    q_projection: FrozenProjection
    k_projection: FrozenProjection

    def __init__(self, *, key: jax.Array, tokens: int = 3, width: int = 4) -> None:
        if width % 2:
            raise ValueError("RoPE requires an even width")
        weight_key, frozen_key = jax.random.split(key)
        self.shared_weight = (
            jax.random.normal(weight_key, (width, width), dtype=jnp.float32) * 0.2
        )
        # Equal initial values represent two independent trainable parameters.
        self.independent_weight = jnp.array(self.shared_weight)
        angles = jnp.arange(tokens, dtype=jnp.float32)[:, None] * (
            10000.0 ** (-jnp.arange(width // 2, dtype=jnp.float32) / (width // 2))
        )
        self.rope = RoPE(jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1))
        frozen = jax.random.normal(frozen_key, (width, width), dtype=jnp.float32) * 0.15
        self.q_projection = FrozenProjection(frozen)
        self.k_projection = FrozenProjection(frozen)

    def __call__(self, x: jax.Array) -> jax.Array:
        q = self.rope(project(x, self.shared_weight) + 0.1 * self.q_projection(x))
        k = self.rope(
            project(x * 0.75, self.shared_weight) + 0.15 * self.k_projection(x)
        )
        attention = jax.nn.softmax((q @ k.T) / x.shape[-1] ** 0.5, axis=-1)
        values = x @ self.independent_weight
        # The same logical weight also has direct and reshaped root uses.
        direct = x @ self.shared_weight
        view = jnp.sum(
            x[:, None, :] * self.shared_weight.reshape(1, *self.shared_weight.shape),
            axis=-1,
        )
        return attention @ values + 0.1 * direct + 0.05 * view


def state_inputs(model: SmallAttention) -> list[StateInput]:
    """Declare this model's state: one entry per array object, with its labels."""
    if model.q_projection.matrix is not model.k_projection.matrix:
        raise ValueError("both frozen projections must hold one array object")
    declared = (
        (model.shared_weight, "shared_weight", "true"),
        (model.independent_weight, "independent_weight", "true"),
        (model.rope.rotations, "rope.rotations", "false"),
        (model.q_projection.matrix, "q_projection.matrix;k_projection.matrix", "false"),
    )
    return [
        StateInput(array, name, {PATH: path, TRAINABLE: trainable})
        for name, (array, path, trainable) in zip(STATE_NAMES, declared)
    ]


def export_model(
    model: SmallAttention, x: jax.Array, *, opset: int = 23
) -> onnx.ModelProto:
    """Export the model itself; its declared arrays become named inputs."""
    return to_onnx(
        model,
        inputs=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
        input_names=["x"],
        state=state_inputs(model),
        opset=opset,
    )


def read_state(proto: onnx.ModelProto) -> tuple[StoredState, ...]:
    """Read this example's four state inputs and their labels from a model."""
    inputs = {value.name for value in proto.graph.input}
    tensors = {tensor.name: tensor for tensor in proto.graph.initializer}
    if inputs & tensors.keys() != set(STATE_NAMES):
        raise ValueError("model does not have this example's four state inputs")
    return tuple(
        StoredState(
            name,
            np.array(onnx.numpy_helper.to_array(tensors[name]), copy=True),
            {entry.key: entry.value for entry in tensors[name].metadata_props},
        )
        for name in STATE_NAMES
    )


def reconstruct(model: SmallAttention, state: Sequence[StoredState]) -> SmallAttention:
    """Place arrays read from a file into this known Python architecture.

    The path labels select the fields. The semicolon path is this example's
    own convention for its two frozen projections: both receive one array
    object, so they stay tied when the rebuilt model is exported again.
    """
    selectors: dict[str, Callable[[SmallAttention], tuple[jax.Array, ...]]] = {
        "shared_weight": lambda m: (m.shared_weight,),
        "independent_weight": lambda m: (m.independent_weight,),
        "rope.rotations": lambda m: (m.rope.rotations,),
        "q_projection.matrix;k_projection.matrix": lambda m: (
            m.q_projection.matrix,
            m.k_projection.matrix,
        ),
    }
    paths = [entry.metadata.get(PATH, "") for entry in state]
    if sorted(paths) != sorted(selectors):
        raise ValueError("state must provide each known path exactly once")
    for entry, path in zip(state, paths):
        selector = selectors[path]
        array = jnp.asarray(entry.value)
        model = eqx.tree_at(
            selector, model, replace=tuple(array for _ in selector(model))
        )
    return model


def save_reload(
    proto: onnx.ModelProto, path: Path, *, external_data: bool
) -> onnx.ModelProto:
    """Save the model, reload it and check that state and labels are unchanged."""
    expected = read_state(proto)
    # Saving with external data rewrites the proto it is given.
    to_save = onnx.ModelProto()
    to_save.CopyFrom(proto)
    onnx.save_model(
        to_save,
        path,
        save_as_external_data=external_data,
        all_tensors_to_one_file=True,
        location=path.name + ".data",
        size_threshold=0,
    )
    restored = onnx.load(path)
    onnx.checker.check_model(restored, full_check=True)
    if [value.name for value in restored.graph.input] != ["x", *STATE_NAMES]:
        raise ValueError("serialized graph inputs changed")
    for before, after in zip(expected, read_state(restored)):
        source, loaded = np.asarray(before.value), np.asarray(after.value)
        if (
            after.metadata != before.metadata
            or source.dtype != loaded.dtype
            or source.shape != loaded.shape
            or source.tobytes() != loaded.tobytes()
        ):
            raise ValueError(f"serialized state changed for {before.name}")
    return restored


def runtime(proto: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def adamw_step(
    model: SmallAttention,
    state: Sequence[StoredState],
    x: jax.Array,
    target: jax.Array,
) -> tuple[tuple[StoredState, ...], float]:
    """Read the example's own labels; frozen state never enters AdamW at all."""
    if any(entry.metadata.get(TRAINABLE) not in {"true", "false"} for entry in state):
        raise ValueError("this example requires explicit true/false trainability")
    trainable = {
        entry.name: jnp.asarray(entry.value)
        for entry in state
        if entry.metadata[TRAINABLE] == "true"
    }
    frozen = {
        entry.name: jnp.asarray(entry.value)
        for entry in state
        if entry.metadata[TRAINABLE] == "false"
    }

    def loss(parameters: dict[str, jax.Array]) -> jax.Array:
        arrays = {**frozen, **parameters}
        rebuilt = reconstruct(
            model,
            tuple(
                StoredState(entry.name, arrays[entry.name], entry.metadata)
                for entry in state
            ),
        )
        return jnp.mean((rebuilt(x) - target) ** 2)

    loss_value, gradients = jax.value_and_grad(loss)(trainable)
    optimizer = optax.adamw(learning_rate=0.01, weight_decay=0.1)
    updates, _ = optimizer.update(gradients, optimizer.init(trainable), trainable)
    updated = optax.apply_updates(trainable, updates)
    return (
        tuple(
            StoredState(
                entry.name, updated.get(entry.name, entry.value), entry.metadata
            )
            for entry in state
        ),
        float(loss_value),
    )


def run_demo(
    output_dir: Path, *, external_data: bool = False, opset: int = 23
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SmallAttention(key=jax.random.key(0))
    x = jnp.linspace(-0.8, 0.9, 12, dtype=jnp.float32).reshape(3, 4)
    path = output_dir / "attention.onnx"
    proto = save_reload(
        export_model(model, x, opset=opset), path, external_data=external_data
    )
    imported = read_state(proto)
    expected_policy = {
        "shared_weight": ("shared_weight", "true"),
        "independent_weight": ("independent_weight", "true"),
        "rope_rotations": ("rope.rotations", "false"),
        "shared_frozen": ("q_projection.matrix;k_projection.matrix", "false"),
    }
    actual_policy = {
        entry.name: (entry.metadata.get(PATH), entry.metadata.get(TRAINABLE))
        for entry in imported
    }
    if actual_policy != expected_policy:
        raise AssertionError("stored state policy differs from this architecture")
    session = runtime(proto)
    actual = session.run(None, {"x": np.asarray(x)})[0]
    np.testing.assert_allclose(actual, model(x), rtol=2e-5, atol=2e-6)
    overrides = {
        entry.name: np.asarray(entry.value) * np.float32(0.7 + index * 0.1)
        + np.float32(0.2 * (index + 1))
        for index, entry in enumerate(imported)
    }
    overridden = session.run(None, {"x": np.asarray(x), **overrides})[0]
    reference = reconstruct(
        model,
        tuple(
            StoredState(entry.name, overrides[entry.name], entry.metadata)
            for entry in imported
        ),
    )(x)
    np.testing.assert_allclose(overridden, reference, rtol=2e-5, atol=2e-6)
    updated, loss = adamw_step(model, imported, x, jnp.zeros_like(x))
    for before, after in zip(imported, updated):
        changed = (
            np.asarray(before.value).tobytes() != np.asarray(after.value).tobytes()
        )
        if changed != (before.name in {"shared_weight", "independent_weight"}):
            raise AssertionError(f"unexpected optimizer update: {before.name}")
    report = {
        "versions": {
            name: version(name)
            for name in ("jax", "equinox", "optax", "onnx", "onnx-ir", "onnxruntime")
        },
        "opset": {entry.domain: entry.version for entry in proto.opset_import}[""],
        "x64_enabled": jax.config.jax_enable_x64,
        "ort_optimization": "ORT_ENABLE_ALL",
        "state_inputs": len(imported),
        "model": str(path),
        "external_data": external_data,
        "max_absolute_error": float(np.max(np.abs(actual - np.asarray(model(x))))),
        "override_max_absolute_error": float(
            np.max(np.abs(overridden - np.asarray(reference)))
        ),
        "adamw_loss": loss,
        "frozen_state_unchanged": True,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--external-data", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_demo(args.output_dir, external_data=args.external_data), indent=2
        )
    )
