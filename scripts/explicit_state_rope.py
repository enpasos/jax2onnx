# scripts/explicit_state_rope.py

"""Small Equinox attention/RoPE example with four explicitly chosen state slots.

Inspired by Clément's RoPE use case; this is not an Equimo/DINO adapter. Run::

    python -m scripts.explicit_state_rope --output-dir /tmp/explicit-state-rope

Modules stay undecorated. The free ONNX Function receives its weight explicitly;
RoPE uses traced JAX operations rather than host evaluation of dynamic state.
The known architecture declares that both frozen projection branches share one
slot. Neither array identity nor equal values infer that policy.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import optax  # type: ignore[import-untyped]

from jax2onnx import onnx_function
from scripts.explicit_state_export import StateSlot, export_with_state


SLOT_NAMES = (
    "shared_weight",
    "independent_weight",
    "rope_rotations",
    "shared_frozen",
)
TRAINABLE = "pkg.example.trainable"


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


def state_slots(model: SmallAttention) -> tuple[StateSlot, ...]:
    """Declare this architecture's slots, checking its chosen frozen alias policy."""
    q, k = np.asarray(model.q_projection.matrix), np.asarray(model.k_projection.matrix)
    if q.dtype != k.dtype or q.shape != k.shape or q.tobytes() != k.tobytes():
        raise ValueError("declared shared frozen projection matrices must match")
    paths = (
        "shared_weight",
        "independent_weight",
        "rope.rotations",
        "q_projection.matrix;k_projection.matrix",
    )
    values = (
        model.shared_weight,
        model.independent_weight,
        model.rope.rotations,
        model.q_projection.matrix,
    )
    return tuple(
        StateSlot(name, value, {"pkg.example.path": path, TRAINABLE: trainable})
        for name, value, path, trainable in zip(
            SLOT_NAMES, values, paths, ("true", "true", "false", "false")
        )
    )


def explicit_forward(model: SmallAttention) -> Callable[..., jax.Array]:
    """Partition once and replace every selected leaf from the four arguments."""
    arrays, static = eqx.partition(model, eqx.is_array)

    def forward(
        x: jax.Array,
        shared_weight: jax.Array,
        independent_weight: jax.Array,
        rope_rotations: jax.Array,
        shared_frozen: jax.Array,
    ) -> jax.Array:
        replaced = eqx.tree_at(
            lambda m: (
                m.shared_weight,
                m.independent_weight,
                m.rope.rotations,
                m.q_projection.matrix,
                m.k_projection.matrix,
            ),
            arrays,
            replace=(
                shared_weight,
                independent_weight,
                rope_rotations,
                shared_frozen,
                shared_frozen,
            ),
        )
        return eqx.combine(replaced, static)(x)

    return forward


def export_model(model: SmallAttention, x: jax.Array) -> ir.Model:
    exported: ir.Model = export_with_state(
        explicit_forward(model),
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        state_slots(model),
        data_input_names=["x"],
    )
    return exported


def read_slots(proto: onnx.ModelProto) -> tuple[StateSlot, ...]:
    """Read only the known state from input/initializer intersections, not ORT IO."""
    inputs = {value.name for value in proto.graph.input}
    tensors = {tensor.name: tensor for tensor in proto.graph.initializer}
    if inputs & tensors.keys() != set(SLOT_NAMES):
        raise ValueError("model does not have this example's four state slots")
    return tuple(
        StateSlot(
            name,
            np.array(onnx.numpy_helper.to_array(tensors[name]), copy=True),
            {entry.key: entry.value for entry in tensors[name].metadata_props},
        )
        for name in SLOT_NAMES
    )


def reconstruct(model: SmallAttention, state: Sequence[StateSlot]) -> SmallAttention:
    """Apply imported path labels to this known Python architecture.

    These selectors are independent of the export closure. The semicolon path
    explicitly denotes this example's shared projection branches; it is not a
    general framework path or alias discovery mechanism.
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
    for slot in state:
        path = slot.metadata.get("pkg.example.path", "")
        if path not in selectors:
            raise ValueError(f"unknown path for this example: {path!r}")
        selector = selectors[path]
        model = eqx.tree_at(
            selector,
            model,
            replace=tuple(jnp.asarray(slot.value) for _ in selector(model)),
        )
    return model


def save_reload(
    model: ir.Model, slots: Sequence[StateSlot], path: Path, *, external_data: bool
) -> onnx.ModelProto:
    """Use the tested serializer, then verify exact defaults and tensor properties."""
    if tuple(slot.name for slot in slots) != SLOT_NAMES:
        raise ValueError("saving requires the four declared state slots in order")
    onnx.save_model(
        ir.to_proto(model),
        path,
        save_as_external_data=external_data,
        all_tensors_to_one_file=True,
        location=path.name + ".data",
        size_threshold=0,
    )
    restored = onnx.load(path)
    onnx.checker.check_model(restored, full_check=True)
    if {value.name for value in restored.graph.input} != {"x", *SLOT_NAMES}:
        raise ValueError("serialized graph inputs changed")
    for expected, actual in zip(slots, read_slots(restored)):
        source, loaded = np.asarray(expected.value), np.asarray(actual.value)
        if (
            actual.name != expected.name
            or actual.metadata != expected.metadata
            or source.dtype != loaded.dtype
            or source.shape != loaded.shape
            or source.tobytes() != loaded.tobytes()
        ):
            raise ValueError(f"serialized state changed for {expected.name}")
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
    slots: Sequence[StateSlot],
    x: jax.Array,
    target: jax.Array,
) -> tuple[tuple[StateSlot, ...], float]:
    """Interpret the example's policy; frozen state never enters AdamW at all."""
    if any(slot.metadata.get(TRAINABLE) not in {"true", "false"} for slot in slots):
        raise ValueError("this example requires explicit true/false trainability")
    trainable = {
        slot.name: jnp.asarray(slot.value)
        for slot in slots
        if slot.metadata[TRAINABLE] == "true"
    }
    frozen = {
        slot.name: jnp.asarray(slot.value)
        for slot in slots
        if slot.metadata[TRAINABLE] == "false"
    }

    def loss(parameters: dict[str, jax.Array]) -> jax.Array:
        state = {**frozen, **parameters}
        rebuilt = reconstruct(
            model,
            tuple(
                StateSlot(slot.name, state[slot.name], slot.metadata) for slot in slots
            ),
        )
        return jnp.mean((rebuilt(x) - target) ** 2)

    loss_value, gradients = jax.value_and_grad(loss)(trainable)
    optimizer = optax.adamw(learning_rate=0.01, weight_decay=0.1)
    updates, _ = optimizer.update(gradients, optimizer.init(trainable), trainable)
    updated = optax.apply_updates(trainable, updates)
    return (
        tuple(
            StateSlot(slot.name, updated.get(slot.name, slot.value), slot.metadata)
            for slot in slots
        ),
        float(loss_value),
    )


def run_demo(output_dir: Path, *, external_data: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SmallAttention(key=jax.random.key(0))
    x = jnp.linspace(-0.8, 0.9, 12, dtype=jnp.float32).reshape(3, 4)
    slots = state_slots(model)
    exported = export_model(model, x)
    path = output_dir / "attention.onnx"
    proto = save_reload(exported, slots, path, external_data=external_data)
    imported = read_slots(proto)
    expected_policy = {
        "shared_weight": ("shared_weight", "true"),
        "independent_weight": ("independent_weight", "true"),
        "rope_rotations": ("rope.rotations", "false"),
        "shared_frozen": ("q_projection.matrix;k_projection.matrix", "false"),
    }
    actual_policy = {
        slot.name: (
            slot.metadata.get("pkg.example.path"),
            slot.metadata.get(TRAINABLE),
        )
        for slot in imported
    }
    if actual_policy != expected_policy:
        raise AssertionError("serialized state policy differs from this architecture")
    session = runtime(proto)
    actual = session.run(None, {"x": np.asarray(x)})[0]
    np.testing.assert_allclose(actual, model(x), rtol=2e-5, atol=2e-6)
    overrides = {
        slot.name: np.asarray(slot.value) * np.float32(0.7 + index * 0.1)
        + np.float32(0.2 * (index + 1))
        for index, slot in enumerate(imported)
    }
    overridden = session.run(None, {"x": np.asarray(x), **overrides})[0]
    reference = reconstruct(
        model,
        tuple(
            StateSlot(slot.name, overrides[slot.name], slot.metadata)
            for slot in imported
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
        "opset": 23,
        "x64_enabled": jax.config.jax_enable_x64,
        "ort_optimization": "ORT_ENABLE_ALL",
        "slot_count": len(slots),
        "model": str(path),
        "external_data": external_data,
        "max_absolute_error": float(np.max(np.abs(actual - np.asarray(model(x))))),
        "override_max_absolute_error": float(
            np.max(np.abs(overridden - np.asarray(reference)))
        ),
        "adamw_loss": loss,
        "frozen_slots_unchanged": True,
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
