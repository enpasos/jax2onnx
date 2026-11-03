#!/usr/bin/env python3
# scripts/quantized_mnist.py

"""Prepare the one fixed MNIST classifier; training and downloads are outside CI.

Run from the repository root in the pinned optional Python 3.12 environment::

    python scripts/quantized_mnist.py train --data /tmp/mnist.npz --workdir /tmp/mnist-run
    python scripts/quantized_mnist.py export --data /tmp/mnist.npz --workdir /tmp/mnist-run
    python scripts/quantized_mnist.py validate --data /tmp/mnist.npz --workdir /tmp/mnist-run

Supply the verified Keras MNIST archive; this script never downloads data.
``train`` reads only training arrays and freezes its final epoch-20 checkpoint.
``export`` uses development images only. ``validate`` explicitly opens the
held-out test arrays after checking the frozen weights, recipe and model hashes.
Use the separate NumPy/ORT-only script for independent deployment acceptance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import distribution, version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable

import numpy as np


QWIX_COMMIT = "9ee1079245682ed82e5da00ce855e4daf77a086b"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
MNIST_PROVENANCE = {
    "url": MNIST_URL,
    "sha256": MNIST_SHA256,
    "source": "https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py",
    "source_blob": "64db64388ccb017f2abb7ba590e25406f76d0725",
    "attribution": (
        "MNIST by Yann LeCun and Corinna Cortes, derived from NIST data; "
        "distributed under Creative Commons Attribution-ShareAlike 3.0."
    ),
    "license": "https://creativecommons.org/licenses/by-sa/3.0/",
}
PINS = {
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "flax": "0.12.8",
    "numpy": "2.4.6",
    "optax": "0.2.8",
    "onnx": "1.23.2",
    "onnxruntime": "1.31.1",
}
KERNEL_SHAPES = {"dense0": (784, 128), "dense1": (128, 10)}
RTOL = ATOL = 1e-6


def recipe() -> dict[str, Any]:
    """Return the fixed recipe, independently hashable before test access."""
    return {
        "model": "784 -> Dense(128, no bias) -> ReLU -> Dense(10, no bias)",
        "dtype": "float32",
        "preprocessing": "uint8 -> float32 / float32(255), row-major flatten",
        "seed": 0,
        "split": "PCG64(0): first 55000 of permutation(60000) train, last 5000 dev",
        "shuffle": "continue the split generator, one permutation(55000) per epoch",
        "epochs": 20,
        "training_batch_size": 128,
        "include_remainder": True,
        "loss": "mean softmax cross-entropy with integer labels",
        "adam": {
            "learning_rate": 1e-3,
            "b1": 0.9,
            "b2": 0.999,
            "eps": 1e-8,
            "eps_root": 0.0,
            "mu_dtype": None,
            "nesterov": False,
        },
        "inference_batch_size": 1,
        "opset": 23,
        "qwix": {
            "version": "0.1.8",
            "commit": QWIX_COMMIT,
            "op_names": ["dot_general"],
            "weight_qtype": "int8",
            "act_qtype": "int8",
            "act_static_scale": False,
            "weight_calibration_method": "absmax",
            "act_calibration_method": "fixed,<127.5 * power-of-two scale>, per layer",
            "activation_calibration": {
                "dense0": "known input maximum 1.0; scale 1/64",
                "dense1": "FP32 baseline post-ReLU maximum over all 5000 development images, batch 1",
                "scale": "smallest positive normal FP32 power of two covering maximum / 127.5",
                "invalid_maximum": "reject zero, negative or non-finite; no fallback",
            },
            "tile_size": None,
            "disable_jit": True,
            "zero_point": None,
        },
        "dataset_sha256": MNIST_SHA256,
        "environment": {"python": "3.12", **PINS},
        "acceptance": {
            "fp32_correct_min": 9700,
            "extra_errors_max": 50,
            "artifact_ratio_max": 0.4,
            "w8a8_rtol": RTOL,
            "w8a8_atol": ATOL,
            "same_w8a8_predictions": True,
        },
    }


def json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")


def require_environment(*, qwix: bool = False) -> dict[str, str]:
    """Check metadata before importing the native preparation stack."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "The optional MNIST preparation environment requires Python 3.12"
        )
    installed = {name: version(name) for name in PINS}
    wrong = {name: found for name, found in installed.items() if found != PINS[name]}
    if wrong:
        raise RuntimeError(
            f"Use scripts/requirements-quantized-mnist.txt; wrong versions: {wrong}"
        )
    if qwix:
        package = distribution("qwix")
        direct_url = json.loads(package.read_text("direct_url.json") or "{}")
        if (
            package.version != "0.1.8"
            or direct_url.get("vcs_info", {}).get("commit_id") != QWIX_COMMIT
        ):
            raise RuntimeError(
                f"Qwix must be installed from immutable commit {QWIX_COMMIT}"
            )
        installed["qwix"] = package.version
        installed["qwix_commit"] = QWIX_COMMIT
    installed["python"] = platform.python_version()
    installed["jax2onnx"] = version("jax2onnx")
    return installed


def fixed_split() -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    generator = np.random.Generator(np.random.PCG64(0))
    indices = generator.permutation(60_000).astype("<i8")
    return indices[:55_000], indices[55_000:], generator


def split_hash(train_indices: np.ndarray, dev_indices: np.ndarray) -> str:
    indices = np.concatenate((train_indices, dev_indices)).astype("<i8")
    return hashlib.sha256(indices.tobytes()).hexdigest()


def preprocess(images: np.ndarray) -> np.ndarray:
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("Expected original uint8 images of shape (batch, 28, 28)")
    return images.reshape(len(images), 784).astype(np.float32) / np.float32(255)


def _verify_dataset(path: Path) -> None:
    if file_hash(path) != MNIST_SHA256:
        raise ValueError(
            f"MNIST archive checksum differs from the canonical file at {MNIST_URL}"
        )


def _load_data(path: Path, *, held_out: bool = False) -> tuple[np.ndarray, np.ndarray]:
    _verify_dataset(path)
    # np.load reads ZIP members lazily. Training never opens x_test or y_test.
    with np.load(path, allow_pickle=False) as archive:
        suffix = "test" if held_out else "train"
        images, labels = archive[f"x_{suffix}"], archive[f"y_{suffix}"]
    count = 10_000 if held_out else 60_000
    if images.dtype != np.uint8 or images.shape != (count, 28, 28):
        raise ValueError(f"Unexpected canonical MNIST {suffix} image representation")
    if labels.dtype != np.uint8 or labels.shape != (count,) or np.any(labels > 9):
        raise ValueError(f"Unexpected canonical MNIST {suffix} labels")
    return images, labels


def make_model() -> Any:
    """Construct a fresh Linen class so export cannot inherit a serving trace."""
    from flax import linen as nn
    import jax.numpy as jnp

    class Classifier(nn.Module):
        def setup(self) -> None:
            self.dense0 = nn.Dense(
                128,
                use_bias=False,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                name="dense0",
            )
            self.dense1 = nn.Dense(
                10,
                use_bias=False,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                name="dense1",
            )

        def hidden(self, inputs: Any) -> Any:
            return nn.relu(self.dense0(inputs))

        def __call__(self, inputs: Any) -> Any:
            return self.dense1(self.hidden(inputs))

    return Classifier()


def _cpu_jax() -> Any:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    if jax.default_backend() != "cpu":
        raise RuntimeError("The qualified example uses official JAX CPU builds only")
    return jax


def train(data: Path, workdir: Path) -> None:
    versions = require_environment()
    workdir.mkdir(parents=True, exist_ok=True)
    if (workdir / "manifest.json").exists() or (workdir / "checkpoint.npz").exists():
        raise FileExistsError(
            "This candidate already started; preserve its evidence and do not retrain"
        )
    images, labels = _load_data(data)
    train_indices, dev_indices, generator = fixed_split()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {
        "state": "training",
        "recipe": recipe(),
        "recipe_sha256": json_hash(recipe()),
        "versions": versions,
        "dataset": MNIST_PROVENANCE,
        "git_revision": revision,
        "preparation_script_sha256": file_hash(Path(__file__).resolve()),
        "split_sha256": split_hash(train_indices, dev_indices),
    }
    # A failed run leaves this marker; it must not silently become a second run.
    _write_json(workdir / "manifest.json", manifest)
    jax = _cpu_jax()
    import jax.numpy as jnp
    import optax

    with jax.enable_x64(False):
        model = make_model()
        params = model.init(jax.random.key(0), jnp.zeros((1, 784), jnp.float32))[
            "params"
        ]
        optimizer = optax.adam(
            1e-3,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            eps_root=0.0,
            mu_dtype=None,
            nesterov=False,
        )
        state = optimizer.init(params)

        @jax.jit
        def step(parameters: Any, opt_state: Any, batch: Any, targets: Any) -> Any:
            def loss(current: Any) -> Any:
                logits = model.apply({"params": current}, batch)
                return optax.softmax_cross_entropy_with_integer_labels(
                    logits, targets
                ).mean()

            value, gradient = jax.value_and_grad(loss)(parameters)
            updates, opt_state = optimizer.update(gradient, opt_state, parameters)
            return optax.apply_updates(parameters, updates), opt_state, value

        losses = []
        for epoch in range(20):
            shuffled = train_indices[generator.permutation(len(train_indices))]
            loss_sum = 0.0
            for start in range(0, len(shuffled), 128):
                indices = shuffled[start : start + 128]
                params, state, loss = step(
                    params,
                    state,
                    jnp.asarray(preprocess(images[indices])),
                    jnp.asarray(labels[indices], dtype=jnp.int32),
                )
                value = float(
                    loss
                )  # Synchronize each batch instead of accumulating queued work.
                if not np.isfinite(value):
                    raise ValueError(f"Non-finite training loss at epoch {epoch + 1}")
                loss_sum += value * len(indices)
            losses.append(loss_sum / len(shuffled))
            print(
                f"epoch {epoch + 1:02d}/20 mean cross-entropy {losses[-1]:.8f}",
                flush=True,
            )
        kernels = {name: np.asarray(params[name]["kernel"]) for name in KERNEL_SHAPES}
    _validate_kernels(kernels)
    checkpoint = workdir / "checkpoint.npz"
    with checkpoint.open("xb") as target:
        np.savez(
            target, **kernels, train_indices=train_indices, dev_indices=dev_indices
        )
    manifest.update(
        state="frozen",
        checkpoint_sha256=file_hash(checkpoint),
        training_mean_cross_entropy=losses,
        epochs_completed=20,
        held_out_accessed=False,
    )
    _write_json(workdir / "manifest.frozen.json", manifest)
    (workdir / "manifest.frozen.json").replace(workdir / "manifest.json")
    print(f"Frozen checkpoint SHA-256: {manifest['checkpoint_sha256']}", flush=True)


def _validate_kernels(kernels: dict[str, np.ndarray]) -> None:
    if set(kernels) != set(KERNEL_SHAPES):
        raise ValueError("Expected exactly dense0 and dense1 kernels")
    for name, shape in KERNEL_SHAPES.items():
        value = kernels[name]
        if (
            value.shape != shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Expected finite FP32 kernel {name}{shape}")


def load_checkpoint(workdir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((workdir / "manifest.json").read_text())
    if manifest["state"] != "frozen" or manifest["epochs_completed"] != 20:
        raise ValueError(
            "Only a frozen final epoch-20 checkpoint can be exported/evaluated"
        )
    if manifest["recipe"] != recipe() or manifest["recipe_sha256"] != json_hash(
        recipe()
    ):
        raise ValueError("The frozen recipe changed; do not select another candidate")
    checkpoint = workdir / "checkpoint.npz"
    if file_hash(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint checksum changed")
    with np.load(checkpoint, allow_pickle=False) as archive:
        kernels = {name: archive[name] for name in KERNEL_SHAPES}
        train_indices, dev_indices = archive["train_indices"], archive["dev_indices"]
    _validate_kernels(kernels)
    expected_train, expected_dev, _ = fixed_split()
    np.testing.assert_array_equal(train_indices, expected_train)
    np.testing.assert_array_equal(dev_indices, expected_dev)
    if split_hash(train_indices, dev_indices) != manifest["split_sha256"]:
        raise ValueError("Frozen data split checksum changed")
    return kernels, manifest


@dataclass
class Exports:
    models: dict[str, Any]
    serving: dict[str, Callable[[Any], Any]]
    storage: dict[str, Any]


def fixed_activation_range(maximum: float) -> dict[str, float]:
    """Choose once from the observed maximum; never search accuracy or ranges."""
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError(
            "Activation maximum must be positive and finite; no fallback range"
        )
    ratio = maximum / 127.5
    if ratio < np.finfo(np.float32).tiny:
        raise ValueError("Activation scale and bound must be normal finite FP32 values")
    fraction, exponent = math.frexp(ratio)
    scale = math.ldexp(1.0, exponent - (fraction == 0.5))
    bound = 127.5 * scale
    if scale < np.finfo(np.float32).tiny or bound > np.finfo(np.float32).max:
        raise ValueError("Activation scale and bound must be normal finite FP32 values")
    return {"maximum": maximum, "scale": scale, "bound": bound}


def _hidden_maximum(kernels: dict[str, np.ndarray], images: np.ndarray) -> float:
    jax = _cpu_jax()
    import jax.numpy as jnp

    model = make_model()
    params = {name: {"kernel": jnp.asarray(value)} for name, value in kernels.items()}
    hidden = jax.jit(
        lambda values: model.apply({"params": params}, values, method=model.hidden)
    )
    maximum = 0.0
    for image in images:
        values = np.asarray(hidden(jnp.asarray(preprocess(image[None]))))
        if (
            values.shape != (1, 128)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Calibration requires finite FP32 baseline post-ReLU hidden values"
            )
        maximum = max(maximum, float(values.max()))
    return maximum


def calibrate(kernels: dict[str, np.ndarray], images: np.ndarray) -> dict[str, Any]:
    """Read only the complete prescribed development split, never test examples."""
    _validate_kernels(kernels)
    if images.dtype != np.uint8 or images.shape != (60_000, 28, 28):
        raise ValueError("Calibration requires the canonical raw training image array")
    _, indices, _ = fixed_split()
    return {
        "source": "FP32 baseline post-ReLU, all prescribed development images, batch 1",
        "count": len(indices),
        "development_indices_sha256": hashlib.sha256(
            indices.astype("<i8").tobytes()
        ).hexdigest(),
        "dense0": fixed_activation_range(1.0),
        "dense1": fixed_activation_range(_hidden_maximum(kernels, images[indices])),
    }


def export_models(
    kernels: dict[str, np.ndarray], calibration: dict[str, Any]
) -> Exports:
    """Export fresh actual Linen/Qwix callables before any native serving trace."""
    _validate_kernels(kernels)
    jax = _cpu_jax()
    import jax.numpy as jnp
    import qwix
    from jax2onnx import to_onnx

    params = {name: {"kernel": jnp.asarray(value)} for name, value in kernels.items()}
    fp_model = make_model()

    def fp32(values: Any) -> Any:
        return fp_model.apply({"params": params}, values)

    rules = []
    for name in KERNEL_SHAPES:
        selected = calibration[name]
        if selected != fixed_activation_range(selected["maximum"]):
            raise ValueError(
                "Frozen activation range differs from the prescribed power-of-two rule"
            )
        method = f"fixed,{selected['bound']}"
        # Verify public Qwix behavior, without private globals or altered arithmetic.
        checked = qwix.quantize(
            jnp.zeros((1, KERNEL_SHAPES[name][0]), jnp.float32),
            jnp.int8,
            calibration_method=method,
        )
        scale = np.asarray(checked.scale)
        if (
            scale.dtype != np.float32
            or not np.all(scale == np.float32(selected["scale"]))
            or checked.zero_point is not None
        ):
            raise ValueError(
                "Public Qwix fixed calibration did not produce the exact FP32 power-of-two scale"
            )
        rules.append(
            qwix.QuantizationRule(
                module_path=name,
                op_names=("dot_general",),
                weight_qtype=jnp.int8,
                act_qtype=jnp.int8,
                act_static_scale=False,
                weight_calibration_method="absmax",
                act_calibration_method=method,
                tile_size=None,
            )
        )
    ptq = qwix.quantize_model(make_model(), qwix.PtqProvider(rules, disable_jit=True))
    abstract = jax.eval_shape(
        ptq.init, jax.random.key(0), jax.ShapeDtypeStruct((1, 784), jnp.float32)
    )
    quantized = qwix.quantize_params(params, abstract["params"])
    prepared = {}
    for name in KERNEL_SHAPES:
        value = quantized[name]["kernel"].array
        if not isinstance(value, qwix.QArray) or value.zero_point is not None:
            raise ValueError(
                "Expected symmetric Qwix QArray weights without zero points"
            )
        qvalue, scale = np.asarray(value.qvalue), np.asarray(value.scale)
        if (
            qvalue.dtype != np.int8
            or scale.dtype != np.float32
            or not np.isfinite(scale).all()
        ):
            raise ValueError("Expected real INT8 weights with finite FP32 scales")
        prepared[name] = {"qvalue": qvalue, "scale": scale}

    def w8a8(values: Any) -> Any:
        return ptq.apply({"params": quantized}, values)

    serving = {"fp32": fp32, "w8a8": w8a8}
    models = {
        name: to_onnx(
            function,
            [jax.ShapeDtypeStruct((1, 784), jnp.float32)],
            opset=23,
            input_names=["pixels"],
            output_names=["logits"],
        )
        for name, function in serving.items()
    }
    storage = check_models(models, prepared, calibration)
    return Exports(models, serving, storage)


def check_models(
    models: dict[str, Any], prepared: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    import onnx

    for model in models.values():
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
        for values, shape in (
            (model.graph.input, (1, 784)),
            (model.graph.output, (1, 10)),
        ):
            if (
                len(values) != 1
                or values[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            ):
                raise ValueError("Expected one FP32 input and one FP32 logits output")
            if (
                tuple(dim.dim_value for dim in values[0].type.tensor_type.shape.dim)
                != shape
            ):
                raise ValueError(f"Expected fixed graph shape {shape}")
        if any(
            tensor.data_location == onnx.TensorProto.EXTERNAL
            for tensor in model.graph.initializer
        ):
            raise ValueError("This small example writes one inline ONNX file per model")
    quantized = onnx.shape_inference.infer_shapes(models["w8a8"], strict_mode=True)
    dots = [node for node in quantized.graph.node if node.op_type == "MatMulInteger"]
    if len(dots) != 2 or any(
        node.op_type in {"Gemm", "MatMul", "Einsum"} for node in quantized.graph.node
    ):
        raise ValueError("Both prepared contractions must lower to MatMulInteger")
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (
            *quantized.graph.input,
            *quantized.graph.value_info,
            *quantized.graph.output,
        )
    }
    constants = {
        tensor.name: onnx.numpy_helper.to_array(tensor)
        for tensor in quantized.graph.initializer
    }
    types.update(
        (tensor.name, tensor.data_type) for tensor in quantized.graph.initializer
    )
    for node in quantized.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    constants[node.output[0]] = onnx.numpy_helper.to_array(attribute.t)
                    types[node.output[0]] = attribute.t.data_type
    producers = {name: node for node in quantized.graph.node for name in node.output}
    scales = {}
    matched = []
    for node in dots:
        if len(node.input) != 2 or any(
            types.get(name) != onnx.TensorProto.INT8 for name in node.input
        ):
            raise ValueError(
                "MatMulInteger must consume two INT8 operands without zero points"
            )
        if types.get(node.output[0]) != onnx.TensorProto.INT32:
            raise ValueError("MatMulInteger must produce INT32")
        # This fixed two-dimensional model needs no weight transform. A dynamic
        # RHS would conceal repeated weight quantization or unused witness constants.
        if node.input[1] not in constants:
            raise ValueError(
                "Integer matrix weights must be stored constants, not computed at runtime"
            )
        stored = constants[node.input[1]]
        matches = [
            name
            for name, value in prepared.items()
            if stored.dtype == np.int8
            and stored.shape == value["qvalue"].shape
            and stored.tobytes() == value["qvalue"].tobytes()
        ]
        if len(matches) != 1:
            raise ValueError("Stored INT8 weight bytes differ from Qwix preparation")
        matched.extend(matches)
        # Follow only the current activation quantizer, stopping at Round so
        # the second layer cannot accidentally match the first layer's Div.
        pending, rounded, visited = [node.input[0]], [], set()
        while pending:
            dependency = pending.pop()
            if dependency in visited:
                continue
            visited.add(dependency)
            producer = producers.get(dependency)
            if producer is None:
                continue
            if producer.op_type == "Round":
                rounded.append(producer)
            elif producer.op_type != "MatMulInteger":
                pending.extend(producer.input)
        if len(rounded) != 1:
            raise ValueError(
                "Expected one activation Round before each integer contraction"
            )
        division = producers[rounded[0].input[0]]
        while division.op_type in {"Reshape", "Identity"}:
            division = producers[division.input[0]]
        if division.op_type != "Div":
            raise ValueError(
                "Expected source activation division before nearest-even Round"
            )
        scale_name = division.input[1]
        pending, visited = [scale_name], set()
        while pending:
            dependency = pending.pop()
            if dependency in visited or dependency in constants:
                continue
            visited.add(dependency)
            if dependency not in producers:
                raise ValueError(
                    "Activation scale must depend only on stored constants"
                )
            pending.extend(producers[dependency].input)
        scales[matches[0]] = {
            "graph_value": scale_name,
            "expected": calibration[matches[0]]["scale"],
        }
    if sorted(matched) != sorted(KERNEL_SHAPES):
        raise ValueError("Each prepared kernel must feed its integer contraction")
    for value in constants.values():
        if value.dtype.kind == "f" and value.size >= min(
            np.prod(shape) for shape in KERNEL_SHAPES.values()
        ):
            raise ValueError(
                "Quantized graph retains a floating-point weight-sized constant"
            )
    return {
        name: {
            "shape": list(value["qvalue"].shape),
            "int8_sha256": hashlib.sha256(value["qvalue"].tobytes()).hexdigest(),
            "scale_shape": list(value["scale"].shape),
            "scale_sha256": hashlib.sha256(value["scale"].tobytes()).hexdigest(),
            "activation_scale": scales[name],
        }
        for name, value in prepared.items()
    }


def native_references(
    serving: dict[str, Callable[[Any], Any]], images: np.ndarray
) -> dict[str, np.ndarray]:
    jax = _cpu_jax()
    import jax.numpy as jnp

    outputs = {}
    for model_name, serve in serving.items():

        def fresh(values: Any) -> Any:
            return serve(values)

        compiled = jax.jit(fresh)
        # A newly jitted callable sweeps the set once, then the same executable
        # sweeps it again. Only the first call compiles; input batch size stays 1.
        for role, function in (
            ("eager", serve),
            ("fresh_jit", compiled),
            ("warm_jit", compiled),
        ):
            result = np.empty((len(images), 10), dtype=np.float32)
            for index in range(len(images)):
                actual = np.asarray(
                    function(jnp.asarray(preprocess(images[index : index + 1])))
                )
                if (
                    actual.shape != (1, 10)
                    or actual.dtype != np.float32
                    or not np.isfinite(actual).all()
                ):
                    raise ValueError(
                        f"Invalid {model_name}/{role} output for image {index}"
                    )
                result[index] = actual[0]
                if len(images) == 10_000 and (index + 1) % 1000 == 0:
                    print(f"{model_name}/{role}: {index + 1}/10000", flush=True)
            outputs[f"{model_name}_{role}"] = result
    return outputs


def check_native_fidelity(outputs: dict[str, np.ndarray]) -> None:
    for left, right in (
        ("eager", "fresh_jit"),
        ("eager", "warm_jit"),
        ("fresh_jit", "warm_jit"),
    ):
        reference = outputs[f"w8a8_{left}"]
        actual = outputs[f"w8a8_{right}"]
        np.testing.assert_allclose(
            actual, reference, rtol=RTOL, atol=ATOL, err_msg=f"native {left}/{right}"
        )
        np.testing.assert_array_equal(
            actual.argmax(axis=1),
            reference.argmax(axis=1),
            err_msg=f"native class predictions {left}/{right}",
        )


def development_probe(exports: Exports, images: np.ndarray) -> dict[str, Any]:
    """Check the first 16 fixed development images plus one legitimate zero image."""
    import onnx
    import onnxruntime as ort

    _, dev_indices, _ = fixed_split()
    probe = np.concatenate(
        (images[dev_indices[:16]], np.zeros((1, 28, 28), dtype=np.uint8))
    )
    outputs = native_references(exports.serving, probe)
    check_native_fidelity(outputs)
    for name, model in exports.models.items():
        inspection = onnx.ModelProto()
        inspection.CopyFrom(model)
        scales = (
            [value["activation_scale"] for value in exports.storage.values()]
            if name == "w8a8"
            else []
        )
        for scale in scales:
            inspection.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    scale["graph_value"], onnx.TensorProto.FLOAT, None
                )
            )
        for optimization in (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ):
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.graph_optimization_level = optimization
            session = ort.InferenceSession(
                model.SerializeToString(),
                options,
                providers=["CPUExecutionProvider"],
            )
            for index, image in enumerate(probe):
                (actual,) = session.run(None, {"pixels": preprocess(image[None])})
                if (
                    actual.shape != (1, 10)
                    or actual.dtype != np.float32
                    or not np.isfinite(actual).all()
                ):
                    raise ValueError(
                        f"Invalid development ORT output for {name}/{optimization}"
                    )
                if name == "w8a8":
                    reference = outputs["w8a8_eager"][index : index + 1]
                    context = f"{name} development image {index}, ORT {optimization}"
                    np.testing.assert_allclose(
                        actual, reference, rtol=RTOL, atol=ATOL, err_msg=context
                    )
                    np.testing.assert_array_equal(
                        actual.argmax(axis=1),
                        reference.argmax(axis=1),
                        err_msg=f"class predictions: {context}",
                    )
            del session
            if scales:
                inspection_session = ort.InferenceSession(
                    inspection.SerializeToString(),
                    options,
                    providers=["CPUExecutionProvider"],
                )
                actual_scales = inspection_session.run(
                    [value["graph_value"] for value in scales],
                    {"pixels": preprocess(probe[:1])},
                )
                for expected, actual_scale in zip(scales, actual_scales):
                    if actual_scale.dtype != np.float32 or not np.all(
                        actual_scale == np.float32(expected["expected"])
                    ):
                        raise ValueError(
                            f"Actual activation Div denominator is not the frozen power of two: {expected}"
                        )
                del inspection_session
    return {
        "scope": "first 16 development images plus zero image; not task-quality acceptance",
        "development_indices": dev_indices[:16].tolist(),
        "zero_image_included": True,
    }


def export(data: Path, workdir: Path) -> None:
    versions = require_environment(qwix=True)
    kernels, manifest = load_checkpoint(workdir)
    for name in ("fp32.onnx", "w8a8.onnx", "export.json"):
        if (workdir / name).exists():
            raise FileExistsError(f"Preserve frozen export artifact: {workdir / name}")
    images, _ = _load_data(data)
    jax = _cpu_jax()
    with jax.enable_x64(False):
        calibration = calibrate(kernels, images)
        exports = export_models(kernels, calibration)
        serialized = {
            name: model.SerializeToString() for name, model in exports.models.items()
        }
        ratio = len(serialized["w8a8"]) / len(serialized["fp32"])
        if ratio > 0.4:
            raise ValueError(
                f"Quantized artifact ratio {ratio:.6f} exceeds the frozen 0.4 gate"
            )
        probe = development_probe(exports, images)
    for name, payload in serialized.items():
        with (workdir / f"{name}.onnx").open("xb") as target:
            target.write(payload)
    _write_json(
        workdir / "export.json",
        {
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "recipe_sha256": manifest["recipe_sha256"],
            "versions": versions,
            "storage": exports.storage,
            "calibration": calibration,
            "calibration_sha256": json_hash(calibration),
            "development_probe": probe,
            "artifacts": {
                name: {
                    "file": f"{name}.onnx",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for name, payload in serialized.items()
            },
            "artifact_ratio": ratio,
            "held_out_accessed": False,
        },
    )
    print(f"Frozen inline ONNX artifact ratio: {ratio:.6f}", flush=True)


def validate(data: Path, workdir: Path) -> None:
    versions = require_environment(qwix=True)
    kernels, manifest = load_checkpoint(workdir)
    frozen = json.loads((workdir / "export.json").read_text())
    if (
        frozen["checkpoint_sha256"] != manifest["checkpoint_sha256"]
        or frozen["recipe_sha256"] != manifest["recipe_sha256"]
    ):
        raise ValueError("Export does not belong to the frozen checkpoint/recipe")
    if json_hash(frozen["calibration"]) != frozen["calibration_sha256"]:
        raise ValueError(
            "Frozen calibration checksum changed; never refit on held-out data"
        )
    target = workdir / "native_outputs.npz"
    if target.exists() or (workdir / "native_validation.json").exists():
        raise FileExistsError("Preserve the existing held-out evaluation evidence")
    jax = _cpu_jax()
    with jax.enable_x64(False):
        # Re-export in memory before native tracing; never rewrite frozen files.
        exports = export_models(kernels, frozen["calibration"])
        for name, model in exports.models.items():
            expected = frozen["artifacts"][name]["sha256"]
            if file_hash(workdir / f"{name}.onnx") != expected:
                raise ValueError(f"Frozen {name} artifact changed")
            if hashlib.sha256(model.SerializeToString()).hexdigest() != expected:
                raise ValueError(
                    f"Fresh {name} export differs from the frozen artifact; investigate before test access"
                )
        images, labels = _load_data(data, held_out=True)
        outputs = native_references(exports.serving, images)
    # Preserve the actual references even if a subsequent acceptance gate fails.
    with target.open("xb") as archive:
        np.savez(archive, **outputs, labels=labels)
    check_native_fidelity(outputs)
    correct = {
        name: int(np.count_nonzero(value.argmax(axis=1) == labels))
        for name, value in outputs.items()
    }
    report = {
        "versions": versions,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "recipe_sha256": manifest["recipe_sha256"],
        "native_outputs_sha256": file_hash(target),
        "count": 10_000,
        "correct": correct,
        "jit_reference_scope": "newly jitted first sweep; warmed second sweep, batch 1",
        "scope": "native references; independent ORT acceptance remains required",
    }
    _write_json(workdir / "native_validation.json", report)
    for role in ("eager", "fresh_jit", "warm_jit"):
        if correct[f"fp32_{role}"] < 9700:
            raise ValueError(
                f"FP32 {role} accuracy failed the fixed 97% gate: {correct}"
            )
        if correct[f"fp32_{role}"] - correct[f"w8a8_{role}"] > 50:
            raise ValueError(
                f"W8A8 {role} accuracy lost more than 50 correct predictions: {correct}"
            )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "export", "validate"))
    parser.add_argument(
        "--data", type=Path, required=True, help="Verified canonical Keras mnist.npz"
    )
    parser.add_argument(
        "--workdir", type=Path, required=True, help="Local ignored candidate directory"
    )
    args = parser.parse_args()
    {"train": train, "export": export, "validate": validate}[args.command](
        args.data, args.workdir
    )


if __name__ == "__main__":
    main()
