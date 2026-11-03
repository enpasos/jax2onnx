# tests/extra_tests/test_quantized_mnist_example.py

"""Offline guards for the real example; no dataset, training or native tracing."""

import hashlib
from importlib.metadata import PackageNotFoundError
import json

import numpy as np
import pytest

from scripts import quantized_mnist as example


def test_pixels_keep_fp32_row_major_layout_and_accept_zero_images():
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    images[0, 0, 0], images[0, 0, 27] = 255, 127
    images[0, 1, 0], images[0, 27, 27] = 64, 1
    result = example.preprocess(images)
    assert result.shape == (2, 784)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(
        result[0, [0, 27, 28, 783]],
        np.array([255, 127, 64, 1], dtype=np.float32) / np.float32(255),
    )
    assert np.count_nonzero(result) == 4
    assert np.count_nonzero(result[1]) == 0
    assert images[0, 0, 0] == 255


@pytest.mark.parametrize(
    "shape,dtype", [((1, 28, 28), np.float32), ((1, 784), np.uint8)]
)
def test_preprocessing_rejects_already_normalized_or_reshaped_inputs(shape, dtype):
    with pytest.raises(ValueError, match="original uint8"):
        example.preprocess(np.zeros(shape, dtype=dtype))


@pytest.mark.parametrize("held_out", [False, True])
def test_dataset_checksum_failure_precedes_all_array_access(
    tmp_path, monkeypatch, held_out
):
    data = tmp_path / "mnist.npz"
    data.write_bytes(b"unverified archive")
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("array accessed")
    )
    with pytest.raises(ValueError, match="checksum differs"):
        example._load_data(data, held_out=held_out)


@pytest.fixture
def frozen_candidate(tmp_path):
    train, dev, _ = example.fixed_split()
    arrays = {
        "dense0": np.full((784, 128), 0.125, dtype=np.float32),
        "dense1": np.full((128, 10), -0.25, dtype=np.float32),
        "train_indices": train,
        "dev_indices": dev,
    }
    np.savez(tmp_path / "checkpoint.npz", **arrays)
    manifest = {
        "state": "frozen",
        "epochs_completed": 20,
        "recipe": example.recipe(),
        "recipe_sha256": example.json_hash(example.recipe()),
        "checkpoint_sha256": hashlib.sha256(
            (tmp_path / "checkpoint.npz").read_bytes()
        ).hexdigest(),
        "split_sha256": example.split_hash(train, dev),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, arrays, manifest


def test_frozen_checkpoint_reloads_exact_typed_weights(frozen_candidate):
    directory, arrays, manifest = frozen_candidate
    kernels, loaded_manifest = example.load_checkpoint(directory)
    assert loaded_manifest == manifest
    assert set(kernels) == {"dense0", "dense1"}
    for name, value in kernels.items():
        assert value.dtype == np.float32
        np.testing.assert_array_equal(value, arrays[name])


@pytest.mark.parametrize("state,epochs", [("training", 0), ("frozen", 19)])
def test_incomplete_training_cannot_open_checkpoint(
    tmp_path, monkeypatch, state, epochs
):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"state": state, "epochs_completed": epochs})
    )
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("checkpoint opened")
    )
    with pytest.raises(ValueError, match="frozen final epoch-20"):
        example.load_checkpoint(tmp_path)


@pytest.mark.parametrize("change_recipe", [False, True])
def test_changed_recipe_or_digest_cannot_reuse_frozen_weights(
    frozen_candidate, change_recipe
):
    directory, _, manifest = frozen_candidate
    if change_recipe:
        manifest["recipe"]["epochs"] = 21
        manifest["recipe_sha256"] = example.json_hash(manifest["recipe"])
    else:
        manifest["recipe_sha256"] = "0" * 64
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen recipe changed"):
        example.load_checkpoint(directory)


def test_checkpoint_tampering_fails_before_array_access(frozen_candidate, monkeypatch):
    directory, _, _ = frozen_candidate
    with (directory / "checkpoint.npz").open("ab") as target:
        target.write(b"changed")
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("checkpoint opened")
    )
    with pytest.raises(ValueError, match="checkpoint checksum changed"):
        example.load_checkpoint(directory)


@pytest.mark.parametrize("change_indices", [False, True])
def test_split_cannot_change_even_with_updated_archive_digest(
    frozen_candidate, change_indices
):
    directory, arrays, manifest = frozen_candidate
    if change_indices:
        train = arrays["train_indices"]
        train[0], train[1] = train[1], train[0]
        np.savez(directory / "checkpoint.npz", **arrays)
        manifest["checkpoint_sha256"] = hashlib.sha256(
            (directory / "checkpoint.npz").read_bytes()
        ).hexdigest()
        manifest["split_sha256"] = example.split_hash(train, arrays["dev_indices"])
    else:
        manifest["split_sha256"] = "0" * 64
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError if change_indices else ValueError):
        example.load_checkpoint(directory)


@pytest.mark.parametrize(
    "maximum,scale",
    [
        (1.0, 1 / 64),
        (63.75, 0.5),
        (127.5, 1.0),
        (float(np.nextafter(np.float32(127.5), np.float32(np.inf))), 2.0),
    ],
)
def test_fixed_activation_range_covers_exact_and_just_above_boundaries(maximum, scale):
    selected = example.fixed_activation_range(maximum)
    assert selected["scale"] == scale
    assert selected["bound"] == 127.5 * scale
    assert selected["bound"] >= maximum
    assert selected["bound"] / 2 < maximum
    assert np.float32(selected["scale"]) == selected["scale"]


@pytest.mark.parametrize("maximum", [0.0, -1.0, np.inf, np.nan, 1e-100, 1e100])
def test_invalid_calibration_maximum_never_selects_a_fallback(maximum):
    with pytest.raises(ValueError, match="positive and finite|normal finite FP32"):
        example.fixed_activation_range(maximum)


def test_calibration_uses_every_prescribed_development_image_only(
    frozen_candidate, monkeypatch
):
    _, arrays, _ = frozen_candidate
    # Encode canonical training indices in two pixels so the native evaluator
    # boundary can be checked without JAX, inference or a real dataset.
    images = np.zeros((60_000, 28, 28), dtype=np.uint8)
    indices = np.arange(60_000)
    images[:, 0, 0], images[:, 0, 1] = indices % 256, indices // 256

    def observe(kernels, selected):
        assert kernels["dense0"].dtype == np.float32
        actual = selected[:, 0, 0].astype(np.int64) + 256 * selected[:, 0, 1].astype(
            np.int64
        )
        np.testing.assert_array_equal(actual, arrays["dev_indices"])
        assert len(actual) == len(np.unique(actual)) == 5000
        return 127.5

    monkeypatch.setattr(example, "_hidden_maximum", observe)
    calibration = example.calibrate(
        {name: arrays[name] for name in example.KERNEL_SHAPES}, images
    )
    assert calibration["count"] == 5000
    assert calibration["dense0"]["scale"] == 1 / 64
    assert calibration["dense1"]["scale"] == 1
    assert (
        calibration["development_indices_sha256"]
        == hashlib.sha256(arrays["dev_indices"].astype("<i8").tobytes()).hexdigest()
    )


def test_optional_fixed_range_full_classifier_uses_actual_export_and_probe():
    try:
        example.require_environment(qwix=True)
    except (PackageNotFoundError, RuntimeError) as error:
        pytest.skip(
            f"Optional fixed-range MNIST integration requires its pinned environment: {error}"
        )

    import jax

    kernels = {
        "dense0": np.linspace(-0.04, 0.045, 784 * 128, dtype=np.float32).reshape(
            784, 128
        ),
        "dense1": np.linspace(-0.6, 0.75, 128 * 10, dtype=np.float32).reshape(128, 10),
    }
    calibration = {
        "dense0": example.fixed_activation_range(1.0),
        "dense1": example.fixed_activation_range(2.0),
    }
    ramp = (np.arange(784) % 256).astype(np.uint8).reshape(28, 28)
    images = np.broadcast_to(ramp, (60_000, 28, 28))
    with jax.enable_x64(False):
        exports = example.export_models(kernels, calibration)
        result = example.development_probe(exports, images)
    assert result["zero_image_included"] is True
    assert len(result["development_indices"]) == 16
