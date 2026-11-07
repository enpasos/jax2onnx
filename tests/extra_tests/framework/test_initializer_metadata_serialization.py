# tests/extra_tests/framework/test_initializer_metadata_serialization.py

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import InitializerMetadataError, to_onnx
from jax2onnx import user_interface as ui


def _force_external_data(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ui._save_model_proto

    def save(proto: onnx.ModelProto, dest: str, **kwargs: Any) -> str:
        return original(proto, dest, **{**kwargs, "external_threshold": 0})

    monkeypatch.setattr(ui, "_save_model_proto", save)


@pytest.mark.parametrize("mode", ["proto", "file"])
def test_normal_float_attribute_encoding_does_not_reject_annotations(
    tmp_path: Path, mode: str
) -> None:
    weight = jnp.asarray([1.25, -2.5], dtype=jnp.float32)
    exported = to_onnx(
        lambda x: jax.nn.leaky_relu(x + weight, negative_slope=0.1),
        [(2,)],
        initializer_metadata=lambda source: (
            {"source": "weight"} if source is weight else None
        ),
        return_mode=mode,
        **({"output_path": tmp_path / "leaky.onnx"} if mode == "file" else {}),
    )
    model = onnx.load(exported) if isinstance(exported, str) else exported
    onnx.checker.check_model(model, full_check=True)
    alpha = next(
        attribute.f
        for node in model.graph.node
        if node.op_type == "LeakyRelu"
        for attribute in node.attribute
        if attribute.name == "alpha"
    )
    assert alpha == float(np.float32(0.1))
    assert {
        entry.key: entry.value for entry in model.graph.initializer[0].metadata_props
    } == {"source": "weight"}


@pytest.mark.parametrize("export_mode", ["standard", "web"])
def test_verified_file_preserves_metadata_and_external_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export_mode: str
) -> None:
    _force_external_data(monkeypatch)
    destination = tmp_path / "weights.onnx"
    destination.write_bytes(b"old model")
    sidecar = tmp_path / "weights.onnx.data"
    sidecar.write_bytes(b"old sidecar, must never be appended to")
    weight = jnp.asarray([1.25, -0.0], dtype=jnp.float32)
    annotations = {"source": "weight", "empty": "", "unicode": "ä雪"}

    result = to_onnx(
        lambda x: x + weight,
        [(2,)],
        initializer_metadata=lambda source: annotations if source is weight else None,
        return_mode="file",
        output_path=destination,
        export_mode=export_mode,
        input_names=["features"],
        output_names=["prediction"],
    )

    assert result == str(destination)
    references = onnx.load(destination, load_external_data=False)
    assert bool(references.graph.initializer[0].external_data) == (
        export_mode == "standard"
    )
    assert sidecar.exists() == (export_mode == "standard")
    loaded = onnx.load(destination)
    tensor = loaded.graph.initializer[0]
    assert {entry.key: entry.value for entry in tensor.metadata_props} == annotations
    assert onnx.numpy_helper.to_array(tensor).tobytes() == np.asarray(weight).tobytes()
    assert loaded.graph.input[0].name == "features"
    assert loaded.graph.output[0].name == "prediction"
    onnx.checker.check_model(loaded, full_check=True)
    assert not list(tmp_path.glob(".jax2onnx-export-*"))


@pytest.mark.parametrize("damage", ["properties", "payload"])
def test_file_contract_failure_leaves_both_destination_files_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    _force_external_data(monkeypatch)
    original_save = ui._save_model_proto

    def corrupt_staging(proto: onnx.ModelProto, dest: str, **kwargs: Any) -> str:
        result = original_save(proto, dest, **kwargs)
        if damage == "properties":
            written = onnx.load(dest, load_external_data=False)
            del written.graph.initializer[0].metadata_props[:]
            onnx.save_model(written, dest)
        else:
            sidecar = Path(dest + ".data")
            data = bytearray(sidecar.read_bytes())
            data[0] ^= 1
            sidecar.write_bytes(data)
        return result

    monkeypatch.setattr(ui, "_save_model_proto", corrupt_staging)
    destination = tmp_path / "weights.onnx"
    sidecar = tmp_path / "weights.onnx.data"
    destination.write_bytes(b"previous model")
    sidecar.write_bytes(b"previous data")
    weight = jnp.asarray([1.25, 2.5], dtype=jnp.float32)

    with pytest.raises(InitializerMetadataError) as caught:
        to_onnx(
            lambda x: x + weight,
            [(2,)],
            initializer_metadata=lambda source: (
                {"source": "weight"} if source is weight else None
            ),
            return_mode="file",
            output_path=destination,
        )

    assert caught.value.reason == "preservation_failed"
    assert destination.read_bytes() == b"previous model"
    assert sidecar.read_bytes() == b"previous data"
    assert set(tmp_path.iterdir()) == {destination, sidecar}


@pytest.mark.parametrize("mode", ["proto", "file"])
@pytest.mark.parametrize("damage", ["properties", "payload", "uses", "duplicate_key"])
def test_serialization_is_checked_against_validated_ir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, damage: str
) -> None:
    original_to_proto = ir.to_proto

    def corrupt_serialization(model: ir.Model) -> onnx.ModelProto:
        proto = original_to_proto(model)
        left, right = proto.graph.initializer
        if damage == "properties":
            del left.metadata_props[:]
        elif damage == "duplicate_key":
            # A map-based decoder would keep the last entry and hide this
            # ambiguous property from an IR-only comparison.
            left.metadata_props[0].value = "corrupted"
            left.metadata_props.add(key="same", value="tag")
        elif damage == "payload":
            left.CopyFrom(
                onnx.numpy_helper.from_array(
                    np.asarray([7.0, 8.0], np.float32), left.name
                )
            )
            left.metadata_props.add(key="same", value="tag")
        else:
            # The values and tags deliberately agree. Changing use association
            # must still be detected by the serialization boundary check.
            for node in proto.graph.node:
                for index, name in enumerate(node.input):
                    if name == left.name:
                        node.input[index] = right.name
                    elif name == right.name:
                        node.input[index] = left.name
        return proto

    monkeypatch.setattr(ir, "to_proto", corrupt_serialization)
    left = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    right = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous model")

    with pytest.raises(InitializerMetadataError) as caught:
        to_onnx(
            lambda x: (x + left, x * right),
            [(2,)],
            initializer_metadata=lambda source: (
                {"same": "tag"} if source is left or source is right else None
            ),
            return_mode=mode,
            **({"output_path": destination} if mode == "file" else {}),
        )

    assert caught.value.reason == "preservation_failed"
    assert destination.read_bytes() == b"previous model"
    assert set(tmp_path.iterdir()) == {destination}


def test_callback_failure_does_not_publish_or_delete_existing_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "model.onnx"
    sidecar = tmp_path / "model.onnx.data"
    destination.write_bytes(b"previous model")
    sidecar.write_bytes(b"previous data")
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    failure = RuntimeError("callback failure")

    def metadata(source: object) -> None:
        raise failure

    with pytest.raises(RuntimeError) as caught:
        to_onnx(
            lambda x: x + weight,
            [(2,)],
            initializer_metadata=metadata,
            return_mode="file",
            export_mode="web",
            output_path=destination,
        )

    assert caught.value is failure
    assert destination.read_bytes() == b"previous model"
    assert sidecar.read_bytes() == b"previous data"
    assert set(tmp_path.iterdir()) == {destination, sidecar}
