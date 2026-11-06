# tests/extra_tests/framework/test_parameter_metadata.py

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import (
    ParameterMetadata,
    annotate_parameters,
    read_parameter_metadata,
    to_onnx,
)


_METADATA_KEY = "pkg.jax2onnx.parameter_metadata"
Model: TypeAlias = onnx.ModelProto | ir.Model
ModelKind: TypeAlias = Literal["proto", "ir"]


def _as_proto(model: Model) -> onnx.ModelProto:
    return model if isinstance(model, onnx.ModelProto) else ir.to_proto(model)


def _small_model(kind: ModelKind) -> Model:
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "weight"], ["projected"]),
            helper.make_node("Tanh", ["projected"], ["prediction"]),
        ],
        "parameter_metadata",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [1, 2])],
        initializer=[
            numpy_helper.from_array(np.eye(2, dtype=np.float32), name="weight")
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    return model if kind == "proto" else ir.from_proto(model)


@pytest.fixture(params=["proto", "ir"])
def model(request: pytest.FixtureRequest) -> Model:
    return _small_model(request.param)


def _set_metadata(model: Model, key: str, value: str) -> None:
    if isinstance(model, onnx.ModelProto):
        model.metadata_props.add(key=key, value=value)
    else:
        model.metadata_props[key] = value


def _without_model_metadata(model: Model) -> bytes:
    proto = onnx.ModelProto()
    proto.CopyFrom(_as_proto(model))
    proto.ClearField("metadata_props")
    return cast(bytes, proto.SerializeToString())


def _export_linear(kind: ModelKind) -> Model:
    weight = jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10
    bias = jnp.asarray([0.2, -0.3], dtype=jnp.float32)

    def linear(features: jax.Array) -> jax.Array:
        return features @ weight + bias

    inputs = [jax.ShapeDtypeStruct((2, 3), jnp.float32)]
    if kind == "ir":
        return to_onnx(
            linear,
            inputs,
            return_mode="ir",
            input_names=["features"],
            output_names=["predictions"],
            opset=18,
        )
    return to_onnx(
        linear,
        inputs,
        input_names=["features"],
        output_names=["predictions"],
        opset=18,
    )


def _linear_parameter_names(model: Model) -> tuple[str, str]:
    initializers = _as_proto(model).graph.initializer
    weight = next(t.name for t in initializers if tuple(t.dims) == (3, 2))
    bias = next(t.name for t in initializers if tuple(t.dims) == (1, 2))
    return weight, bias


def _run(model: onnx.ModelProto, features: NDArray[np.float32]) -> NDArray[np.float32]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return cast(
        NDArray[np.float32], np.asarray(session.run(None, {"features": features})[0])
    )


def test_missing_annotations_are_unknown_and_empty_update_is_noop(model: Model) -> None:
    before = _as_proto(model).SerializeToString()
    assert read_parameter_metadata(model) == {}
    assert annotate_parameters(model, {}) is None
    assert read_parameter_metadata(model) == {}
    assert _as_proto(model).SerializeToString() == before


def test_explicit_unknown_frozen_and_trainable_are_distinct(model: Model) -> None:
    annotate_parameters(
        model,
        {
            "weight": ParameterMetadata(trainable=False, source_path="layers.0.weight"),
            "features": ParameterMetadata(),
        },
    )
    assert read_parameter_metadata(model) == {
        "weight": ParameterMetadata(trainable=False, source_path="layers.0.weight"),
        "features": ParameterMetadata(trainable=None),
    }
    annotate_parameters(model, {"weight": ParameterMetadata(trainable=True)})
    assert read_parameter_metadata(model) == {
        "weight": ParameterMetadata(trainable=True),
        "features": ParameterMetadata(),
    }
    # Reading returns a snapshot, so a consumer cannot mutate the model by editing it.
    snapshot = read_parameter_metadata(model)
    snapshot.clear()
    assert len(read_parameter_metadata(model)) == 2


def test_parameter_metadata_is_frozen() -> None:
    metadata = ParameterMetadata(trainable=True, source_path="dense.kernel")
    with pytest.raises(FrozenInstanceError):
        setattr(metadata, "trainable", False)


@pytest.mark.parametrize("kind", ["proto", "ir"])
def test_export_metadata_roundtrip_preserves_model_and_forward_result(
    kind: ModelKind, tmp_path: Path
) -> None:
    model = _export_linear(kind)
    weight, bias = _linear_parameter_names(model)
    _set_metadata(model, "application.owner", "caller")
    before = _without_model_metadata(model)
    features = np.arange(6, dtype=np.float32).reshape(2, 3) / 5
    original_result = _run(_as_proto(model), features)
    annotations = {
        weight: ParameterMetadata(trainable=True, source_path="params.dense.kernel"),
        bias: ParameterMetadata(trainable=False, source_path="params.dense.bias"),
        "features": ParameterMetadata(trainable=False),
    }
    assert annotate_parameters(model, annotations) is None
    assert _without_model_metadata(model) == before
    props = {entry.key: entry.value for entry in _as_proto(model).metadata_props}
    assert props["application.owner"] == "caller"
    manifest = json.loads(props[_METADATA_KEY])
    assert manifest == {
        "version": 1,
        "parameters": {
            name: {
                "trainable": metadata.trainable,
                **(
                    {"source_path": metadata.source_path}
                    if metadata.source_path is not None
                    else {}
                ),
            }
            for name, metadata in annotations.items()
        },
    }

    path = tmp_path / f"linear-{kind}.onnx"
    if isinstance(model, ir.Model):
        ir.save(model, path)
    else:
        onnx.save_model(model, path)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    assert read_parameter_metadata(reloaded) == annotations
    roundtripped_ir = ir.from_proto(reloaded)
    assert read_parameter_metadata(roundtripped_ir) == annotations
    reserialized = ir.to_proto(roundtripped_ir)
    assert read_parameter_metadata(reserialized) == annotations
    assert _without_model_metadata(reserialized) == before
    np.testing.assert_array_equal(_run(reserialized, features), original_result)
    expected = features @ (
        np.arange(6, dtype=np.float32).reshape(3, 2) / 10
    ) + np.asarray([0.2, -0.3], dtype=np.float32)
    np.testing.assert_allclose(original_result, expected, rtol=1e-6, atol=1e-6)


def test_downstream_consumer_updates_only_explicitly_trainable_initializers(
    tmp_path: Path,
) -> None:
    model = _export_linear("proto")
    assert isinstance(model, onnx.ModelProto)
    weight, bias = _linear_parameter_names(model)
    annotate_parameters(
        model,
        {
            weight: ParameterMetadata(trainable=True, source_path="dense.kernel"),
            bias: ParameterMetadata(trainable=False),
        },
    )
    path = tmp_path / "consumer.onnx"
    onnx.save_model(model, path)
    reloaded = onnx.load_model(path)
    metadata = read_parameter_metadata(reloaded)
    before = {
        tensor.name: numpy_helper.to_array(tensor).copy()
        for tensor in reloaded.graph.initializer
    }
    trainable = {
        name for name, info in metadata.items() if info.trainable is True
    } & before.keys()
    frozen = {
        name for name, info in metadata.items() if info.trainable is False
    } & before.keys()
    unknown = before.keys() - metadata.keys()
    assert trainable == {weight}
    assert frozen == {bias}
    assert unknown  # Exporter-generated constants carry no inferred training role.
    delta = np.float32(0.125)
    for tensor in reloaded.graph.initializer:
        if tensor.name in trainable:
            tensor.CopyFrom(
                numpy_helper.from_array(before[tensor.name] + delta, name=tensor.name)
            )
        else:
            np.testing.assert_array_equal(
                numpy_helper.to_array(tensor), before[tensor.name]
            )
    onnx.checker.check_model(reloaded, full_check=True)
    features = np.arange(6, dtype=np.float32).reshape(2, 3) / 5
    original = _run(model, features)
    updated = _run(reloaded, features)
    expected = features @ (before[weight] + delta) + before[bias]
    np.testing.assert_allclose(updated, expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(updated, original)
    assert read_parameter_metadata(reloaded) == metadata


@pytest.mark.parametrize("tied", [True, False])
def test_annotations_follow_exported_identity_instead_of_equal_payloads(
    tied: bool,
) -> None:
    left = jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10
    right = left if tied else jnp.array(left, copy=True)

    def two_uses(features: jax.Array) -> jax.Array:
        return features @ left + jnp.tanh(features) @ right

    model = to_onnx(two_uses, [(2, 3)], opset=18)
    weights = [
        tensor for tensor in model.graph.initializer if tuple(tensor.dims) == (3, 2)
    ]
    assert len(weights) == (1 if tied else 2)
    first_name = weights[0].name
    annotate_parameters(
        model,
        {first_name: ParameterMetadata(trainable=True, source_path="shared.weight")},
    )
    if tied:
        consumers = [node for node in model.graph.node if first_name in node.input]
        assert len(consumers) == 2
        assert list(read_parameter_metadata(model)) == [first_name]
    else:
        np.testing.assert_array_equal(
            numpy_helper.to_array(weights[0]), numpy_helper.to_array(weights[1])
        )
        # Equal source paths remain informational and do not merge distinct values.
        annotate_parameters(
            model,
            {
                weights[1].name: ParameterMetadata(
                    trainable=False, source_path="shared.weight"
                )
            },
        )
        assert read_parameter_metadata(model) == {
            first_name: ParameterMetadata(trainable=True, source_path="shared.weight"),
            weights[1].name: ParameterMetadata(
                trainable=False, source_path="shared.weight"
            ),
        }


@pytest.mark.parametrize("name", ["missing", "projected", "prediction", "in_0"])
def test_annotation_rejects_unknown_intermediate_and_pre_rename_names_atomically(
    model: Model, name: str
) -> None:
    annotate_parameters(model, {"weight": ParameterMetadata(trainable=False)})
    before = _as_proto(model).SerializeToString()
    with pytest.raises(ValueError):
        annotate_parameters(
            model,
            {
                "weight": ParameterMetadata(trainable=True),
                name: ParameterMetadata(trainable=True),
            },
        )
    assert _as_proto(model).SerializeToString() == before


def test_annotation_uses_final_custom_input_name() -> None:
    model = to_onnx(
        lambda x: jnp.tanh(x),
        [(2, 3)],
        input_names=["caller.weight"],
        output_names=["caller.output"],
        opset=18,
    )
    annotate_parameters(model, {"caller.weight": ParameterMetadata(trainable=True)})
    assert read_parameter_metadata(model) == {
        "caller.weight": ParameterMetadata(trainable=True)
    }
    with pytest.raises(ValueError):
        annotate_parameters(model, {"in_0": ParameterMetadata(trainable=True)})


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[]",
        "null",
        '{"version":true,"parameters":{}}',
        '{"version":2,"parameters":{}}',
        '{"version":1.0,"parameters":{}}',
        '{"parameters":{}}',
        '{"version":1,"parameters":[]}',
        '{"version":1,"parameters":[],"extra":0}',
        '{"version":1,"parameters":{},"extra":0}',
        '{"version":1,"parameters":{"weight":{}}}',
        '{"version":1,"parameters":{"weight":null}}',
        '{"version":1,"parameters":{"weight":{"trainable":1}}}',
        '{"version":1,"parameters":{"weight":{"trainable":"true"}}}',
        '{"version":1,"parameters":{"weight":{"trainable":true,"source_path":null}}}',
        '{"version":1,"parameters":{"weight":{"trainable":true,"source_path":""}}}',
        '{"version":1,"parameters":{"weight":{"trainable":true,"source_path":12}}}',
        '{"version":1,"parameters":{"weight":{"trainable":true,"role":"weight"}}}',
        '{"version":1,"parameters":{"projected":{"trainable":true}}}',
        '{"version":1,"parameters":{"stale.weight":{"trainable":true}}}',
        '{"version":1,"version":1,"parameters":{}}',
        '{"version":1,"parameters":{"weight":{"trainable":true,"trainable":false}}}',
        '{"version":1,"parameters":{"weight":{"trainable":true},"weight":{"trainable":false}}}',
    ],
)
def test_malformed_or_stale_manifest_is_rejected_without_mutation(
    model: Model, payload: str
) -> None:
    _set_metadata(model, _METADATA_KEY, payload)
    before = _as_proto(model).SerializeToString()
    with pytest.raises(ValueError):
        read_parameter_metadata(model)
    with pytest.raises(ValueError):
        annotate_parameters(model, {"features": ParameterMetadata(trainable=False)})
    assert _as_proto(model).SerializeToString() == before


def test_duplicate_reserved_model_metadata_key_is_rejected() -> None:
    model = _small_model("proto")
    assert isinstance(model, onnx.ModelProto)
    payload = '{"version":1,"parameters":{}}'
    _set_metadata(model, _METADATA_KEY, payload)
    _set_metadata(model, _METADATA_KEY, payload)
    before = model.SerializeToString()
    with pytest.raises(ValueError):
        read_parameter_metadata(model)
    with pytest.raises(ValueError):
        annotate_parameters(model, {"weight": ParameterMetadata(trainable=True)})
    assert model.SerializeToString() == before


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"trainable": 1}, TypeError),
        ({"trainable": "yes"}, TypeError),
        ({"trainable": np.bool_(True)}, TypeError),
        ({"source_path": 1}, TypeError),
        ({"source_path": ""}, ValueError),
    ],
)
def test_parameter_metadata_rejects_invalid_field_types(
    kwargs: dict[str, Any], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        ParameterMetadata(**kwargs)


@pytest.mark.parametrize(
    "annotations,exception",
    [
        ([("weight", ParameterMetadata(trainable=True))], TypeError),
        ({"weight": {"trainable": True}}, TypeError),
        ({1: ParameterMetadata(trainable=True)}, TypeError),
        ({"": ParameterMetadata(trainable=True)}, ValueError),
    ],
)
def test_invalid_annotation_mapping_is_rejected_atomically(
    model: Model, annotations: Any, exception: type[Exception]
) -> None:
    before = _as_proto(model).SerializeToString()
    with pytest.raises(exception):
        annotate_parameters(model, annotations)
    assert _as_proto(model).SerializeToString() == before


def test_tensor_metadata_does_not_imply_parameter_annotations() -> None:
    model = _small_model("proto")
    assert isinstance(model, onnx.ModelProto)
    tensor = model.graph.initializer[0]
    tensor.metadata_props.add(key="trainable", value="true")
    tensor.metadata_props.add(key="source_path", value="tensor.weight")
    tensor_before = tensor.SerializeToString()
    assert read_parameter_metadata(model) == {}
    annotate_parameters(model, {"weight": ParameterMetadata(trainable=False)})
    assert tensor.SerializeToString() == tensor_before
    assert read_parameter_metadata(model) == {
        "weight": ParameterMetadata(trainable=False)
    }


def test_input_initializer_overlap_refers_to_one_parameter() -> None:
    model = _small_model("proto")
    assert isinstance(model, onnx.ModelProto)
    model.graph.input.append(
        helper.make_tensor_value_info("weight", TensorProto.FLOAT, [2, 2])
    )
    for representation in (model, ir.from_proto(model)):
        annotate_parameters(
            representation, {"weight": ParameterMetadata(trainable=True)}
        )
        assert read_parameter_metadata(representation) == {
            "weight": ParameterMetadata(trainable=True)
        }


def test_sparse_initializer_name_can_be_annotated() -> None:
    model = _small_model("proto")
    assert isinstance(model, onnx.ModelProto)
    model.graph.sparse_initializer.append(
        helper.make_sparse_tensor(
            numpy_helper.from_array(np.ones(2, dtype=np.float32), name="sparse.weight"),
            numpy_helper.from_array(np.asarray([[0, 0], [1, 1]], dtype=np.int64)),
            [2, 2],
        )
    )
    onnx.checker.check_model(model, full_check=True)
    annotate_parameters(model, {"sparse.weight": ParameterMetadata(trainable=True)})
    assert read_parameter_metadata(model) == {
        "sparse.weight": ParameterMetadata(trainable=True)
    }


def test_nested_scope_initializer_is_not_a_root_parameter() -> None:
    branch = helper.make_graph(
        [helper.make_node("Identity", ["branch.weight"], ["branch.output"])],
        "branch",
        [],
        [helper.make_tensor_value_info("branch.output", TensorProto.FLOAT, [2])],
        initializer=[
            numpy_helper.from_array(np.ones(2, dtype=np.float32), name="branch.weight")
        ],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If",
                ["condition"],
                ["prediction"],
                then_branch=branch,
                else_branch=branch,
            )
        ],
        "nested",
        [helper.make_tensor_value_info("condition", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    for representation in (model, ir.from_proto(model)):
        with pytest.raises(ValueError):
            annotate_parameters(
                representation, {"branch.weight": ParameterMetadata(trainable=True)}
            )


@pytest.mark.parametrize("duplicate", ["input", "dense", "dense_sparse", "node_output"])
def test_ambiguous_root_definitions_are_rejected_atomically(duplicate: str) -> None:
    model = _small_model("proto")
    assert isinstance(model, onnx.ModelProto)
    if duplicate == "input":
        model.graph.input.append(model.graph.input[0])
    elif duplicate == "dense":
        model.graph.initializer.append(model.graph.initializer[0])
    elif duplicate == "dense_sparse":
        model.graph.sparse_initializer.append(
            helper.make_sparse_tensor(
                numpy_helper.from_array(np.ones(2, dtype=np.float32), name="weight"),
                numpy_helper.from_array(np.asarray([[0, 0], [1, 1]], dtype=np.int64)),
                [2, 2],
            )
        )
    else:
        model.graph.node[0].output[0] = "weight"
    before = model.SerializeToString()
    with pytest.raises(ValueError):
        read_parameter_metadata(model)
    with pytest.raises(ValueError):
        annotate_parameters(model, {"weight": ParameterMetadata(trainable=True)})
    assert model.SerializeToString() == before


def test_annotation_preserves_external_data_references_and_existing_training_info() -> (
    None
):
    model = _small_model("proto")
    assert isinstance(model, onnx.ModelProto)
    tensor = model.graph.initializer[0]
    onnx.external_data_helper.set_external_data(
        tensor,
        location="caller-owned-weights.bin",
        offset=64,
        length=len(tensor.raw_data),
    )
    tensor.ClearField("raw_data")
    training = model.training_info.add()
    training.initialization.CopyFrom(helper.make_graph([], "initialization", [], []))
    training.algorithm.CopyFrom(helper.make_graph([], "algorithm", [], []))
    training.update_binding.add(key="weight", value="caller.updated_weight")
    model.doc_string = "A caller-prepared model with existing training information."
    _set_metadata(model, "application.note", "preserve me")
    before = _without_model_metadata(model)
    annotate_parameters(model, {"weight": ParameterMetadata(trainable=True)})
    assert read_parameter_metadata(model) == {
        "weight": ParameterMetadata(trainable=True)
    }
    assert _without_model_metadata(model) == before
    assert {entry.key: entry.value for entry in model.metadata_props}[
        "application.note"
    ] == "preserve me"


@pytest.mark.parametrize("unsupported", [None, object(), onnx.GraphProto()])
def test_unsupported_model_types_are_rejected(unsupported: Any) -> None:
    with pytest.raises(TypeError):
        read_parameter_metadata(unsupported)
    with pytest.raises(TypeError):
        annotate_parameters(unsupported, {})
