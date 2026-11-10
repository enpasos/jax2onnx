# tests/extra_tests/test_state_inputs_rope.py

"""Checks for the state-input walkthrough in scripts/state_inputs_rope.py."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import StateInput, to_onnx
from scripts import state_inputs_rope as example
from scripts.state_inputs_rope import StoredState

REPOSITORY = Path(__file__).resolve().parents[2]
WALKTHROUGH = REPOSITORY / "docs/user_guide/state_inputs_walkthrough.md"


@pytest.fixture(scope="module")
def exported():
    model = example.SmallAttention(key=jax.random.key(0))
    x = jnp.linspace(-0.8, 0.9, 12, dtype=jnp.float32).reshape(3, 4)
    proto = example.export_model(model, x)
    return model, x, example.read_state(proto), proto, example.runtime(proto)


def _changed(state):
    return {
        entry.name: np.asarray(entry.value) * np.float32(0.6 + 0.1 * index)
        + np.linspace(-0.3, 0.4, entry.value.size, dtype=np.float32).reshape(
            entry.value.shape
        )
        + np.float32(0.2 * (index + 1))
        for index, entry in enumerate(state)
    }


def _reference(model, state, arrays):
    return example.reconstruct(
        model,
        tuple(
            StoredState(entry.name, arrays[entry.name], entry.metadata)
            for entry in state
        ),
    )


def _assert_stored_policy(state):
    # These expectations must not come from the writer's declarations.
    assert {
        entry.name: (
            entry.metadata["pkg.example.path"],
            entry.metadata["pkg.example.trainable"],
        )
        for entry in state
    } == {
        "shared_weight": ("shared_weight", "true"),
        "independent_weight": ("independent_weight", "true"),
        "rope_rotations": ("rope.rotations", "false"),
        "shared_frozen": ("q_projection.matrix;k_projection.matrix", "false"),
    }


def _tensor_signature(tensor):
    value = onnx.numpy_helper.to_array(tensor)
    return value.dtype.str, value.shape, value.tobytes()


def _node_tensors(nodes):
    for node in nodes:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from _graph_tensors(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph in attribute.graphs:
                    yield from _graph_tensors(graph)


def _graph_tensors(graph):
    yield from graph.initializer
    yield from _node_tensors(graph.node)


def _outside_inventory(proto):
    # A multiset of actual tensor payloads, not a source provenance inference.
    tensors = [
        tensor
        for tensor in proto.graph.initializer
        if tensor.name not in example.STATE_NAMES
    ]
    tensors.extend(_node_tensors(proto.graph.node))
    for function in proto.functions:
        tensors.extend(_node_tensors(function.node))
    return Counter(_tensor_signature(tensor) for tensor in tensors)


def _ordinary(model, x):
    return to_onnx(
        model, inputs=[jax.ShapeDtypeStruct(x.shape, x.dtype)], input_names=["x"]
    )


def test_defaults_inventory_and_shared_function_edges(exported):
    model, x, state, proto, session = exported
    onnx.checker.check_model(proto, full_check=True)
    _assert_stored_policy(state)
    actual = session.run(None, {"x": np.asarray(x)})[0]
    assert actual.dtype == np.float32 and actual.shape == x.shape
    np.testing.assert_allclose(actual, model(x), rtol=2e-5, atol=2e-6)
    assert [value.name for value in proto.graph.input] == ["x", *example.STATE_NAMES]
    assert [value.name for value in session.get_inputs()] == ["x"]
    assert {value.name for value in session.get_overridable_initializers()} == set(
        example.STATE_NAMES
    )
    for entry in state:
        assert sum(t.name == entry.name for t in proto.graph.initializer) == 1
        tensor = next(t for t in proto.graph.initializer if t.name == entry.name)
        assert _outside_inventory(proto)[_tensor_signature(tensor)] == 0
    np.testing.assert_array_equal(state[0].value, state[1].value)
    functions = [f for f in proto.functions if f.name == "project"]
    assert len(functions) == 1
    function = functions[0]
    calls = [n for n in proto.graph.node if n.domain == function.domain]
    assert len(calls) == 2
    assert all(node.input[1] == "shared_weight" for node in calls)
    assert any(function.input[1] in n.input for n in function.node)


def _op_types(proto):
    types = Counter(node.op_type for node in proto.graph.node)
    for function in proto.functions:
        types.update(f"{function.name}.{node.op_type}" for node in function.node)
    return types


def _float_weights(proto):
    return Counter(
        _tensor_signature(tensor)
        for tensor in proto.graph.initializer
        if tensor.data_type == onnx.TensorProto.FLOAT
        and onnx.numpy_helper.to_array(tensor).size > 1
    )


def test_state_export_keeps_the_ordinary_computation(exported):
    # The walkthrough states that only the weight interface differs.
    model, x, _, proto, session = exported
    ordinary = _ordinary(model, x)
    assert [value.name for value in ordinary.graph.input] == ["x"]
    assert not set(example.STATE_NAMES) & {t.name for t in ordinary.graph.initializer}
    assert _op_types(ordinary) == _op_types(proto)
    assert _float_weights(ordinary) == _float_weights(proto)
    np.testing.assert_allclose(
        example.runtime(ordinary).run(None, {"x": np.asarray(x)})[0],
        session.run(None, {"x": np.asarray(x)})[0],
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("selected", [*example.STATE_NAMES, "all"])
def test_single_and_joint_overrides_match_reconstructed_jax(exported, selected):
    model, x, state, _, session = exported
    changed = _changed(state)
    overrides = changed if selected == "all" else {selected: changed[selected]}
    arrays = {entry.name: entry.value for entry in state} | overrides
    actual = session.run(None, {"x": np.asarray(x), **overrides})[0]
    np.testing.assert_allclose(
        actual, _reference(model, state, arrays)(x), rtol=2e-5, atol=2e-6
    )
    assert not np.allclose(actual, model(x), rtol=1e-6, atol=1e-7)
    if selected == "independent_weight":
        assert arrays["shared_weight"] is state[0].value
        assert not np.array_equal(arrays["shared_weight"], arrays["independent_weight"])


def test_initializer_edit_uses_fresh_session_and_updates_shared_consumers(exported):
    model, x, state, proto, original_session = exported
    edited = onnx.ModelProto()
    edited.CopyFrom(proto)
    changed = _changed(state)["shared_weight"]
    tensor = next(t for t in edited.graph.initializer if t.name == "shared_weight")
    replacement = onnx.numpy_helper.from_array(changed, name=tensor.name)
    replacement.metadata_props.extend(tensor.metadata_props)
    tensor.CopyFrom(replacement)
    arrays = {entry.name: entry.value for entry in state} | {"shared_weight": changed}
    actual = example.runtime(edited).run(None, {"x": np.asarray(x)})[0]
    np.testing.assert_allclose(
        actual, _reference(model, state, arrays)(x), rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        original_session.run(None, {"x": np.asarray(x)})[0],
        model(x),
        rtol=2e-5,
        atol=2e-6,
    )


def test_changed_state_leaves_all_other_tensors_unchanged(exported):
    model, x, state, proto, _ = exported
    changed = _reference(model, state, _changed(state))
    assert _outside_inventory(proto) == _outside_inventory(
        example.export_model(changed, x)
    )


def test_tied_arrays_are_one_input_and_copies_stay_separate(exported):
    model, x, _, proto, _ = exported
    assert model.q_projection.matrix is model.k_projection.matrix
    assert model.shared_weight is not model.independent_weight
    np.testing.assert_array_equal(model.shared_weight, model.independent_weight)
    untied = eqx.tree_at(
        lambda m: m.k_projection.matrix, model, jnp.array(model.k_projection.matrix)
    )
    with pytest.raises(ValueError, match="one array object"):
        example.state_inputs(untied)


@pytest.mark.parametrize("external_data", [False, True])
def test_plain_and_external_reload_preserve_exact_state(
    exported, tmp_path, external_data
):
    model, x, _, proto, _ = exported
    before = proto.SerializeToString(deterministic=True)
    path = tmp_path / "attention.onnx"
    restored = example.save_reload(proto, path, external_data=external_data)
    assert proto.SerializeToString(deterministic=True) == before
    _assert_stored_policy(example.read_state(restored))
    stored = onnx.load(path, load_external_data=False)
    for tensor in stored.graph.initializer:
        if tensor.name in example.STATE_NAMES:
            assert (tensor.data_location == onnx.TensorProto.EXTERNAL) == external_data
    assert (tmp_path / "attention.onnx.data").exists() == external_data
    np.testing.assert_allclose(
        example.runtime(restored).run(None, {"x": np.asarray(x)})[0],
        model(x),
        rtol=2e-5,
        atol=2e-6,
    )


def test_saving_rejects_a_model_without_the_state_inputs_before_writing(
    exported, tmp_path
):
    model, x, _, _, _ = exported
    with pytest.raises(ValueError, match="four state inputs"):
        example.save_reload(
            _ordinary(model, x), tmp_path / "invalid.onnx", external_data=True
        )
    assert not list(tmp_path.iterdir())


def test_model_rebuilt_from_the_file_exports_to_the_same_file(exported):
    model, x, state, proto, _ = exported
    rebuilt = example.reconstruct(model, state)
    assert rebuilt.q_projection.matrix is rebuilt.k_projection.matrix
    np.testing.assert_array_equal(rebuilt(x), model(x))
    assert example.export_model(rebuilt, x).SerializeToString(
        deterministic=True
    ) == proto.SerializeToString(deterministic=True)


def test_stored_labels_exclude_both_frozen_arrays_from_adamw(exported):
    model, x, state, _, _ = exported
    rebuilt = example.reconstruct(model, state)
    updated, loss = example.adamw_step(model, state, x, jnp.zeros_like(x))
    assert np.isfinite(loss) and loss > 0
    for before, after in zip(state, updated):
        same_bits = (
            np.asarray(before.value).tobytes() == np.asarray(after.value).tobytes()
        )
        assert same_bits == (before.name in {"rope_rotations", "shared_frozen"})
    # This is a real stop_gradient path, but the reader must also prevent decay.
    gradients = jax.grad(lambda rotations: jnp.sum(example.RoPE(rotations)(x)))(
        model.rope.rotations
    )
    np.testing.assert_array_equal(gradients, np.zeros_like(model.rope.rotations))
    np.testing.assert_array_equal(updated[2].value, rebuilt.rope.rotations)
    np.testing.assert_array_equal(updated[3].value, rebuilt.q_projection.matrix)
    np.testing.assert_array_equal(updated[3].value, rebuilt.k_projection.matrix)


def test_reconstruction_follows_the_stored_labels(exported):
    model, x, state, _, session = exported
    changed = _changed(state)
    swapped_paths = {
        "shared_weight": "independent_weight",
        "independent_weight": "shared_weight",
    }
    mislabeled = tuple(
        StoredState(
            entry.name,
            changed[entry.name],
            {
                **entry.metadata,
                "pkg.example.path": swapped_paths.get(
                    entry.name, entry.metadata["pkg.example.path"]
                ),
            },
        )
        for entry in state
    )
    rebuilt = example.reconstruct(model, mislabeled)
    np.testing.assert_array_equal(rebuilt.shared_weight, changed["independent_weight"])
    np.testing.assert_array_equal(rebuilt.independent_weight, changed["shared_weight"])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            session.run(None, {"x": np.asarray(x), **changed})[0],
            rebuilt(x),
            rtol=2e-5,
            atol=2e-6,
        )


@pytest.mark.parametrize(
    "invalid", ["missing_entry", "duplicate", "unknown", "missing_path"]
)
@pytest.mark.parametrize("consumer", ["reconstruct", "adamw"])
def test_reconstruction_requires_a_complete_path_bijection(exported, invalid, consumer):
    model, x, state, _, _ = exported
    if invalid == "missing_entry":
        incomplete = state[:-1]
    else:
        metadata = dict(state[-1].metadata)
        if invalid == "missing_path":
            del metadata["pkg.example.path"]
        else:
            metadata["pkg.example.path"] = (
                state[0].metadata["pkg.example.path"]
                if invalid == "duplicate"
                else "unknown.path"
            )
        incomplete = (
            *state[:-1],
            StoredState(state[-1].name, state[-1].value, metadata),
        )
    with pytest.raises(ValueError, match="each known path exactly once"):
        if consumer == "reconstruct":
            example.reconstruct(model, incomplete)
        else:
            example.adamw_step(model, incomplete, x, jnp.zeros_like(x))


@pytest.mark.parametrize("opset", [17, 23])
@pytest.mark.parametrize("external_data", [False, True])
def test_demo_checks_stored_policy_and_independent_reference(
    tmp_path, external_data, opset
):
    report = example.run_demo(tmp_path, external_data=external_data, opset=opset)
    assert report["opset"] == opset
    assert report["state_inputs"] == 4
    assert report["frozen_state_unchanged"] is True
    assert report["override_max_absolute_error"] < 2e-6
    assert (tmp_path / "report.json").exists()


@pytest.mark.parametrize("mutation", ["swapped_paths", "trainable_rotations"])
def test_demo_rejects_incorrect_stored_policy(tmp_path, monkeypatch, mutation):
    original_state_inputs = example.state_inputs

    def wrong_policy(model):
        changed = []
        for entry in original_state_inputs(model):
            metadata = dict(entry.metadata)
            if mutation == "swapped_paths" and entry.name in {
                "shared_weight",
                "independent_weight",
            }:
                metadata["pkg.example.path"] = (
                    "independent_weight"
                    if entry.name == "shared_weight"
                    else "shared_weight"
                )
            if mutation == "trainable_rotations" and entry.name == "rope_rotations":
                metadata["pkg.example.trainable"] = "true"
            changed.append(StateInput(entry.array, entry.name, metadata))
        return changed

    monkeypatch.setattr(example, "state_inputs", wrong_policy)
    with pytest.raises(AssertionError, match="stored state policy"):
        example.run_demo(tmp_path)
    assert not (tmp_path / "report.json").exists()


def test_walkthrough_page_runs_as_one_session(tmp_path, monkeypatch, capsys):
    # The page promises that its Python blocks form one sequential session.
    blocks = re.findall(r"^```python\n(.*?)^```", WALKTHROUGH.read_text(), re.M | re.S)
    assert len(blocks) >= 8
    source = "\n".join(blocks)
    assert source.count("/tmp/state-inputs-walkthrough") == 1
    monkeypatch.chdir(tmp_path)
    exec(
        compile(
            source.replace("/tmp/state-inputs-walkthrough", str(tmp_path)),
            str(WALKTHROUGH),
            "exec",
        ),
        {"__name__": "walkthrough"},
    )
    assert (tmp_path / "attention.onnx").is_file()
    assert (tmp_path / "attention-trained.onnx").is_file()
    printed = capsys.readouterr().out
    assert "['x', 'shared_weight', 'independent_weight'" in printed
