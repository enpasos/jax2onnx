# tests/extra_tests/test_explicit_state_rope.py

"""Integration checks for the known Equinox architecture, including SP-09 leaks."""

from __future__ import annotations

from collections import Counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import StateInput, to_onnx
from scripts import explicit_state_rope as example
from scripts.explicit_state_export import StateSlot, export_with_state


@pytest.fixture(scope="module")
def exported():
    model = example.SmallAttention(key=jax.random.key(0))
    x = jnp.linspace(-0.8, 0.9, 12, dtype=jnp.float32).reshape(3, 4)
    exported_model = example.export_model(model, x)
    proto = ir.to_proto(exported_model)
    slots = example.read_slots(proto)
    return model, x, slots, exported_model, proto, example.runtime(proto)


def _changed(slots):
    return {
        slot.name: np.asarray(slot.value) * np.float32(0.6 + 0.1 * index)
        + np.linspace(-0.3, 0.4, slot.value.size, dtype=np.float32).reshape(
            slot.value.shape
        )
        + np.float32(0.2 * (index + 1))
        for index, slot in enumerate(slots)
    }


def _reference(model, slots, state):
    return example.reconstruct(
        model,
        tuple(StateSlot(slot.name, state[slot.name], slot.metadata) for slot in slots),
    )


def _assert_imported_policy(slots):
    # These expectations must not come from the writer's declarations.
    assert {
        slot.name: (
            slot.metadata["pkg.example.path"],
            slot.metadata["pkg.example.trainable"],
        )
        for slot in slots
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
        if tensor.name not in example.SLOT_NAMES
    ]
    tensors.extend(_node_tensors(proto.graph.node))
    for function in proto.functions:
        tensors.extend(_node_tensors(function.node))
    return Counter(_tensor_signature(tensor) for tensor in tensors)


def test_defaults_inventory_and_shared_function_edges(exported):
    model, x, slots, exported_model, proto, session = exported
    onnx.checker.check_model(proto, full_check=True)
    _assert_imported_policy(example.read_slots(proto))
    actual = session.run(None, {"x": np.asarray(x)})[0]
    assert actual.dtype == np.float32 and actual.shape == x.shape
    np.testing.assert_allclose(actual, model(x), rtol=2e-5, atol=2e-6)
    assert [value.name for value in session.get_inputs()] == ["x"]
    assert {value.name for value in session.get_overridable_initializers()} == set(
        example.SLOT_NAMES
    )
    inputs = {value.name: value for value in exported_model.graph.inputs}
    assert len({id(inputs[name]) for name in example.SLOT_NAMES}) == 4
    for slot in slots:
        assert exported_model.graph.initializers[slot.name] is inputs[slot.name]
        assert sum(t.name == slot.name for t in proto.graph.initializer) == 1
        tensor = next(t for t in proto.graph.initializer if t.name == slot.name)
        assert _outside_inventory(proto)[_tensor_signature(tensor)] == 0
    np.testing.assert_array_equal(slots[0].value, slots[1].value)
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


def test_explicit_export_keeps_the_ordinary_computation(exported):
    # The user guide states that only the weight interface differs.
    model, x, _, _, proto, session = exported
    ordinary = ir.to_proto(
        to_onnx(
            model,
            inputs=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
            input_names=["x"],
            return_mode="ir",
        )
    )
    assert [value.name for value in ordinary.graph.input] == ["x"]
    assert not set(example.SLOT_NAMES) & {t.name for t in ordinary.graph.initializer}
    assert _op_types(ordinary) == _op_types(proto)
    assert _float_weights(ordinary) == _float_weights(proto)
    np.testing.assert_allclose(
        example.runtime(ordinary).run(None, {"x": np.asarray(x)})[0],
        session.run(None, {"x": np.asarray(x)})[0],
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("selected", [*example.SLOT_NAMES, "all"])
def test_independent_and_all_slot_overrides_match_reconstructed_jax(exported, selected):
    model, x, slots, _, _, session = exported
    changed = _changed(slots)
    overrides = changed if selected == "all" else {selected: changed[selected]}
    state = {slot.name: slot.value for slot in slots} | overrides
    actual = session.run(None, {"x": np.asarray(x), **overrides})[0]
    np.testing.assert_allclose(
        actual, _reference(model, slots, state)(x), rtol=2e-5, atol=2e-6
    )
    assert not np.allclose(actual, model(x), rtol=1e-6, atol=1e-7)
    if selected == "independent_weight":
        assert state["shared_weight"] is slots[0].value
        assert not np.array_equal(state["shared_weight"], state["independent_weight"])


def test_initializer_edit_uses_fresh_session_and_updates_shared_consumers(exported):
    model, x, slots, exported_model, _, original_session = exported
    proto = ir.to_proto(exported_model)
    changed = _changed(slots)["shared_weight"]
    tensor = next(t for t in proto.graph.initializer if t.name == "shared_weight")
    replacement = onnx.numpy_helper.from_array(changed, name=tensor.name)
    replacement.metadata_props.extend(tensor.metadata_props)
    tensor.CopyFrom(replacement)
    state = {slot.name: slot.value for slot in slots} | {"shared_weight": changed}
    actual = example.runtime(proto).run(None, {"x": np.asarray(x)})[0]
    np.testing.assert_allclose(
        actual, _reference(model, slots, state)(x), rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        original_session.run(None, {"x": np.asarray(x)})[0],
        model(x),
        rtol=2e-5,
        atol=2e-6,
    )


def test_all_changed_sources_leave_outside_tensor_multiset_unchanged(exported):
    model, x, slots, _, proto, _ = exported
    changed = _reference(model, slots, _changed(slots))
    changed_proto = ir.to_proto(example.export_model(changed, x))
    assert _outside_inventory(proto) == _outside_inventory(changed_proto)


def test_partial_closed_state_leak_is_detected_by_both_sp09_checks(exported):
    model, x, slots, _, _, _ = exported

    def bad_export(source):
        forward = example.explicit_forward(source)

        def partial_leak(x, weight, independent, rotations, frozen):
            # Half the direct residual still uses the closed source. Every slot
            # has consumers, and the default forward remains exactly correct.
            return forward(x, weight, independent, rotations, frozen) + 0.05 * (
                x @ source.shared_weight - x @ weight
            )

        return ir.to_proto(
            export_with_state(
                partial_leak,
                [jax.ShapeDtypeStruct(x.shape, x.dtype)],
                example.state_slots(source),
                data_input_names=["x"],
            )
        )

    bad_proto = bad_export(model)  # Inventory checks alone accept this adapter.
    session = example.runtime(bad_proto)
    np.testing.assert_allclose(
        session.run(None, {"x": np.asarray(x)})[0], model(x), rtol=2e-5, atol=2e-6
    )
    changed = _changed(slots)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            session.run(None, {"x": np.asarray(x), **changed})[0],
            _reference(model, slots, changed)(x),
            rtol=2e-5,
            atol=2e-6,
        )
    assert _outside_inventory(bad_proto) != _outside_inventory(
        bad_export(_reference(model, slots, changed))
    )


@pytest.mark.parametrize("external_data", [False, True])
def test_plain_and_external_reload_preserve_exact_state(
    exported, tmp_path, external_data
):
    model, x, slots, exported_model, _, _ = exported
    path = tmp_path / "attention.onnx"
    restored = example.save_reload(
        exported_model, slots, path, external_data=external_data
    )
    _assert_imported_policy(example.read_slots(restored))
    stored = onnx.load(path, load_external_data=False)
    for tensor in stored.graph.initializer:
        if tensor.name in example.SLOT_NAMES:
            assert (tensor.data_location == onnx.TensorProto.EXTERNAL) == external_data
    assert (tmp_path / "attention.onnx.data").exists() == external_data
    np.testing.assert_allclose(
        example.runtime(restored).run(None, {"x": np.asarray(x)})[0],
        model(x),
        rtol=2e-5,
        atol=2e-6,
    )


def test_imported_metadata_excludes_both_frozen_arrays_from_adamw(exported):
    model, x, _, _, proto, _ = exported
    imported = example.read_slots(proto)
    _assert_imported_policy(imported)
    rebuilt = example.reconstruct(model, imported)
    np.testing.assert_array_equal(rebuilt(x), model(x))
    updated, loss = example.adamw_step(model, imported, x, jnp.zeros_like(x))
    assert np.isfinite(loss) and loss > 0
    for before, after in zip(imported, updated):
        same_bits = (
            np.asarray(before.value).tobytes() == np.asarray(after.value).tobytes()
        )
        assert same_bits == (before.name in {"rope_rotations", "shared_frozen"})
    # This is a real stop_gradient path, but the importer must also prevent decay.
    gradients = jax.grad(lambda rotations: jnp.sum(example.RoPE(rotations)(x)))(
        model.rope.rotations
    )
    np.testing.assert_array_equal(gradients, np.zeros_like(model.rope.rotations))
    np.testing.assert_array_equal(updated[2].value, rebuilt.rope.rotations)
    np.testing.assert_array_equal(updated[3].value, rebuilt.q_projection.matrix)
    np.testing.assert_array_equal(updated[3].value, rebuilt.k_projection.matrix)


def test_reconstruction_resolves_imported_labels_for_observable_overrides(exported):
    model, x, slots, _, _, session = exported
    changed = _changed(slots)
    swapped_paths = {
        "shared_weight": "independent_weight",
        "independent_weight": "shared_weight",
    }
    mislabeled = tuple(
        StateSlot(
            slot.name,
            changed[slot.name],
            {
                **slot.metadata,
                "pkg.example.path": swapped_paths.get(
                    slot.name, slot.metadata["pkg.example.path"]
                ),
            },
        )
        for slot in slots
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
    "invalid", ["missing_slot", "duplicate", "unknown", "missing_path"]
)
@pytest.mark.parametrize("consumer", ["reconstruct", "adamw"])
def test_reconstruction_requires_a_complete_path_bijection(exported, invalid, consumer):
    model, x, slots, _, _, _ = exported
    if invalid == "missing_slot":
        state = slots[:-1]
    else:
        metadata = dict(slots[-1].metadata)
        if invalid == "missing_path":
            del metadata["pkg.example.path"]
        else:
            metadata["pkg.example.path"] = (
                slots[0].metadata["pkg.example.path"]
                if invalid == "duplicate"
                else "unknown.path"
            )
        state = (*slots[:-1], StateSlot(slots[-1].name, slots[-1].value, metadata))
    with pytest.raises(ValueError, match="each known path exactly once"):
        if consumer == "reconstruct":
            example.reconstruct(model, state)
        else:
            example.adamw_step(model, state, x, jnp.zeros_like(x))


@pytest.mark.parametrize("opset", [17, 23])
@pytest.mark.parametrize("external_data", [False, True])
def test_demo_checks_serialized_policy_and_independent_reference(
    tmp_path, monkeypatch, external_data, opset
):
    original_export = example.export_with_state

    def export_at_selected_opset(*args, **kwargs):
        return original_export(*args, **kwargs, opset=opset)

    monkeypatch.setattr(example, "export_with_state", export_at_selected_opset)
    report = example.run_demo(tmp_path, external_data=external_data)
    assert report["opset"] == opset
    assert report["frozen_slots_unchanged"] is True
    assert report["override_max_absolute_error"] < 2e-6
    assert (tmp_path / "report.json").exists()


@pytest.mark.parametrize("mutation", ["swapped_paths", "trainable_rotations"])
def test_demo_rejects_incorrect_serialized_policy(tmp_path, monkeypatch, mutation):
    original_slots = example.state_slots

    def wrong_policy(model):
        slots = original_slots(model)
        changed = []
        for slot in slots:
            metadata = dict(slot.metadata)
            if mutation == "swapped_paths" and slot.name in {
                "shared_weight",
                "independent_weight",
            }:
                metadata["pkg.example.path"] = (
                    "independent_weight"
                    if slot.name == "shared_weight"
                    else "shared_weight"
                )
            if mutation == "trainable_rotations" and slot.name == "rope_rotations":
                metadata["pkg.example.trainable"] = "true"
            changed.append(StateSlot(slot.name, slot.value, metadata))
        return tuple(changed)

    monkeypatch.setattr(example, "state_slots", wrong_policy)
    with pytest.raises(AssertionError, match="serialized state policy"):
        example.run_demo(tmp_path)
    assert not (tmp_path / "report.json").exists()


def test_demo_rejects_partial_closed_state_leak(tmp_path, monkeypatch):
    original_forward = example.explicit_forward

    def leaking_forward(model):
        forward = original_forward(model)

        def partial_leak(x, weight, independent, rotations, frozen):
            return forward(x, weight, independent, rotations, frozen) + 0.05 * (
                x @ model.shared_weight - x @ weight
            )

        return partial_leak

    monkeypatch.setattr(example, "explicit_forward", leaking_forward)
    with pytest.raises(AssertionError):
        example.run_demo(tmp_path)
    assert not (tmp_path / "report.json").exists()


def test_declared_frozen_alias_policy_rejects_divergent_source_branches(exported):
    model = exported[0]
    divergent = eqx.tree_at(
        lambda m: m.k_projection.matrix, model, model.k_projection.matrix + 1
    )
    with pytest.raises(ValueError, match="declared shared frozen"):
        example.state_slots(divergent)


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "reordered"])
def test_saving_rejects_incomplete_state_references_before_writing(
    exported, tmp_path, invalid
):
    _, _, slots, model, _, _ = exported
    references = {
        "missing": slots[:-1],
        "duplicate": (slots[0], slots[0], *slots[2:]),
        "reordered": tuple(reversed(slots)),
    }[invalid]
    with pytest.raises(ValueError, match="four declared state slots"):
        example.save_reload(
            model, references, tmp_path / "invalid.onnx", external_data=True
        )
    assert not list(tmp_path.iterdir())


def test_to_onnx_state_reproduces_the_recipe_file(exported):
    # The public API automates this recipe: the same file, byte for byte.
    model, x, _, _, proto, _ = exported
    state = [
        StateInput(slot.value, slot.name, slot.metadata)
        for slot in example.state_slots(model)
    ]
    api = to_onnx(
        model,
        inputs=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
        input_names=["x"],
        state=state,
    )
    assert api.SerializeToString(deterministic=True) == proto.SerializeToString(
        deterministic=True
    )
