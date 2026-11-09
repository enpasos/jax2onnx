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

from scripts import explicit_state_rope as example
from scripts.explicit_state_export import export_with_state


@pytest.fixture(scope="module")
def exported():
    model = example.SmallAttention(key=jax.random.key(0))
    x = jnp.linspace(-0.8, 0.9, 12, dtype=jnp.float32).reshape(3, 4)
    slots = example.state_slots(model)
    exported_model = example.export_model(model, x)
    proto = ir.to_proto(exported_model)
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


def _reference(model, state):
    # Independent, explicit reconstruction of this known architecture, without
    # the export adapter's partition/forward closure or any ONNX-to-JAX importer.
    return eqx.tree_at(
        lambda m: (
            m.shared_weight,
            m.independent_weight,
            m.rope.rotations,
            m.q_projection.matrix,
            m.k_projection.matrix,
        ),
        model,
        replace=tuple(
            jnp.asarray(state[name]) for name in (*example.SLOT_NAMES, "shared_frozen")
        ),
    )


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
    assert any(
        n.op_type in {"MatMul", "Gemm"} and "shared_weight" in n.input
        for n in proto.graph.node
    )
    assert any(
        n.op_type == "Reshape" and n.input[0] == "shared_weight"
        for n in proto.graph.node
    )
    frozen_values = {"shared_frozen"}
    for node in proto.graph.node:
        if node.op_type == "Identity" and node.input[0] in frozen_values:
            frozen_values.update(node.output)
    assert (
        sum(
            node.op_type in {"MatMul", "Gemm"}
            and any(name in frozen_values for name in node.input)
            for node in proto.graph.node
        )
        == 2
    )


@pytest.mark.parametrize("selected", [*example.SLOT_NAMES, "all"])
def test_independent_and_all_slot_overrides_match_reconstructed_jax(exported, selected):
    model, x, slots, _, _, session = exported
    changed = _changed(slots)
    overrides = changed if selected == "all" else {selected: changed[selected]}
    state = {slot.name: slot.value for slot in slots} | overrides
    actual = session.run(None, {"x": np.asarray(x), **overrides})[0]
    np.testing.assert_allclose(
        actual, _reference(model, state)(x), rtol=2e-5, atol=2e-6
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
        actual, _reference(model, state)(x), rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        original_session.run(None, {"x": np.asarray(x)})[0],
        model(x),
        rtol=2e-5,
        atol=2e-6,
    )


def test_all_changed_sources_leave_outside_tensor_multiset_unchanged(exported):
    model, x, slots, _, proto, _ = exported
    changed = _reference(model, _changed(slots))
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
            _reference(model, changed)(x),
            rtol=2e-5,
            atol=2e-6,
        )
    assert _outside_inventory(bad_proto) != _outside_inventory(
        bad_export(_reference(model, changed))
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
    rebuilt = _reference(model, {slot.name: slot.value for slot in imported})
    np.testing.assert_array_equal(rebuilt(x), model(x))
    updated, loss = example.adamw_step(model, imported, x, jnp.zeros_like(x))
    assert np.isfinite(loss) and loss > 0
    for before, after in zip(imported, updated):
        same_bits = (
            np.asarray(before.value).tobytes() == np.asarray(after.value).tobytes()
        )
        assert same_bits == (before.metadata[example.TRAINABLE] == "false")
    # This is a real stop_gradient path, but the importer must also prevent decay.
    gradients = jax.grad(lambda rotations: jnp.sum(example.RoPE(rotations)(x)))(
        model.rope.rotations
    )
    np.testing.assert_array_equal(gradients, np.zeros_like(model.rope.rotations))
    np.testing.assert_array_equal(updated[3].value, rebuilt.q_projection.matrix)
    np.testing.assert_array_equal(updated[3].value, rebuilt.k_projection.matrix)


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
