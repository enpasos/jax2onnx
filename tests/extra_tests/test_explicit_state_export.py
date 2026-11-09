# tests/extra_tests/test_explicit_state_export.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function
from scripts import explicit_state_export as example
from scripts.explicit_state_export import StateSlot, export_with_state


def _session(model: ir.Model) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(
        ir.to_proto(model).SerializeToString(),
        options,
        providers=["CPUExecutionProvider"],
    )


@pytest.mark.parametrize("same_array_object", [False, True])
@pytest.mark.parametrize("opset", [13, 17, 23])
def test_defaults_and_independent_overrides_of_equal_slots(
    same_array_object: bool,
    opset: int,
) -> None:
    first = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    second = first if same_array_object else first.copy()

    def forward(x: jax.Array, left: jax.Array, right: jax.Array) -> jax.Array:
        return x @ left + 2 * (x @ right)

    model = export_with_state(
        forward,
        [jax.ShapeDtypeStruct((1, 2), np.float32)],
        [
            StateSlot("left", first, {"path": "layer/left"}),
            StateSlot("right", second, {}),
        ],
        data_input_names=["x"],
        opset=opset,
    )
    assert model.graph.opset_imports[""] == opset
    values = {value.name: value for value in model.graph.inputs}
    assert set(values) == {"x", "left", "right"}
    assert values["left"] is not values["right"]
    assert model.graph.initializers["left"] is values["left"]
    assert model.graph.initializers["right"] is values["right"]
    assert values["left"].const_value is not values["right"].const_value
    assert not np.shares_memory(
        values["left"].const_value.numpy(), values["right"].const_value.numpy()
    )
    session = _session(model)
    assert [value.name for value in session.get_inputs()] == ["x"]
    assert {value.name for value in session.get_overridable_initializers()} == {
        "left",
        "right",
    }
    x = np.array([[2.0, -1.0]], dtype=np.float32)
    replacement = np.array([[0.0, 1.0], [2.0, -3.0]], dtype=np.float32)
    for overrides, left, right in [
        ({}, first, second),
        ({"left": replacement}, replacement, second),
        ({"right": replacement}, first, replacement),
    ]:
        actual = session.run(None, {"x": x, **overrides})[0]
        expected = forward(jnp.asarray(x), jnp.asarray(left), jnp.asarray(right))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_symbolic_data_shape_remains_an_ordinary_dynamic_input() -> None:
    spec = jax.ShapeDtypeStruct(jax.export.symbolic_shape("B, 2"), np.float32)
    weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    model = export_with_state(
        lambda x, weight: x @ weight,
        [spec],
        [StateSlot("weight", weight, {})],
        data_input_names=["x"],
    )
    session = _session(model)
    assert session.get_inputs()[0].shape == ["B", 2]
    for batch in [1, 3]:
        x = np.arange(batch * 2, dtype=np.float32).reshape(batch, 2)
        actual = session.run(None, {"x": x})[0]
        np.testing.assert_array_equal(actual, jnp.asarray(x) @ jnp.asarray(weight))


@pytest.mark.parametrize("external", [False, True])
def test_owned_noncontiguous_bits_and_tensor_metadata_roundtrip(
    external: bool, tmp_path: Path
) -> None:
    backing = np.array(
        [0, 0x80000000, 0x7FC00011, 0xFFC00022, 0x3F800000, 0xBF800000],
        dtype=np.uint32,
    ).view(np.float32)
    source = backing.reshape(2, 3).T
    assert not source.flags.c_contiguous
    payload = source.tobytes(order="C")
    properties = {
        "pkg.example.path": "blocks/π/weight",
        "opaque": "",
        "trainable": "false",
    }
    expected_properties = properties.copy()
    model = export_with_state(
        lambda x, state: x + state,
        [jax.ShapeDtypeStruct(source.shape, np.float32)],
        [StateSlot("state", source, properties)],
        data_input_names=["x"],
    )
    backing[:] = 7.0
    properties["opaque"] = "changed"
    value = model.graph.initializers["state"]
    tensor = value.const_value
    assert tensor is not None
    assert tensor.numpy().flags.c_contiguous
    assert not np.shares_memory(tensor.numpy(), source)
    assert tensor.numpy().tobytes(order="C") == payload
    assert dict(tensor.metadata_props) == expected_properties
    assert not value.metadata_props
    path = tmp_path / "model.onnx"
    # save_model mutates the proto for external data, so always give it a fresh one.
    onnx.save_model(
        ir.to_proto(model),
        path,
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="model.data" if external else None,
        size_threshold=0,
    )
    onnx.checker.check_model(path, full_check=True)
    restored = onnx.load_model(path)
    stored = next(t for t in restored.graph.initializer if t.name == "state")
    assert onnx.numpy_helper.to_array(stored).tobytes(order="C") == payload
    assert {
        entry.key: entry.value for entry in stored.metadata_props
    } == expected_properties
    if external:
        assert (tmp_path / "model.data").is_file()
    loaded_ir = ir.load(path)
    loaded_input = next(
        value for value in loaded_ir.graph.inputs if value.name == "state"
    )
    assert loaded_ir.graph.initializers["state"] is loaded_input
    assert loaded_input.const_value.numpy().tobytes(order="C") == payload
    assert dict(loaded_input.const_value.metadata_props) == expected_properties


@pytest.mark.parametrize("x64", [False, True])
def test_weak_fp32_scalar_keeps_fp16_activation_output(x64: bool) -> None:
    with jax.enable_x64(x64):
        # Construct a weak FP32 source even when the caller enables x64.
        with jax.enable_x64(False):
            state = jnp.asarray(0.5)
        assert state.weak_type and state.dtype == jnp.float32
        x = np.array([-2.0, 0.0, 3.0], dtype=np.float16)

        def forward(x: jax.Array, scale: jax.Array) -> jax.Array:
            return x * scale

        expected = np.asarray(forward(jnp.asarray(x), state))
        assert expected.dtype == np.float16
        model = export_with_state(
            forward,
            [jax.ShapeDtypeStruct(x.shape, x.dtype)],
            [StateSlot("scale", state, {})],
            data_input_names=["x"],
        )
        assert model.graph.outputs[0].dtype == ir.DataType.FLOAT16
        scale = model.graph.initializers["scale"]
        assert scale.shape == ir.Shape(()) and scale.dtype == ir.DataType.FLOAT
        actual = _session(model).run(None, {"x": x})[0]
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_unused_state_is_rejected_or_explicitly_retained() -> None:
    spec = jax.ShapeDtypeStruct((2,), np.float32)
    state = np.ones(2, dtype=np.float32)
    with pytest.raises(ValueError, match="unused"):
        export_with_state(
            lambda x, state: x + 1,
            [spec],
            [StateSlot("state", state, {})],
            data_input_names=["x"],
        )
    model = export_with_state(
        lambda x, state: x + 1,
        [spec],
        [StateSlot("state", state, {}, allow_unused=True)],
        data_input_names=["x"],
    )
    value = model.graph.initializers["state"]
    assert value.is_graph_input() and not value.uses()
    actual = _session(model).run(None, {"x": state, "state": state * 9})[0]
    np.testing.assert_array_equal(actual, state + 1)


def test_state_returned_directly_is_used() -> None:
    state = np.array([1.0, -2.0], dtype=np.float32)
    model = export_with_state(
        lambda state: state,
        [],
        [StateSlot("state", state, {})],
        data_input_names=[],
    )
    value = model.graph.initializers["state"]
    assert value.is_graph_output()
    np.testing.assert_array_equal(_session(model).run(None, {})[0], state)


@pytest.mark.parametrize(
    "slot, message",
    [
        (StateSlot("bad/name", np.ones(2, np.float32), {}), "C90"),
        (StateSlot("é", np.ones(2, np.float32), {}), "C90"),
        (StateSlot("state", [1.0, 2.0], {}), "JAX or NumPy"),
        (StateSlot("state", np.ones(2, np.float16), {}), "FP32"),
        (StateSlot("state", np.ones(2, np.float64), {}), "FP32"),
        (StateSlot("state", np.ones(2, np.int32), {}), "FP32"),
        (StateSlot("state", np.ones(2, dtype=">f4"), {}), "FP32"),
        (StateSlot("state", np.ones(0, np.float32), {}), "empty"),
        (StateSlot("state", np.ones(2, np.float32), {}, allow_unused=1), "bool"),
        (StateSlot("state", np.ones(2, np.float32), []), "mapping"),
        (StateSlot("state", np.ones(2, np.float32), {"": "value"}), "nonempty"),
        (StateSlot("state", np.ones(2, np.float32), {1: "value"}), "string"),
        (StateSlot("state", np.ones(2, np.float32), {"key": 1}), "string"),
        (StateSlot("state", np.ones(2, np.float32), {"\ud800": "v"}), "UTF-8"),
        (StateSlot("state", np.ones(2, np.float32), {"key": "\ud800"}), "UTF-8"),
    ],
)
def test_invalid_source_records_fail_before_conversion_without_artifacts(
    slot: StateSlot, message: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_export(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Invalid source record reached conversion")

    monkeypatch.setattr(example, "to_onnx", unexpected_export)
    monkeypatch.chdir(tmp_path)
    with pytest.raises((TypeError, ValueError), match=message):
        export_with_state(
            lambda x, state: x + state,
            [jax.ShapeDtypeStruct((2,), np.float32)],
            [slot],
            data_input_names=["x"],
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "data_names, slot_names, message",
    [
        (["x"], ["state", "state"], "unique"),
        (["state"], ["state"], "unique"),
        ([""], ["state"], "nonempty"),
        ([], ["state"], "one data input name"),
    ],
)
def test_invalid_signature_names_fail_before_conversion(
    data_names: list[str],
    slot_names: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_export(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Invalid names reached conversion")

    monkeypatch.setattr(example, "to_onnx", unexpected_export)
    with pytest.raises(ValueError, match=message):
        export_with_state(
            lambda *args: args[0],
            [jax.ShapeDtypeStruct((2,), np.float32)],
            [StateSlot(name, np.ones(2, np.float32), {}) for name in slot_names],
            data_input_names=data_names,
        )


@pytest.mark.parametrize(
    "defect",
    ["shape", "dtype", "constant", "initializer", "missing", "unknown", "reordered"],
)
def test_exported_slot_conflicts_fail_before_any_attachment(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = ir.Value(
        name="first", shape=ir.Shape((2,)), type=ir.TensorType(ir.DataType.FLOAT)
    )
    second = ir.Value(
        name="second", shape=ir.Shape((2,)), type=ir.TensorType(ir.DataType.FLOAT)
    )
    if defect == "shape":
        second.shape = ir.Shape((1, 2))
    elif defect == "dtype":
        second.dtype = ir.DataType.FLOAT16
    elif defect in {"constant", "initializer"}:
        second.const_value = ir.tensor(np.ones(2, np.float32), name="second")
    inputs = [first] if defect == "missing" else [first, second]
    if defect == "reordered":
        inputs.reverse()
    if defect == "unknown":
        second.name = "unexpected"
    graph = ir.Graph(inputs, list(inputs), nodes=[], opset_imports={"": 23})
    if defect == "initializer":
        graph.register_initializer(second)
    model = ir.Model(graph, ir_version=10)
    original_initializers = dict(graph.initializers)
    monkeypatch.setattr(example, "to_onnx", lambda *args, **kwargs: model)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        ValueError, match="shape|dtype|constant|initializer|names differ"
    ):
        export_with_state(
            lambda first, second: (first, second),
            [],
            [
                StateSlot(name, np.ones(2, np.float32), {})
                for name in ["first", "second"]
            ],
            data_input_names=[],
        )
    assert first.const_value is None
    assert dict(graph.initializers) == original_initializers
    assert not list(tmp_path.iterdir())


def test_host_evaluation_of_selected_state_fails_without_closure_fallback(
    tmp_path: Path,
) -> None:
    def forward(x: jax.Array, state: jax.Array) -> jax.Array:
        return x + np.asarray(state)

    with pytest.raises(
        Exception, match="TracerArrayConversionError|__array__|traced array"
    ):
        export_with_state(
            forward,
            [jax.ShapeDtypeStruct((2,), np.float32)],
            [StateSlot("state", np.ones(2, np.float32), {})],
            data_input_names=["x"],
        )
    assert not list(tmp_path.iterdir())


def test_decorated_stateful_module_fails_without_closure_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @onnx_function
    class Stateful(eqx.Module):
        weight: jax.Array

        def __call__(self, x: jax.Array) -> jax.Array:
            return x * self.weight

    def forward(x: jax.Array, state: jax.Array) -> jax.Array:
        # Reconstructing a decorated module does not pass state as a free
        # ONNX Function argument. This unsupported boundary must fail loudly.
        return Stateful(state)(x)

    state = np.array([2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(forward(jnp.ones(2), jnp.asarray(state)), state)
    monkeypatch.chdir(tmp_path)
    with pytest.raises((jax.errors.TracerArrayConversionError, RuntimeError)):
        export_with_state(
            forward,
            [jax.ShapeDtypeStruct((2,), np.float32)],
            [StateSlot("state", state, {})],
            data_input_names=["x"],
        )
    assert not list(tmp_path.iterdir())


def test_cond_branch_capture_uses_the_overridable_state_slot() -> None:
    def forward(x: jax.Array, state: jax.Array) -> jax.Array:
        return jax.lax.cond(
            jnp.sum(x) > 0,
            lambda value: value + state,
            lambda value: value - state,
            x,
        )

    state = np.array([2.0, 3.0], dtype=np.float32)
    replacement = np.array([5.0, -7.0], dtype=np.float32)
    model = export_with_state(
        forward,
        [jax.ShapeDtypeStruct((2,), np.float32)],
        [StateSlot("state", state, {})],
        data_input_names=["x"],
    )
    assert any(node.op_type == "If" for node in model.graph)
    assert model.graph.initializers["state"].uses()
    session = _session(model)
    for x in [np.ones(2, np.float32), -np.ones(2, np.float32)]:
        for overrides, active_state in [
            ({}, state),
            ({"state": replacement}, replacement),
        ]:
            expected = forward(jnp.asarray(x), jnp.asarray(active_state))
            actual = session.run(None, {"x": x, **overrides})[0]
            np.testing.assert_array_equal(actual, expected)
