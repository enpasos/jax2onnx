# tests/extra_tests/framework/test_initializer_metadata_faults.py

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx import InitializerMetadataError, onnx_function, to_onnx
from jax2onnx import user_interface
from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.plugins.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
    import_all_plugins,
)


@pytest.fixture(autouse=True)
def _clear_local_functions() -> Iterator[None]:
    yield
    for qualified in tuple(ONNX_FUNCTION_PLUGIN_REGISTRY):
        if qualified.startswith(f"{__name__}."):
            PLUGIN_REGISTRY.pop(f"onnx_fn::{qualified}", None)
            ONNX_FUNCTION_PLUGIN_REGISTRY.pop(qualified, None)


def _inject(
    monkeypatch: pytest.MonkeyPatch, mutate: Callable[[ir.Model], None]
) -> list[ir.Model]:
    original = user_interface.postprocess_ir_model
    calls: list[ir.Model] = []

    def corrupt(model: ir.Model) -> Any:
        result = original(model)
        calls.append(model)
        mutate(model)
        return result

    monkeypatch.setattr(user_interface, "postprocess_ir_model", corrupt)
    monkeypatch.setenv("JAX2ONNX_STRICT_OPTIMIZER_FAILURES", "0")
    return calls


def _tagged(model: ir.Model) -> list[ir.Value]:
    return [
        value
        for value in model.graph.initializers.values()
        if value.const_value is not None and value.const_value.metadata_props
    ]


def _replacement(value: ir.Value, *, tagged: bool = True) -> ir.Value:
    assert value.const_value is not None
    tensor = ir.tensor(np.array(value.const_value.numpy(), copy=True))
    if tagged:
        tensor.metadata_props.update(value.const_value.metadata_props)
    return ir.Value(
        name=f"{value.name}_fabricated",
        type=value.type,
        shape=value.shape,
        const_value=tensor,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "remove_key",
        "change_value",
        "swap_tags",
        "swap_payloads",
        "dtype",
        "count",
        "order",
        "merge_equal_sources",
        "fabricated_equal",
        "orphan_survivor",
        "delete_final_carrier",
    ],
)
@pytest.mark.parametrize("mode", ["ir", "proto"])
def test_final_preservation_faults_cannot_return_success(
    monkeypatch: pytest.MonkeyPatch, fault: str, mode: str
) -> None:
    left = jnp.asarray([1.25, 2.5, 3.75], dtype=jnp.float32)
    right = jnp.asarray(
        np.asarray(left) if fault == "merge_equal_sources" else [4.25, 5.5, 6.75],
        dtype=jnp.float32,
    )

    def annotate(source: object) -> dict[str, str] | None:
        if source is left:
            return {"source": "left", "opaque": "DO_NOT_LEAK_CALLBACK_VALUE"}
        if source is right:
            return {"source": "right", "opaque": "DO_NOT_LEAK_CALLBACK_VALUE"}
        return None

    def mutate(model: ir.Model) -> None:
        values = {
            value.const_value.metadata_props["source"]: value
            for value in _tagged(model)
            if value.const_value is not None
        }
        a, b = values["left"], values["right"]
        assert a.const_value is not None and b.const_value is not None
        if fault == "remove_key":
            del a.const_value.metadata_props["opaque"]
        elif fault == "change_value":
            a.const_value.metadata_props["opaque"] = "corrupted"
        elif fault == "swap_tags":
            (
                a.const_value.metadata_props["source"],
                b.const_value.metadata_props["source"],
            ) = ("right", "left")
        elif fault == "swap_payloads":
            left_entries = dict(a.const_value.metadata_props)
            right_entries = dict(b.const_value.metadata_props)
            a.const_value = ir.tensor(np.array(right, copy=True))
            b.const_value = ir.tensor(np.array(left, copy=True))
            a.const_value.metadata_props.update(left_entries)
            b.const_value.metadata_props.update(right_entries)
        elif fault in {"dtype", "count", "order"}:
            array = np.asarray(a.const_value.numpy())
            array = (
                array.astype(np.float16)
                if fault == "dtype"
                else array[:-1] if fault == "count" else array[::-1].copy()
            )
            tensor = ir.tensor(array)
            tensor.metadata_props.update(a.const_value.metadata_props)
            a.const_value = tensor
        elif fault in {"merge_equal_sources", "orphan_survivor"}:
            a.replace_all_uses_with(b, replace_graph_outputs=True)
            if fault == "merge_equal_sources":
                del model.graph.initializers[a.name]
        elif fault == "fabricated_equal":
            replacement = _replacement(a)
            model.graph.register_initializer(replacement)
            a.replace_all_uses_with(replacement, replace_graph_outputs=True)
            del model.graph.initializers[a.name]
        else:
            a.replace_all_uses_with(model.graph.inputs[0], replace_graph_outputs=True)
            del model.graph.initializers[a.name]

    calls = _inject(monkeypatch, mutate)
    with pytest.raises(InitializerMetadataError) as error:
        to_onnx(
            lambda x: (x + left, x * right),
            [(3,)],
            return_mode=mode,
            initializer_metadata=annotate,
        )
    assert len(calls) == 1
    assert error.value.reason == "preservation_failed"
    diagnostic = str(error.value)
    assert diagnostic
    assert "DO_NOT_LEAK_CALLBACK_VALUE" not in diagnostic
    assert "1.25" not in diagnostic and "6.75" not in diagnostic


@pytest.mark.parametrize("shape", [(3,), (1, 3)])
@pytest.mark.parametrize(
    "fault", ["untagged_copy", "replaced_tensor", "replaced_output"]
)
def test_valid_tagged_survivor_cannot_hide_one_lost_live_representation(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], fault: str
) -> None:
    weight = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)

    def apply(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return x + weight, x * jnp.reshape(weight, shape)

    def mutate(model: ir.Model) -> None:
        carriers = _tagged(model)
        assert carriers
        if fault == "replaced_tensor" and len(carriers) > 1:
            # Keep a separately registered same-source carrier valid and live.
            victim = carriers[-1]
            victim.const_value = _replacement(victim, tagged=False).const_value
        elif fault == "replaced_output":
            # Replacing a witnessed use must fail even with its source live elsewhere.
            output = model.graph.outputs[1]
            tensor = ir.tensor(
                np.asarray([4.0, 5.0, 6.0], dtype=np.float32).reshape(shape)
            )
            replacement = ir.Value(
                name="fabricated_output",
                type=output.type,
                shape=output.shape,
                const_value=tensor,
            )
            model.graph.register_initializer(replacement)
            model.graph.outputs[1] = replacement
        else:
            source = carriers[-1]
            uses = list(source.uses())
            assert uses
            replacement = _replacement(source, tagged=False)
            model.graph.register_initializer(replacement)
            uses[-1].node.replace_input_with(uses[-1].idx, replacement)

    calls = _inject(monkeypatch, mutate)
    with pytest.raises(InitializerMetadataError) as error:
        to_onnx(
            apply,
            [(3,)],
            initializer_metadata=lambda source: (
                {"source": "weight"} if source is weight else None
            ),
        )
    assert len(calls) == 1
    assert error.value.reason == "preservation_failed"


def test_losing_only_function_representation_cannot_hide_behind_root_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = jnp.asarray([2.0, 3.0], dtype=jnp.float32)

    @onnx_function
    class Inner:
        def __call__(self, x: jax.Array) -> jax.Array:
            return x + weight

    inner = Inner()

    def mutate(model: ir.Model) -> None:
        assert len(_tagged(model)) == 1
        functions = list(model.functions.values())
        assert len(functions) == 1
        tensors = [
            attribute.as_tensor()
            for node in functions[0]
            for attribute in node.attributes.values()
            if attribute.type == ir.AttributeType.TENSOR
        ]
        assert len(tensors) == 1
        tensors[0].metadata_props.clear()

    calls = _inject(monkeypatch, mutate)
    with pytest.raises(InitializerMetadataError) as error:
        to_onnx(
            lambda x: (x * weight, inner(x)),
            [(2,)],
            initializer_metadata=lambda source: (
                {"source": "shared"} if source is weight else None
            ),
        )
    assert len(calls) == 1
    assert error.value.reason == "preservation_failed"


def test_already_disconnected_accepted_capture_is_a_preservation_failure() -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    observed: list[object] = []

    def apply(x: jax.Array) -> jax.Array:
        _ = x + weight
        return x

    def annotate(source: object) -> dict[str, str]:
        observed.append(source)
        return {"source": "already disconnected"}

    with pytest.raises(InitializerMetadataError) as error:
        to_onnx(apply, [(2,)], initializer_metadata=annotate)
    assert observed and all(source is weight for source in observed)
    assert error.value.reason == "preservation_failed"


def test_lowering_cannot_fabricate_derived_constant_while_original_source_survives() -> (
    None
):
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    import_all_plugins()
    original = PLUGIN_REGISTRY["add"]
    lowered: list[JaxprEqn] = []

    class CorruptAdd:
        def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
            lowered.append(eqn)
            # Deliberately bypass the protected payload-reading helpers.
            ctx.bind_const_for_var(eqn.outvars[0], np.asarray(weight))

    PLUGIN_REGISTRY.register_canonical("add", CorruptAdd())
    try:
        with pytest.raises(InitializerMetadataError) as error:
            to_onnx(
                lambda x: (x + weight, weight),
                [(2,)],
                initializer_metadata=lambda source: (
                    {"source": "weight"} if source is weight else None
                ),
            )
        assert not lowered
        assert error.value.reason == "preservation_failed"
    finally:
        PLUGIN_REGISTRY["add"] = original


@pytest.mark.parametrize("callback_mode", ["omitted", "none", "empty"])
def test_external_lowering_keeps_ordinary_behavior_without_accepted_annotations(
    callback_mode: str,
) -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    import_all_plugins()
    original = PLUGIN_REGISTRY["add"]
    lowered: list[JaxprEqn] = []

    class ExternalAdd:
        def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
            lowered.append(eqn)
            inputs = [ctx.require_value_for_var(var) for var in eqn.invars]
            output = ctx.allocate_value_for_var(eqn.outvars[0])
            ctx.builder.op("Add", inputs, output=output)
            ctx.bind_value_for_var(eqn.outvars[0], output)

    PLUGIN_REGISTRY.register_canonical("add", ExternalAdd())
    try:
        options: dict[str, Any] = {}
        observed: list[object] = []
        if callback_mode != "omitted":

            def annotate(source: object) -> Any:
                observed.append(source)
                return None if callback_mode == "none" else {}

            options["initializer_metadata"] = annotate
        model = to_onnx(lambda x: x + weight, [(2,)], **options)
        assert lowered
        if callback_mode != "omitted":
            assert observed and all(source is weight for source in observed)
        from tests.extra_tests.framework.test_initializer_metadata_contract import (
            _properties,
            _run,
            _tensors,
        )

        assert all(not _properties(tensor) for tensor in _tensors(model))
        x = np.asarray([3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(_run(model, x)[0], x + np.asarray(weight))
    finally:
        PLUGIN_REGISTRY["add"] = original


def test_external_lowering_visited_before_later_annotation_cannot_escape_admission() -> (
    None
):
    skipped = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    accepted = jnp.asarray([3.0, 4.0], dtype=jnp.float32)
    import_all_plugins()
    original = PLUGIN_REGISTRY["add"]
    lowered: list[JaxprEqn] = []
    observed: list[object] = []

    class ExternalAdd:
        def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
            lowered.append(eqn)
            inputs = [ctx.require_value_for_var(var) for var in eqn.invars]
            output = ctx.allocate_value_for_var(eqn.outvars[0])
            ctx.builder.op("Add", inputs, output=output)
            ctx.bind_value_for_var(eqn.outvars[0], output)

    @onnx_function
    class Inner:
        def __call__(self, x: jax.Array) -> jax.Array:
            return x + accepted

    inner = Inner()

    def annotate(source: object) -> dict[str, str] | None:
        observed.append(source)
        return {"source": "accepted later"} if source is accepted else None

    PLUGIN_REGISTRY.register_canonical("add", ExternalAdd())
    try:
        with pytest.raises(InitializerMetadataError) as error:
            to_onnx(lambda x: inner(x + skipped), [(2,)], initializer_metadata=annotate)
        assert len(lowered) == 1
        assert {id(source) for source in observed} == {id(skipped), id(accepted)}
        assert error.value.reason == "preservation_failed"
    finally:
        PLUGIN_REGISTRY["add"] = original
