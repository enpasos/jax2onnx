# tests/extra_tests/converter/test_source_runtime_parameters.py

"""Genuine root BOOL operands across tracing, Function scopes and final boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx import user_interface as ui
from jax2onnx.converter import conversion_api
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins.plugin_system import import_all_plugins
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter.test_source_dropout_parameters import (
    DropoutDiagnosticValidator,
)


@pytest.fixture(autouse=True)
def _configuration() -> Iterator[None]:
    from tests.extra_tests.converter.source_precision_support import (
        _global_source_configuration,
    )

    previous = _global_source_configuration()
    import_all_plugins()
    try:
        yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)


def _source(fn: Any, inputs: Any, params: Any, **options: Any) -> Any:
    return ui._to_onnx(
        fn,
        inputs,
        input_params=params,
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=options.pop(
            "validator", DropoutDiagnosticValidator()
        ),
        **options,
    )


def _proto(artifact: Any) -> onnx.ModelProto:
    if isinstance(artifact, ir.Model):
        return ir.to_proto(artifact)
    return (
        artifact if isinstance(artifact, onnx.ModelProto) else onnx.load_model(artifact)
    )


def _session(model: Any, optimize: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(
        _proto(model).SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _select(x: Any, *, flag: Any = True) -> Any:
    return jax.lax.select(flag, x, x + np.float32(20))


@onnx_function
def _named_select(x: Any, *, flag: Any = True) -> Any:
    return jax.lax.select(flag, x, x + np.float32(2))


@onnx_function
def _nested_select(x: Any, *, flag: Any = True) -> Any:
    return _named_select(x, flag=flag)


@onnx_function
def _unused_flag(x: Any, *, flag: Any) -> Any:
    return x + np.float32(1)


def _mixed(x: Any, *, flag: Any = True) -> Any:
    return _nested_select(_select(x, flag=flag), flag=flag)


def _python_if(x: Any, *, flag: Any = True) -> Any:
    value = x if flag else x + np.float32(20)
    return _named_select(value, flag=flag)


@pytest.mark.parametrize(
    "declaration",
    [
        True,
        False,
        np.bool_(True),
        np.asarray(False),
        jnp.asarray(True),
        jax.ShapeDtypeStruct((), np.bool_),
        jax.ShapeDtypeStruct((), np.bool_, weak_type=True),
    ],
)
def test_named_declarations_pair_actual_vars_once_and_preserve_report(
    declaration: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = []
    real = conversion_api._trace_to_jaxpr

    def observe(**kwargs: Any) -> Any:
        trace = real(**kwargs)
        traces.append(trace)
        return trace

    monkeypatch.setattr(conversion_api, "_trace_to_jaxpr", observe)
    artifact = _source(
        _select,
        [jax.ShapeDtypeStruct((2,), np.float32)],
        {"flag": declaration},
        input_names=["features"],
    )
    assert len(traces) == 1
    trace = traces[0]
    assert trace.frozen_params == {}
    assert isinstance(trace.source_inputs, tuple)
    assert (
        len(trace.sds_list) == len(trace.jaxpr.invars) == len(trace.source_inputs) == 2
    )
    for index, binding in enumerate(trace.source_inputs):
        assert binding.trace_index == index
        assert binding.var is trace.jaxpr.invars[index]
        assert binding.internal_name == f"in_{index}"
    assert trace.source_inputs[0].positional_index == 0
    assert trace.source_inputs[1].positional_index is None
    assert trace.source_inputs[1].keyword_name == "flag"
    report = get_precision_report(artifact)
    assert report is not None
    assert [item.exported.name for item in report.inputs] == ["features", "flag"]
    assert report.inputs[1].supplied.name == "flag"
    assert report.inputs[1].supplied.shape == report.inputs[1].traced.shape == ()
    assert report.inputs[1].normalized.dtype == report.inputs[1].traced.dtype == "bool"
    assert report.inputs[1].normalized.weak_type == bool(
        getattr(declaration, "weak_type", False)
    )
    assert {item.category: item.status for item in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.UNVERIFIED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    for optimize in (False, True):
        session = _session(artifact, optimize)
        for flag in (True, False):
            x = np.asarray([2, 4], np.float32)
            np.testing.assert_array_equal(
                session.run(None, {"features": x, "flag": np.asarray(flag)})[0],
                np.asarray(_select(x, flag=flag)),
            )


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("kind", ["mixed", "complete-adapter"])
def test_complete_root_and_function_use_the_same_actual_operand(
    kind: str, optimize: bool
) -> None:
    original = _mixed if kind == "mixed" else _python_if

    def adapter(x: Any, *, flag: Any) -> Any:
        return jax.lax.cond(
            flag,
            lambda value: original(value, flag=True),
            lambda value: original(value, flag=False),
            x,
        )

    target = original if kind == "mixed" else adapter
    x = np.asarray([2, 4], np.float32)
    # Independent original calls occur outside the converter's patched world.
    references = {flag: np.asarray(original(x, flag=flag)) for flag in (True, False)}
    np.testing.assert_array_equal(references[False], [24, 26])
    artifact = _source(target, [jax.ShapeDtypeStruct(x.shape, x.dtype)], {"flag": True})
    assert artifact.functions
    session = _session(artifact, optimize)
    assert [value.name for value in session.get_inputs()] == ["in_0", "flag"]
    for flag in (True, False):
        np.testing.assert_array_equal(
            session.run(None, {"in_0": x, "flag": np.asarray(flag)})[0],
            references[flag],
        )


def test_native_python_if_retains_its_error_and_complete_external_recipe() -> None:
    with pytest.raises(jax.errors.TracerBoolConversionError) as caught:
        _source(_python_if, [jax.ShapeDtypeStruct((2,), np.float32)], {"flag": False})
    assert any("complete unchanged original" in note for note in caught.value.__notes__)
    # Fixed configuration is distinct from a runtime declaration.
    artifact = _source(
        partial(_python_if, flag=False), [jax.ShapeDtypeStruct((2,), np.float32)], {}
    )
    assert len(artifact.graph.input) == 1
    np.testing.assert_array_equal(
        _session(artifact, False).run(None, {"in_0": np.asarray([2, 4], np.float32)})[
            0
        ],
        [24, 26],
    )
    assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize("scope", ["if", "fori", "scan", "while", "zero"])
@pytest.mark.parametrize("optimize", [False, True])
def test_nested_control_flow_retains_genuine_flag_without_extra_ports(
    scope: str, optimize: bool
) -> None:
    def model(x: Any, *, runtime: Any) -> Any:
        def step(value: Any) -> Any:
            return _nested_select(value, flag=runtime)

        if scope == "if":
            return jax.lax.cond(
                jax.lax.reduce_sum(x, (0,)) > np.float32(0), step, step, x
            )
        if scope == "scan":
            return jax.lax.scan(
                lambda value, unused: (step(value), None), x, None, length=2
            )[0]
        if scope == "while":
            return jax.lax.while_loop(
                lambda carry: carry[0] < np.int32(2),
                lambda carry: (carry[0] + np.int32(1), step(carry[1])),
                (np.int32(0), x),
            )[1]
        return jax.lax.fori_loop(
            0, 0 if scope == "zero" else 2, lambda i, value: step(value), x
        )

    x = np.asarray([2, 4], np.float32)
    references = {flag: np.asarray(model(x, runtime=flag)) for flag in (True, False)}
    artifact = _source(
        model, [jax.ShapeDtypeStruct((2,), np.float32)], {"runtime": True}
    )
    assert [value.name for value in artifact.graph.input] == ["in_0", "runtime"]
    assert not any(
        "sink" in name for node in artifact.graph.node for name in node.output
    )
    session = _session(artifact, optimize)
    for flag in (True, False):
        np.testing.assert_array_equal(
            session.run(None, {"in_0": x, "runtime": np.asarray(flag)})[0],
            references[flag],
        )


@pytest.mark.parametrize("mode", ["ir", "proto", "file"])
def test_zero_positional_multiple_flags_order_aliases_and_rename_roundtrip(
    mode: str, tmp_path: Path
) -> None:
    def model(*, second: Any, first: Any) -> Any:
        return second, first, second

    artifact = _source(
        model,
        [],
        {"second": False, "first": True},
        return_mode=mode,
        output_path=tmp_path / "flags.onnx",
        input_names=[],
        output_names=["a", "b", "c"],
    )
    proto = _proto(artifact)
    assert [value.name for value in proto.graph.input] == ["second", "first"]
    assert [value.name for value in proto.graph.output] == ["a", "b", "c"]
    report = get_precision_report(artifact)
    assert report is not None
    assert [item.index for item in report.inputs] == [0, 1]
    assert [item.supplied.name for item in report.inputs] == ["second", "first"]
    for optimize in (False, True):
        actual = _session(artifact, optimize).run(
            None, {"second": np.asarray(False), "first": np.asarray(True)}
        )
        assert [value.item() for value in actual] == [False, True, False]


@pytest.mark.parametrize("mode", ["ir", "proto", "file"])
def test_unused_declared_formal_survives_without_a_sink(
    mode: str, tmp_path: Path
) -> None:
    artifact = _source(
        lambda x, *, unused: x + np.float32(1),
        [jax.ShapeDtypeStruct((2,), np.float32)],
        {"unused": True},
        return_mode=mode,
        output_path=tmp_path / "unused.onnx",
    )
    proto = _proto(artifact)
    assert [value.name for value in proto.graph.input] == ["in_0", "unused"]
    assert all("unused" not in node.input for node in proto.graph.node)
    report = get_precision_report(artifact)
    assert report is not None and len(report.inputs) == 2
    for optimize in (False, True):
        session = _session(artifact, optimize)
        for flag in (True, False):
            np.testing.assert_array_equal(
                session.run(
                    None,
                    {
                        "in_0": np.asarray([2, 4], np.float32),
                        "unused": np.asarray(flag),
                    },
                )[0],
                [3, 5],
            )


@pytest.mark.parametrize("override", ["default", "true", "false", "typed"])
def test_same_named_local_configuration_is_not_promoted_by_root_spelling(
    override: str,
) -> None:
    def model(x: Any, *, flag: Any) -> Any:
        value = _select(x, flag=flag)
        if override == "default":
            return _named_select(value)
        if override == "typed":
            return _named_select(value, flag=jnp.asarray(False))
        return _named_select(value, flag=override == "true")

    x = np.asarray([2, 4], np.float32)
    expected = {flag: np.asarray(model(x, flag=flag)) for flag in (True, False)}
    # Interleave policies and original calls without clearing Function caches.
    to_onnx(
        model, [jax.ShapeDtypeStruct((2,), np.float32)], input_params={"flag": True}
    )
    artifact = _source(model, [jax.ShapeDtypeStruct((2,), np.float32)], {"flag": True})
    for optimize in (False, True):
        session = _session(artifact, optimize)
        for flag in (True, False):
            np.testing.assert_array_equal(
                session.run(None, {"in_0": x, "flag": np.asarray(flag)})[0],
                expected[flag],
            )
            np.testing.assert_array_equal(model(x, flag=flag), expected[flag])


def test_diagnostic_handshake_requires_a_callback_and_accepts_false_valued_callable() -> (
    None
):
    spec = jax.ShapeDtypeStruct((2,), np.float32)
    # The audited facade now admits named operands through its ordinary validator.
    artifact = ui._to_onnx(
        _select,
        [spec],
        input_params={"flag": True},
        precision_policy=PrecisionPolicy.SOURCE,
    )
    assert get_precision_report(artifact).profile == "source-static-v1"
    core = conversion_api._convert_to_ir(
        fn=_select,
        inputs=[spec],
        input_params={"flag": True},
        model_name="default",
        opset=23,
        enable_double_precision=False,
        record_primitive_calls_file=None,
        precision_policy=PrecisionPolicy.SOURCE,
        jaxpr_validator=DropoutDiagnosticValidator(),
    )
    assert [value.name for value in core.graph.inputs] == ["in_0", "in_1"]

    class FalseValidator(DropoutDiagnosticValidator):
        def __bool__(self) -> bool:
            return False

    artifact = _source(_select, [spec], {"flag": True}, validator=FalseValidator())
    assert [value.name for value in artifact.graph.input] == ["in_0", "flag"]


@pytest.mark.parametrize(
    "declaration",
    [
        "True",
        [True],
        (True,),
    ],
)
def test_unsupported_named_declarations_fail_without_coercion(declaration: Any) -> None:
    with pytest.raises(
        SourcePrecisionError, match="named parameter 'flag'.*declaration"
    ):
        _source(
            _select, [jax.ShapeDtypeStruct((2,), np.float32)], {"flag": declaration}
        )
    assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize("name", ["", 1])
def test_named_port_requires_a_nonempty_string(name: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="nonempty strings|string keys"):
        _source(lambda **kwargs: np.float32(1), [], {name: True})


def test_full_name_vector_handles_legal_swap_and_in_n_collision() -> None:
    # Swap positional internal names simultaneously; named inputs stay declaration-named.
    artifact = _source(
        lambda x, y, *, runtime: (x, y, runtime),
        [jax.ShapeDtypeStruct((), np.float32)] * 2,
        {"runtime": True},
        input_names=["in_1", "in_0"],
        output_names=["x", "y", "flag"],
    )
    assert [value.name for value in artifact.graph.input] == ["in_1", "in_0", "runtime"]

    def named(x: Any, *, in_0: Any) -> Any:
        return jax.lax.select(in_0, x, -x)

    with pytest.raises(ValueError, match="collid|duplicate|unique|already"):
        _source(named, [jax.ShapeDtypeStruct((), np.float32)], {"in_0": True})
    artifact = _source(
        named,
        [jax.ShapeDtypeStruct((), np.float32)],
        {"in_0": True},
        input_names=["features"],
    )
    assert [value.name for value in artifact.graph.input] == ["features", "in_0"]


@pytest.mark.parametrize("kind", ["missing", "extra", "reordered", "wrong-type"])
def test_checked_clone_reattachment_rejects_corrupted_actual_inputs(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalize = conversion_api._build_and_finalize_ir_model

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        model = finalize(*args, **kwargs)
        if kind == "missing":
            model.graph.inputs.pop()
        elif kind == "extra":
            model.graph.inputs.append(ir.val("extra", dtype=ir.DataType.BOOL, shape=()))
        elif kind == "reordered":
            model.graph.inputs.reverse()
        else:
            model.graph.inputs[0].type = ir.TensorType(ir.DataType.INT32)
        return model

    monkeypatch.setattr(conversion_api, "_build_and_finalize_ir_model", corrupt)
    with pytest.raises(
        (ValueError, RuntimeError),
        match="strict|shorter|longer|source type|root bindings",
    ):
        _source(lambda *, a, b: (a, b), [], {"a": True, "b": False})


@pytest.mark.parametrize("boundary", ["proto", "file"])
@pytest.mark.parametrize("role", ["input", "output"])
def test_same_typed_serialized_boundary_reorder_is_rejected(
    boundary: str, role: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if boundary == "proto":
        original = ir.to_proto

        def swapped(model: Any) -> Any:
            proto = original(model)
            getattr(proto.graph, role).reverse()
            return proto

        monkeypatch.setattr(ir, "to_proto", swapped)
    else:
        original_load = onnx.load_model

        def swapped_load(*args: Any, **kwargs: Any) -> Any:
            proto = original_load(*args, **kwargs)
            getattr(proto.graph, role).reverse()
            return proto

        monkeypatch.setattr(onnx, "load_model", swapped_load)
    with pytest.raises(SourcePrecisionError, match="names/order"):
        _source(
            lambda *, a, b: (a, b),
            [],
            {"a": True, "b": False},
            return_mode="file" if boundary == "file" else "proto",
            output_path=tmp_path / "corrupt.onnx",
            output_names=["first", "second"],
        )


def test_source_context_rejects_name_only_or_non_bool_flag() -> None:
    context = IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    assert context._promote_call_input_params is False
    with pytest.raises(RuntimeError, match="actual bound"):
        context.ensure_external_flag("flag", None)
    var = object()
    context.bind_value_for_var(
        var, ir.val("numeric", dtype=ir.DataType.FLOAT, shape=())
    )
    with pytest.raises(RuntimeError, match="dtype BOOL"):
        context.ensure_training_mode("flag", var)
    assert len(context.builder.inputs) == 0


def test_unused_function_operand_has_a_real_formal_without_an_identity_sink() -> None:
    artifact = _source(
        lambda x, *, flag: _unused_flag(x, flag=flag),
        [jax.ShapeDtypeStruct((2,), np.float32)],
        {"flag": True},
    )
    function = next(iter(artifact.functions))
    call = next(node for node in artifact.graph.node if node.domain == function.domain)
    assert list(call.input) == ["in_0", "flag"]
    assert len(function.input) == 2
    assert all(function.input[1] not in node.input for node in function.node)
    assert not any(node.op_type == "Identity" for node in function.node)
    for optimize in (False, True):
        session = _session(artifact, optimize)
        for flag in (True, False):
            np.testing.assert_array_equal(
                session.run(
                    None,
                    {"in_0": np.asarray([2, 4], np.float32), "flag": np.asarray(flag)},
                )[0],
                [3, 5],
            )


def test_duplicate_positional_and_named_arguments_remain_a_real_callable_error() -> (
    None
):
    with pytest.raises(TypeError, match="multiple values"):
        _source(
            lambda flag: flag,
            [jax.ShapeDtypeStruct((), np.bool_)],
            {"flag": True},
        )
