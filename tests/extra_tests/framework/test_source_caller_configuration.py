# tests/extra_tests/framework/test_source_caller_configuration.py

"""Caller state restoration around the existing private SOURCE export route."""

from contextlib import ExitStack

import jax
import jax.numpy as jnp
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat import jax as compat
from jax2onnx.converter.precision_policy import PrecisionPolicy


_NAMES = (
    "jax_enable_x64",
    "jax_default_matmul_precision",
    "jax_numpy_dtype_promotion",
    "jax_numpy_rank_promotion",
)
_INITIAL = (False, "default", "standard", "allow")
_CHANGED = (True, "highest", "strict", "raise")


def _states():
    native = compat._native_config
    return (
        native.enable_x64,
        native.default_matmul_precision,
        native.numpy_dtype_promotion,
        native.numpy_rank_promotion,
    )


def _snapshot():
    return tuple(
        (state.get_global(), state.get_local(), state.value) for state in _states()
    )


def _assert_exact(before):
    after = _snapshot()
    for old, new in zip(before, after, strict=True):
        assert all(left is right for left, right in zip(old, new, strict=True))


def _update(values):
    for name, value in zip(_NAMES, values, strict=True):
        jax.config.update(name, value)


@pytest.fixture(autouse=True)
def _restore_actual_test_environment():
    saved = _snapshot()
    try:
        _update(_INITIAL)
        for state in _states():
            state.set_local(compat._native_config.config_ext.unset)
            hook = state._update_thread_local_hook
            if hook is not None:
                hook(None)
        yield
    finally:
        # Independent native cleanup prevents a failed repair assertion from
        # contaminating unrelated tests; it is not an assertion oracle.
        for state, (global_value, local_value, _) in zip(_states(), saved, strict=True):
            state._set(global_value)
            state.set_local(local_value)
            hook = state._update_thread_local_hook
            if hook is not None:
                hook(
                    None
                    if local_value is compat._native_config.config_ext.unset
                    else local_value
                )


class _CallbackFailure(RuntimeError):
    pass


@pytest.mark.parametrize("callback_fails", [False, True])
@pytest.mark.parametrize("enclosing_locals", [False, True])
def test_real_source_export_restores_global_and_local_after_callback_updates(
    callback_fails, enclosing_locals
):
    with ExitStack() as stack:
        if enclosing_locals:
            # Every local differs from its global; x64 must restore globalFalse
            # while preserving the enclosing caller's effective/localTrue.
            for state, value in zip(
                _states(), (True, "high", "strict", "warn"), strict=True
            ):
                stack.enter_context(state(value))
        before = _snapshot()
        seen = []
        failure = _CallbackFailure("original source callback failed")

        def model(x):
            _update(_CHANGED)
            observed = _snapshot()
            seen.append(observed)
            for original, current in zip(before, observed, strict=True):
                assert current[1] is original[1]  # no hidden local overlay
                if current[1] is compat._native_config.config_ext.unset:
                    assert current[2] is current[0]  # update visible inside call
                else:
                    assert current[2] is original[2]  # caller local wins
            assert compat._native_config.enable_x64.get_global() is True
            if callback_fails:
                raise failure
            return x + jnp.asarray(1, dtype=x.dtype)

        if callback_fails:
            with pytest.raises(_CallbackFailure) as caught:
                ui._to_onnx(
                    model,
                    [jax.ShapeDtypeStruct((2,), jnp.float32)],
                    precision_policy=PrecisionPolicy.SOURCE,
                )
            assert caught.value is failure
        else:
            result = ui._to_onnx(
                model,
                [jax.ShapeDtypeStruct((2,), jnp.float32)],
                precision_policy=PrecisionPolicy.SOURCE,
            )
            assert result.graph.input and result.graph.output
        assert seen
        _assert_exact(before)
    # Exiting the original caller contexts must expose the original globalFalse.
    assert compat._native_config.enable_x64.get_global() is False
    assert compat._native_config.enable_x64.value is False


@pytest.mark.parametrize("inner_fails", [False, True])
def test_native_helper_is_reentrant_and_preserves_outer_callback_updates(inner_fails):
    before = _snapshot()
    with compat.preserve_caller_configuration():
        _update(_CHANGED)
        outer = _snapshot()
        failure = _CallbackFailure("inner callback failed")
        try:
            with compat.preserve_caller_configuration():
                _update(_INITIAL)
                if inner_fails:
                    raise failure
        except _CallbackFailure as caught:
            assert inner_fails and caught is failure
        _assert_exact(outer)
    _assert_exact(before)


@pytest.mark.parametrize("old_flag", [False, True])
def test_existing_public_legacy_flag_does_not_enter_new_source_scope(
    monkeypatch, old_flag
):
    def source_scope_must_not_run():
        raise AssertionError("LEGACY entered SOURCE preservation scope")

    monkeypatch.setattr(ui, "preserve_caller_configuration", source_scope_must_not_run)
    seen = []

    def model(x):
        state = _states()[0]
        seen.append(
            (state.get_global(), state.get_local(), bool(jax.config.x64_enabled))
        )
        return x + jnp.asarray(1, dtype=x.dtype)

    outside = _snapshot()
    with jax.enable_x64(not old_flag):
        inside = _snapshot()
        result = ui.to_onnx(
            model,
            [jax.ShapeDtypeStruct((2,), jnp.float32)],
            enable_double_precision=old_flag,
        )
        _assert_exact(inside)
    _assert_exact(outside)
    assert result.graph.input and result.graph.output
    assert seen and all(
        global_value is old_flag
        and local_value is (not old_flag)
        and effective_value is (not old_flag)
        for global_value, local_value, effective_value in seen
    )


def _observe_boundary_update(before, seen, failure):
    _update(_CHANGED)
    observed = _snapshot()
    seen.append(observed)
    for initial, current, changed in zip(before, observed, _CHANGED, strict=True):
        assert current[0] == changed
        assert current[1] is initial[1]
        if current[1] is compat._native_config.config_ext.unset:
            assert current[2] is current[0]
        else:
            assert current[2] is initial[2]
    if failure is not None:
        raise failure


@pytest.mark.parametrize(
    "stage", ["input", "protocol", "name", "path", "mapping", "log-repr"]
)
@pytest.mark.parametrize("boundary_fails", [False, True])
@pytest.mark.parametrize("enclosing_locals", [False, True])
def test_source_whole_entry_restores_early_user_protocol_settings(
    stage,
    boundary_fails,
    enclosing_locals,
    tmp_path,
):
    outside = _snapshot()
    with ExitStack() as stack:
        if enclosing_locals:
            for state, value in zip(
                _states(), (True, "high", "strict", "warn"), strict=True
            ):
                stack.enter_context(state(value))
        before = _snapshot()
        seen = []
        failure = (
            _CallbackFailure("early source boundary failed") if boundary_fails else None
        )

        def touch():
            _observe_boundary_update(before, seen, failure)

        spec = jax.ShapeDtypeStruct((2,), jnp.float32)

        class Inputs(list):
            def __iter__(self):
                touch()
                return super().__iter__()

        class Names(list):
            def __iter__(self):
                touch()
                return super().__iter__()

        class Declaration:
            dtype = jnp.float32
            weak_type = False

            @property
            def shape(self):
                touch()
                return (2,)

        class Destination:
            def __fspath__(self):
                touch()
                return str(tmp_path / "early-config.onnx")

        class Parameters(dict):
            def items(self):
                touch()
                return super().items()

        def model(x, *, bias):
            return x + bias

        class LoggedInputs(list):
            def __repr__(self):
                touch()
                return "SourceBoundaryInputs"

        inputs = [Declaration() if stage == "protocol" else spec]
        if stage == "input":
            inputs = Inputs(inputs)
        elif stage == "log-repr":
            inputs = LoggedInputs(inputs)
        names = Names(["data"]) if stage == "name" else ["data"]
        params = {"bias": spec}
        if stage == "mapping":
            params = Parameters(params)
        options = {}
        if stage == "path":
            options = {"return_mode": "file", "output_path": Destination()}
        if boundary_fails:
            with pytest.raises(_CallbackFailure) as caught:
                ui._to_onnx(
                    model,
                    inputs,
                    input_params=params,
                    input_names=names,
                    output_names=["result"],
                    precision_policy=PrecisionPolicy.SOURCE,
                    **options,
                )
            assert caught.value is failure
        else:
            result = ui._to_onnx(
                model,
                inputs,
                input_params=params,
                input_names=names,
                output_names=["result"],
                precision_policy=PrecisionPolicy.SOURCE,
                **options,
            )
            if stage == "path":
                assert result == str(tmp_path / "early-config.onnx")
                assert (tmp_path / "early-config.onnx").is_file()
            else:
                assert result.graph.input and result.graph.output
        assert seen, "the selected real early protocol was not reached"
        _assert_exact(before)
    _assert_exact(outside)


@pytest.mark.parametrize("stage", ["naming", "report", "save"])
@pytest.mark.parametrize("boundary_fails", [False, True])
@pytest.mark.parametrize("enclosing_locals", [False, True])
def test_source_whole_entry_restores_finalization_settings_after_late_scope(
    stage,
    boundary_fails,
    enclosing_locals,
    monkeypatch,
    tmp_path,
):
    outside = _snapshot()
    with ExitStack() as stack:
        if enclosing_locals:
            for state, value in zip(
                _states(), (True, "high", "strict", "warn"), strict=True
            ):
                stack.enter_context(state(value))
        before = _snapshot()
        seen = []
        failure = (
            _CallbackFailure("source finalization failed") if boundary_fails else None
        )
        if stage == "naming":
            owner, attribute = ui, "_apply_custom_io_names_on_ir"
        elif stage == "report":
            owner, attribute = ui, "_build_source_report"
        else:
            owner, attribute = ui.onnx, "save_model"
        original = getattr(owner, attribute)

        def observed_stage(*args, **kwargs):
            # A real successful stage still runs on the actual export. The
            # failure case raises the original exception after its setting update.
            _observe_boundary_update(before, seen, failure)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, attribute, observed_stage)
        model_calls = []

        def model(x):
            model_calls.append(_snapshot())
            return x + jnp.asarray(1, dtype=x.dtype)

        options = {"return_mode": "ir"} if stage == "report" else {}
        if stage == "save":
            options = {
                "return_mode": "file",
                "output_path": tmp_path / "final-config.onnx",
            }
        if boundary_fails:
            with pytest.raises(_CallbackFailure) as caught:
                ui._to_onnx(
                    model,
                    [jax.ShapeDtypeStruct((2,), jnp.float32)],
                    input_names=["data"],
                    output_names=["result"],
                    precision_policy=PrecisionPolicy.SOURCE,
                    **options,
                )
            assert caught.value is failure
        else:
            result = ui._to_onnx(
                model,
                [jax.ShapeDtypeStruct((2,), jnp.float32)],
                input_names=["data"],
                output_names=["result"],
                precision_policy=PrecisionPolicy.SOURCE,
                **options,
            )
            if stage == "save":
                assert result == str(tmp_path / "final-config.onnx")
                assert (tmp_path / "final-config.onnx").is_file()
            elif stage == "report":
                assert result.graph.inputs and result.graph.outputs
            else:
                assert result.graph.input and result.graph.output
        assert seen and model_calls
        _assert_exact(before)
    _assert_exact(outside)
