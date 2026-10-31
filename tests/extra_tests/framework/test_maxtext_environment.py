# tests/extra_tests/framework/test_maxtext_environment.py

"""Keep MaxText's configuration side effects inside model construction."""

from contextlib import nullcontext
import importlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


_STUB_FLAGS = "stub-generated-xla-flags"


def _stub_maxtext(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    incoming: str | None,
    failure: str | None = None,
) -> tuple[Any, Path, object, object, list[str], ValueError]:
    # Import before setting dummy flags. Every operation below is stubbed;
    # these strings must never reach the native JAX backend.
    module = importlib.import_module("jax2onnx.plugins.examples.maxtext.maxtext_models")
    if incoming is None:
        monkeypatch.delenv("XLA_FLAGS", raising=False)
    else:
        monkeypatch.setenv("XLA_FLAGS", incoming)
    active_flags = incoming or _STUB_FLAGS
    events: list[str] = []
    error = ValueError(f"stub {failure} failure")
    config = object()
    model = object()
    supplied_rngs = object()
    default_rngs = object()
    output = object()
    config_path = tmp_path / "configs" / "models" / "stub-model.yml"
    base_path = config_path.parent.parent / "base.yml"
    base_path.parent.mkdir(parents=True)
    base_path.touch()
    expected_argv = [
        "train.py",
        str(base_path),
        "model_name=stub-model",
        "run_name=jax2onnx_export",
        "base_output_directory=/tmp/maxtext_export",
        "dataset_path=/tmp/dataset",
        "per_device_batch_size=3",
        "max_target_length=7",
        "max_prefill_predict_length=7",
        "enable_checkpointing=false",
        "use_jax_splash=false",
        "num_decoder_layers=2",
    ]

    def observe(stage: str) -> None:
        assert os.environ.get("XLA_FLAGS") == active_flags
        events.append(stage)
        if stage == failure:
            raise error

    def initialize(argv: list[str]) -> object:
        assert argv == expected_argv
        assert os.environ.get("XLA_FLAGS") == incoming
        if not incoming:
            os.environ["XLA_FLAGS"] = _STUB_FLAGS
        observe("initialize")
        return config

    def make_rngs(seed: int) -> object:
        assert seed == 0
        observe("rngs")
        return default_rngs

    def restore_random() -> None:
        observe("restore_random")

    def patch_random() -> Any:
        observe("patch_random")
        return restore_random

    def from_config(
        actual_config: object, *, model_mode: object, rngs: object
    ) -> object:
        assert actual_config is config
        assert model_mode == "stub-mode"
        assert rngs is default_rngs or rngs is supplied_rngs
        observe("create")
        return model

    def strip(actual_model: object) -> None:
        assert actual_model is model
        observe("strip")

    def wrap(actual_model: object, *, seq_len: int, model_mode: object) -> object:
        assert actual_model is model
        assert seq_len == 7
        assert model_mode == "stub-mode"
        observe("wrapper")
        return output

    monkeypatch.setattr(module, "MAXTEXT_AVAILABLE", True)
    monkeypatch.setattr(module, "pyconfig", SimpleNamespace(initialize=initialize))
    monkeypatch.setattr(
        module, "model_creation_utils", SimpleNamespace(from_config=from_config)
    )
    monkeypatch.setattr(module, "MODEL_MODE_TRAIN", "stub-mode")
    monkeypatch.setattr(module, "nnx", SimpleNamespace(Rngs=make_rngs))
    monkeypatch.setattr(
        module, "_effective_model_overrides", lambda _: {"num_decoder_layers": 2}
    )
    monkeypatch.setattr(module, "_patch_random_initializers", patch_random)
    monkeypatch.setattr(module, "_maxtext_construction_readiness", nullcontext)
    monkeypatch.setattr(module, "_strip_nnx_rngs", strip)
    monkeypatch.setattr(module, "MaxTextTextOnlyWrapper", wrap)
    return module, config_path, output, supplied_rngs, events, error


@pytest.mark.parametrize("incoming", [None, "", "stub-caller-xla-flags"])
def test_maxtext_restores_caller_flags_after_two_complete_constructions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, incoming: str | None
) -> None:
    module, config_path, output, supplied_rngs, events, _ = _stub_maxtext(
        monkeypatch, tmp_path, incoming
    )
    # Exercise explicit and default RNG paths in sequential model requests.
    for rngs in (supplied_rngs, None):
        start = len(events)
        assert (
            module.get_maxtext_model(
                str(config_path), batch_size=3, seq_len=7, rngs=rngs
            )
            is output
        )
        assert os.environ.get("XLA_FLAGS") == incoming
        assert ("XLA_FLAGS" in os.environ) is (incoming is not None)
        assert events[start:] == [
            "initialize",
            *(["rngs"] if rngs is None else []),
            "patch_random",
            "create",
            "restore_random",
            "strip",
            "wrapper",
        ]


@pytest.mark.parametrize("incoming", [None, "", "stub-caller-xla-flags"])
@pytest.mark.parametrize(
    "failure", ["initialize", "rngs", "patch_random", "create", "strip", "wrapper"]
)
def test_maxtext_restores_caller_flags_when_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    incoming: str | None,
    failure: str,
) -> None:
    module, config_path, _, _, events, error = _stub_maxtext(
        monkeypatch, tmp_path, incoming, failure
    )
    with pytest.raises((ValueError, RuntimeError)) as caught:
        module.get_maxtext_model(str(config_path), batch_size=3, seq_len=7)
    if failure == "create":
        assert type(caught.value) is RuntimeError
        assert caught.value.__cause__ is error
        assert str(caught.value) == (
            "MaxText model init failed for stub-model.yml: stub create failure"
        )
    else:
        assert caught.value is error
    assert os.environ.get("XLA_FLAGS") == incoming
    assert ("XLA_FLAGS" in os.environ) is (incoming is not None)
    if failure in {"create", "strip", "wrapper"}:
        assert events.count("restore_random") == 1


def _readiness_subject(monkeypatch, dense_general):
    module = importlib.import_module("jax2onnx.plugins.examples.maxtext.maxtext_models")
    monkeypatch.setattr(module, "_import_first_available", lambda _: dense_general)
    return module


@pytest.mark.parametrize(
    ("dtype_name", "enable_x64"),
    [
        ("float16", False),
        ("bfloat16", False),
        ("float32", False),
        ("float32", True),
        ("float64", True),
    ],
)
def test_dense_readiness_preserves_original_parameters_rng_and_identities(
    monkeypatch, dtype_name, enable_x64
):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    calls = []

    class Dense(nnx.Module):
        def __init__(self, values, *, rngs, scale, marker):
            init = jax.nn.initializers.variance_scaling(
                1.0, "fan_in", "truncated_normal"
            )
            kernel = init(rngs.params(), (values.size, values.size), values.dtype)
            self.kernel = nnx.Param(kernel / jnp.asarray(scale, values.dtype))
            self.bias = nnx.Param(
                jax.random.uniform(
                    rngs.params(), values.shape, values.dtype, minval=-1.0, maxval=1.0
                )
            )
            self.special = nnx.Param(values)
            calls.append(
                (
                    self,
                    values,
                    rngs,
                    scale,
                    marker,
                    (self.kernel, self.bias, self.special),
                )
            )

    module = _readiness_subject(monkeypatch, Dense)
    original = Dense.__init__
    signature = inspect.signature(original)
    waits = []
    native_ready = module.jax.block_until_ready

    def observe_ready(state):
        waits.append(tuple(id(leaf) for leaf in jax.tree_util.tree_leaves(state)))
        return native_ready(state)

    monkeypatch.setattr(module.jax, "block_until_ready", observe_ready)
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", enable_x64)
    try:
        values = jnp.asarray(
            [
                -0.0,
                0.0,
                1.5,
                -2.75,
                float("nan"),
                float("inf"),
                -float("inf"),
                2.0**-24,
            ],
            dtype=dtype_name,
        )
        marker = object()
        models, streams = [], []
        for ready in (False, True):
            rngs = nnx.Rngs(42)
            restore = module._patch_random_initializers()
            try:
                context = (
                    module._maxtext_construction_readiness() if ready else nullcontext()
                )
                with context:
                    assert inspect.signature(Dense.__init__) == signature
                    model = Dense(values, rngs=rngs, scale=0.125, marker=marker)
            finally:
                restore()
            models.append(model)
            streams.append(rngs)
            call = calls[-1]
            assert call[0] is model and call[1] is values and call[2] is rngs
            assert call[3] == 0.125 and call[4] is marker
            assert all(
                a is b
                for a, b in zip(
                    call[5], (model.kernel, model.bias, model.special), strict=True
                )
            )
            assert Dense.__init__ is original
            assert len(waits) == int(ready)
        assert len(calls) == 2
        for before, after in zip(
            (models[0].kernel, models[0].bias, models[0].special),
            (models[1].kernel, models[1].bias, models[1].special),
            strict=True,
        ):
            a, b = np.asarray(before.get_value()), np.asarray(after.get_value())
            assert a.shape == b.shape and a.dtype == b.dtype == np.dtype(dtype_name)
            # Readiness preserves the parameter buffers, including their stored bits.
            assert a.tobytes() == b.tobytes()
            np.testing.assert_array_equal(np.signbit(a[a == 0]), np.signbit(b[b == 0]))
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        assert np.any(np.asarray(models[0].bias.get_value()) != 0)
        counts = [int(np.asarray(r.default.count.get_value())) for r in streams]
        assert counts == [2, 2]
        keys = [
            np.asarray(jax.random.key_data(r.default.key.get_value())) for r in streams
        ]
        np.testing.assert_array_equal(*keys)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


def test_dense_readiness_preserves_direct_constructor_contract(monkeypatch):
    payload, keyword, returned, state = object(), object(), object(), {"leaf": object()}
    calls, waits = [], []

    class Dense:
        def __init__(self, value, /, *, option):
            calls.append((self, value, option))
            return returned

    module = _readiness_subject(monkeypatch, Dense)
    original = Dense.__init__
    monkeypatch.setattr(module.nnx, "state", lambda instance: state)
    monkeypatch.setattr(
        module.jax, "block_until_ready", lambda actual: waits.append(actual)
    )
    instance = object.__new__(Dense)
    with module._maxtext_construction_readiness():
        assert inspect.signature(Dense.__init__) == inspect.signature(original)
        assert Dense.__init__(instance, payload, option=keyword) is returned
    assert calls == [(instance, payload, keyword)] and waits == [state]
    assert Dense.__init__ is original


@pytest.mark.parametrize("failure", ["constructor", "state", "ready", "body"])
def test_dense_readiness_restores_exact_constructor_on_failure(monkeypatch, failure):
    error = ValueError("original failure")
    calls = []

    class Dense:
        def __init__(self):
            calls.append(self)
            if failure == "constructor":
                raise error

    module = _readiness_subject(monkeypatch, Dense)
    original = Dense.__init__

    def state(instance):
        if failure == "state":
            raise error
        return {}

    def ready(value):
        if failure == "ready":
            raise error
        return value

    monkeypatch.setattr(module.nnx, "state", state)
    monkeypatch.setattr(module.jax, "block_until_ready", ready)
    with pytest.raises(ValueError) as caught:
        with module._maxtext_construction_readiness():
            Dense()
            if failure == "body":
                raise error
    assert caught.value is error and len(calls) == 1
    assert Dense.__init__ is original


def test_dense_readiness_nested_scopes_restore_their_own_constructor(monkeypatch):
    calls = []

    class Dense:
        def __init__(self):
            calls.append(self)

    module = _readiness_subject(monkeypatch, Dense)
    original = Dense.__init__
    monkeypatch.setattr(module.nnx, "state", lambda instance: {})
    with module._maxtext_construction_readiness():
        outer = Dense.__init__
        error = ValueError("nested body")
        with pytest.raises(ValueError) as caught:
            with module._maxtext_construction_readiness():
                inner = Dense.__init__
                assert inner is not outer
                Dense()
                raise error
        assert caught.value is error and Dense.__init__ is outer
        Dense()
    assert Dense.__init__ is original and len(calls) == 2


def test_dense_readiness_skips_mixed_traced_state_and_preserves_graph(monkeypatch):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    class Dense(nnx.Module):
        def __init__(self, value, fixed):
            self.kernel = nnx.Param(value)
            self.fixed = nnx.Param(fixed)

    module = _readiness_subject(monkeypatch, Dense)
    original = Dense.__init__
    fixed = jnp.asarray([-0.0, 1.5], dtype=jnp.float32)

    def build(value, ready):
        context = module._maxtext_construction_readiness() if ready else nullcontext()
        with context:
            model = Dense(value, fixed)
            return model.kernel.get_value() + model.fixed.get_value()

    def forbidden(value):
        pytest.fail("traced construction must skip the whole readiness call")

    monkeypatch.setattr(module.jax, "block_until_ready", forbidden)
    spec = jax.ShapeDtypeStruct((2,), jnp.float32)
    baseline = jax.make_jaxpr(lambda value: build(value, False))(spec)
    wrapped = jax.make_jaxpr(lambda value: build(value, True))(spec)
    assert str(baseline.jaxpr) == str(wrapped.jaxpr)
    assert len(baseline.consts) == len(wrapped.consts)
    for a, b in zip(baseline.consts, wrapped.consts, strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert Dense.__init__ is original


def test_shared_random_patch_does_not_install_dense_readiness(monkeypatch):
    calls = []

    class Dense:
        def __init__(self):
            calls.append(self)

    module = _readiness_subject(monkeypatch, Dense)
    original = Dense.__init__
    monkeypatch.setattr(
        module.nnx, "state", lambda instance: pytest.fail("unexpected state")
    )
    restore = module._patch_random_initializers()
    try:
        Dense()
        assert Dense.__init__ is original
    finally:
        restore()
    assert len(calls) == 1


@pytest.mark.parametrize("fail_create", [False, True])
def test_model_builder_limits_dense_readiness_to_from_config(
    monkeypatch, tmp_path, fail_create
):
    module = importlib.import_module("jax2onnx.plugins.examples.maxtext.maxtext_models")
    scope = module._maxtext_construction_readiness
    module, config_path, output, rngs, events, error = _stub_maxtext(
        monkeypatch, tmp_path, None, "create" if fail_create else None
    )
    original_create = module.model_creation_utils.from_config
    original_strip = module._strip_nnx_rngs
    waits = []

    class Dense:
        def __init__(self):
            pass

    original = Dense.__init__
    monkeypatch.setattr(module, "_maxtext_construction_readiness", scope)
    monkeypatch.setattr(module, "_import_first_available", lambda _: Dense)
    monkeypatch.setattr(module.nnx, "Module", object, raising=False)
    monkeypatch.setattr(module.nnx, "state", lambda instance: {}, raising=False)
    monkeypatch.setattr(
        module.jax, "block_until_ready", lambda state: waits.append(state)
    )

    def create(*args, **kwargs):
        assert Dense.__init__ is not original
        Dense()
        return original_create(*args, **kwargs)

    def strip(model):
        assert Dense.__init__ is original
        original_strip(model)

    monkeypatch.setattr(module.model_creation_utils, "from_config", create)
    monkeypatch.setattr(module, "_strip_nnx_rngs", strip)
    if fail_create:
        with pytest.raises(RuntimeError) as caught:
            module.get_maxtext_model(
                str(config_path), batch_size=3, seq_len=7, rngs=rngs
            )
        assert caught.value.__cause__ is error
    else:
        assert (
            module.get_maxtext_model(
                str(config_path), batch_size=3, seq_len=7, rngs=rngs
            )
            is output
        )
    assert len(waits) == 1 and Dense.__init__ is original


def test_real_maxtext_dense_readiness_parameter_parity():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    module = importlib.import_module("jax2onnx.plugins.examples.maxtext.maxtext_models")
    if not module.MAXTEXT_AVAILABLE:
        pytest.skip("the optional MaxText source is unavailable")
    dense_general = module._import_first_available(
        (
            ("MaxText.layers.linears", "DenseGeneral"),
            ("maxtext.layers.linears", "DenseGeneral"),
        )
    )
    native_initializer = module._import_first_available(
        (
            ("MaxText.layers.initializers", "nd_dense_init"),
            ("maxtext.layers.initializers", "nd_dense_init"),
        )
    )(1.0, "fan_in", "uniform")

    def query_initializer(*args):
        return native_initializer(*args) / jnp.asarray(2.0, dtype=args[2])

    original = dense_general.__init__
    models, streams = [], []
    for ready in (False, True):
        rngs = nnx.Rngs(42)
        restore = module._patch_random_initializers()
        try:
            context = (
                module._maxtext_construction_readiness() if ready else nullcontext()
            )
            with context:
                assert (dense_general.__init__ is not original) is ready
                model = dense_general(
                    in_features_shape=2,
                    out_features_shape=3,
                    kernel_init=query_initializer,
                    kernel_axes=("embed", "mlp"),
                    use_bias=True,
                    dtype=jnp.float32,
                    weight_dtype=jnp.float32,
                    rngs=rngs,
                )
        finally:
            restore()
        assert dense_general.__init__ is original
        assert model.kernel_init is query_initializer
        leaves = jax.tree_util.tree_leaves(nnx.state(model))
        assert len(leaves) == 2 and all(isinstance(leaf, jax.Array) for leaf in leaves)
        assert {id(leaf) for leaf in leaves} == {
            id(model.kernel.get_value()),
            id(model.bias.get_value()),
        }
        models.append(model)
        streams.append(rngs)
    assert np.any(np.asarray(models[0].kernel.get_value()) != 0)
    for name, shape in (("kernel", (2, 3)), ("bias", (3,))):
        before = np.asarray(getattr(models[0], name).get_value())
        after = np.asarray(getattr(models[1], name).get_value())
        assert before.shape == after.shape == shape
        assert before.dtype == after.dtype == np.float32
        assert before.tobytes() == after.tobytes()
        np.testing.assert_array_equal(
            np.signbit(before[before == 0]), np.signbit(after[after == 0])
        )
    assert [int(np.asarray(r.default.count.get_value())) for r in streams] == [2, 2]
    np.testing.assert_array_equal(
        *[np.asarray(jax.random.key_data(r.default.key.get_value())) for r in streams]
    )
