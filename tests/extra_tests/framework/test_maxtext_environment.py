# tests/extra_tests/framework/test_maxtext_environment.py

"""Keep MaxText's configuration side effects inside model construction."""

import importlib
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
