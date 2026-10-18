# tests/extra_tests/converter/test_fixed_source_preparation_poc.py

"""Execute the finite proof; no dependency on historical /tmp receipts."""

from contextlib import ExitStack
import json

import jax
from jax._src import config as jax_config
import jax.numpy as jnp
import numpy as np
import pytest

from internal_docs.precision.poc import fixed_source_preparation as poc


def test_explicit_pretrace_edit_keeps_original_and_its_repeated_calls_untouched():
    source, _ = poc.source_texts()
    original = poc.compile_source(source)
    prepared, prepared_source = poc.prepare_exact(source, replace_selected_cast=True)
    with pytest.raises(ValueError, match="No explicit authorization"):
        poc.prepare_exact(source)
    for changed in (
        source.replace("float32", "float16"),
        source + "\n",
        source.replace("exact_region", "nested_region"),
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            poc.prepare_exact(changed, replace_selected_cast=True)
    with poc.preserved_settings():
        feed = jnp.asarray([1e8, 1e8 + 1, -1e8, 0, 16777216], dtype=jnp.float64)
        original_jit, prepared_jit = jax.jit(original), jax.jit(prepared)
        for fn, expected in (
            (original, np.array([0, 0, 0, 1, 0], np.float32)),
            (prepared, np.ones(5, np.float64)),
            (original_jit, np.array([0, 0, 0, 1, 0], np.float32)),
            (prepared_jit, np.ones(5, np.float64)),
            (original_jit, np.array([0, 0, 0, 1, 0], np.float32)),
        ):
            assert poc.same_bits(fn(feed), expected)
    assert poc.source_texts()[0] == source == poc.EXACT_SOURCE
    assert "float64" in prepared_source and "float32" in source


@pytest.mark.parametrize(
    "dtype,values,expected",
    [
        (jnp.float16, [2048, 2050, 2052, 2054], np.array([[-3, -1, 1, 3]], np.float32)),
        (jnp.bfloat16, [256, 258, 260, 262], np.array([[-3, -1, 1, 3]], np.float32)),
        (jnp.float32, [1e8, 1e8 + 1, 1e8 + 2, 1e8 + 3], np.zeros((1, 4), np.float32)),
        (
            jnp.float64,
            [1e8, 1e8 + 1, 1e8 + 2, 1e8 + 3],
            np.array([[-1.5, -0.5, 0.5, 1.5]], np.float64),
        ),
    ],
)
def test_minimum_source_decides_from_actual_input_before_trace(dtype, values, expected):
    minimum = poc.compile_source(poc.source_texts()[1])
    with poc.preserved_settings():
        feed = jnp.asarray([values], dtype=dtype)
        assert poc.same_bits(minimum(feed), expected)
        assert poc.same_bits(jax.jit(minimum)(feed), expected)


def test_protected_rounding_sum_castback_and_shared_bitcast_are_preserved():
    original = poc.compile_source(poc.source_texts()[0])
    prepared, _ = poc.prepare_exact(poc.EXACT_SOURCE, replace_selected_cast=True)
    with poc.preserved_settings():
        feed = jnp.asarray([1e8, 1e8 + 1, -1e8, 0, 16777216], dtype=jnp.float64)
        old = poc.protected_consumers(original, feed)
        new = jax.jit(lambda x: poc.protected_consumers(prepared, x))(feed)
    assert all(poc.same_bits(a, b) for a, b in zip(old[2:], new[2:], strict=True))
    assert poc.same_bits(new[1], np.ones(5, np.float32))
    rounded = np.array([1e8, 1e8, -1e8, 0, 16777216], np.float32)
    assert poc.same_bits(new[2], rounded)
    assert poc.same_bits(new[3], np.array([0, 0, 0, 1, 0], np.float32))
    assert poc.same_bits(new[4], rounded.view(np.uint32))
    assert poc.same_bits(new[5], np.array([1, 1.0009765625], np.float16))
    assert str(new[6].dtype) == "bfloat16"
    assert (
        np.asarray(new[6]).tobytes() == np.array([0x3F80, 0x3F81], np.uint16).tobytes()
    )
    assert poc.same_bits(new[7], np.array(116777216, np.float32))


def test_linear_configuration_reuses_actual_captured_weights_without_mutating_original():
    original, prepared = poc.linear_models()
    original_bytes = {
        key: np.asarray(getattr(original, key)[...]).tobytes()
        for key in ("kernel", "bias")
    }
    with poc.preserved_settings():
        feed = jnp.asarray(
            np.arange(256).reshape(2, 128) / 256 - 0.5, dtype=jnp.float32
        )
        baseline = original(feed)
        assert str(prepared(feed).dtype) == "float64"
        assert str(baseline.dtype) == "float32"
        assert poc.same_bits(original(feed), baseline)
        for key, shape in (("kernel", (128, 64)), ("bias", (64,))):
            old, new = getattr(original, key)[...], getattr(prepared, key)[...]
            assert old.shape == new.shape == shape
            assert old.dtype == jnp.float32 and new.dtype == jnp.float64
            assert poc.same_bits(new, np.asarray(old).astype(np.float64))
            assert np.asarray(old).tobytes() == original_bytes[key]


@pytest.mark.parametrize("raise_error", [False, True])
def test_five_hidden_and_effective_settings_restore_with_nested_contexts(raise_error):
    with ExitStack() as outer:
        for name, value in zip(
            poc.SETTING_NAMES, (False, "bfloat16", None, "strict", "raise"), strict=True
        ):
            outer.enter_context(jax_config.config_states[name](value))
        before = poc.snapshot()
        try:
            with poc.preserved_settings():
                assert poc.snapshot()["effective"] == poc.INTENDED_SETTINGS
                for name, value in zip(
                    poc.SETTING_NAMES,
                    (False, "default", None, "strict", "raise"),
                    strict=True,
                ):
                    jax.config.update(name, value)
                if raise_error:
                    raise RuntimeError("restoration probe")
        except RuntimeError as exc:
            assert raise_error and str(exc) == "restoration probe"
        assert poc.snapshot() == before


def test_fresh_ordinary_source_export_saved_actual_outputs_and_profile_barriers(
    tmp_path,
):
    output = tmp_path / "fresh-epoch"
    report = poc.run(output)
    assert report["status"] == "validated"
    assert report["before"] == report["after"]
    assert report["counts"] == {"exports": 12, "sessions": 4, "runs": 4}
    contract = json.loads((output / "contract.json").read_text())
    assert report["epoch"] == contract["epoch"]
    assert len(report["cases"]) == 12
    for name, expected in (
        ("exact_original", np.array([0, 0, 0, 1, 0], np.float32)),
        ("exact_prepared", np.ones(5, np.float64)),
    ):
        row = report["cases"][name]
        assert row["native"] == "validated" and row["export"] == "admitted"
        assert row["numerical_target_acceptance"] == "validated"
        assert len(row["runtime"]) == 2
        for target in row["runtime"]:
            assert target["artifact"] == row["artifact"]
            assert (
                target["runtime_execution"]
                == target["numerical_acceptance"]
                == "validated"
            )
            assert len(target["actual_outputs"]) == 1
            actual = np.load(
                output / target["actual_outputs"][0]["path"], allow_pickle=False
            )
            assert poc.same_bits(actual, expected)
            for mode in ("eager", "jit"):
                reference = np.load(output / row[mode][0]["path"], allow_pickle=False)
                assert poc.same_bits(actual, reference)
    barriers = {
        name: row
        for name, row in report["cases"].items()
        if row["export"] == "source-rejected"
    }
    assert len(barriers) == 8
    for name, row in barriers.items():
        assert row["native"] == "validated" and row["runtime"] == []
        assert row["numerical_target_acceptance"] is None
        barrier = row["first_barrier"]
        assert barrier["type"] == "SourcePrecisionError"
        assert barrier["fields"]["phase"] == "profile-validation"
        assert barrier["fields"]["category"] == "profile-exclusion"
        assert barrier["fields"]["primitive"] == (
            "dot_general"
            if name.startswith("linear_")
            else "jax.numpy.mean" if name.startswith("minimum_") else "jax.numpy.sum"
        )
    for name in ("exact_original_intermediates", "exact_prepared_intermediates"):
        assert report["cases"][name]["export"] == "admitted"
        assert report["cases"][name]["runtime"] == []
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(
        poc.digest((output / path).read_bytes()) == sha
        for path, sha in manifest.items()
    )


def test_output_must_be_new_and_outside_repository(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        poc.run(poc.ROOT / "generated-preparation-output")
    with pytest.raises(FileExistsError):
        poc.run(tmp_path)
