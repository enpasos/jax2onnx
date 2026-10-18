# tests/extra_tests/framework/test_source_precision_inventory.py

"""Migration identities preserve metadata collisions and missing optional scope."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.source_precision_inventory import (
    build_inventory,
    canonical_contract,
    compare_inventories,
    handwritten_inventory,
    inventory_json,
    optional_inventory,
)


def test_inventory_preserves_aliases_and_colliding_case_names() -> None:
    first = {
        "context": "primitives.test",
        "component": "add",
        "testcases": [
            {"testcase": "same"},
            {"testcase": "same", "input_params": {"scale": 2}},
        ],
    }
    alias = {
        "context": "primitives.test",
        "component": "add",
        "testcases": [{"testcase": "same"}],
    }
    inventory = build_inventory(
        [
            ("plugin::add_alias", "sample.alias", alias),
            ("plugin::add", "sample.add", first),
        ],
        [("onnx_fn::body", "sample.function")],
        manual=[],
        optional={},
    )
    assert inventory["counts"]["raw_case_rows"] == 3
    assert inventory["counts"]["distinct_case_names"] == 1
    assert inventory["counts"]["registry_names"] == 3
    assert inventory["cases"][0][:3] == (0, 0, "same")
    assert inventory["cases"][1][:3] == (0, 1, "same")
    assert inventory["cases"][2][:3] == (1, 0, "same")
    assert inventory["name_collisions"] == [
        (("primitives.test", "add", "same"), [0, 1, 2])
    ]
    assert inventory["source_audit"] == "not-run"
    assert inventory["numerical_acceptance"] == "unverified"


def test_inventory_serialization_is_deterministic_and_compact() -> None:
    row = (
        "plugin::add",
        "sample.add",
        {"component": "add", "testcases": [{"testcase": "add"}]},
    )
    inventory = build_inventory(
        [row], [], manual=[("tests/manual.py", ["test_b", "test_a"])], optional={}
    )
    payload = inventory_json(inventory)
    assert payload == inventory_json(inventory)
    assert '    [0,0,"add",0,23,0]' in payload
    assert json.loads(payload)["manual"] == [["tests/manual.py", ["test_a", "test_b"]]]


def test_optional_config_discovery_retains_absent_environment_and_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JAX2ONNX_MAXTEXT_SRC", raising=False)
    monkeypatch.delenv("JAX2ONNX_MAXTEXT_MODELS", raising=False)
    config = tmp_path / "tmp/maxtext/src/maxtext/configs/models"
    config.mkdir(parents=True)
    (config / "llama.yml").write_text("")
    (config / "quant-model.yml").write_text("")
    source = tmp_path / "jax2onnx/plugins/examples/maxtext/maxtext_models.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        'DEFAULT_MODELS = ("llama.yml",)\ndef _register_examples():\n    skip_patterns = ["quant"]\n'
    )
    optional = optional_inventory(tmp_path)
    maxtext = optional["maxtext"]
    assert maxtext["source_environment_configured"] is False
    assert maxtext["discovery"][0]["configs"] == ["llama.yml", "quant-model.yml"]
    assert maxtext["discovery"][0]["registration_exclusions"] == ["quant-model.yml"]
    assert maxtext["registration_filters"]["default_models"] == ["llama.yml"]
    assert optional["maxdiffusion"]["discovery"][0]["present"] is False
    assert maxtext["source_audit"] == "not-run"


def test_manual_inventory_is_separate_from_generated_entrypoints(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "tests/extra_tests"
    directory.mkdir(parents=True)
    (directory / "test_boundary.py").write_text(
        "def test_boundary():\n    pass\nclass TestNames:\n    def test_alias(self):\n        pass\n"
    )
    generated = tmp_path / "tests/primitives"
    generated.mkdir()
    (generated / "test_fake.py").write_text("def test_generated():\n    pass\n")
    assert handwritten_inventory(tmp_path) == [
        (
            "tests/extra_tests/test_boundary.py",
            ("TestNames.test_alias", "test_boundary"),
        )
    ]


def test_semantic_compare_keeps_known_version_conditional_baseline_cells() -> None:
    metadata = {
        "component": "one_minus_square",
        "testcases": [{"testcase": "near_one"}],
    }
    expected = build_inventory(
        [("plugin::one_minus_square", "sample", metadata)], [], manual=[], optional={}
    )
    actual = build_inventory([], [], manual=[], optional={})
    differences, observations = compare_inventories(
        expected, actual, unavailable=frozenset({"plugin::one_minus_square"})
    )
    assert differences == []
    assert len(observations) == 1 and "Retained baseline" in observations[0]
    assert len(expected["cases"]) == 1
    # A disappearing registration with its JAX feature still present is a drift.
    differences, _ = compare_inventories(expected, actual)
    assert "Missing registrations: plugin::one_minus_square" in differences


def test_semantic_compare_detects_case_loss_without_index_renumber_noise() -> None:
    left = ("plugin::a", "sample.a", {"testcases": [{"testcase": "a"}]})
    right = ("plugin::b", "sample.b", {"testcases": [{"testcase": "b"}]})
    expected = build_inventory([left, right], [], manual=[], optional={})
    actual = build_inventory([right], [], manual=[], optional={})
    differences, _ = compare_inventories(expected, actual)
    assert differences == [
        "Missing registrations: plugin::a",
        "Missing cases: plugin::a::0",
    ]


@pytest.mark.parametrize(
    "field,old,new",
    [
        ("input_dtypes", ["float64"], ["float32"]),
        ("input_shapes", [(2, 3)], [(1,)]),
        ("rtol", 1e-5, 1e-2),
        ("input_params", {"deterministic": True}, {"deterministic": False}),
    ],
)
def test_contract_changes_cannot_hide_behind_same_testcase_identity(
    field, old, new
) -> None:
    def inventory(value):
        case = {"testcase": "same", field: value}
        return build_inventory(
            [("plugin::a", "sample", {"testcases": [case]})], [], manual=[], optional={}
        )

    differences, _ = compare_inventories(inventory(old), inventory(new))
    assert differences == ["Changed cases: plugin::a::0"]


def test_typed_domain_payload_and_unresolved_recipes_are_explicit() -> None:
    import numpy as np

    class Opaque:
        pass

    def inventory(value):
        case = {"testcase": "domain", "input_values": [value], "callable": Opaque()}
        return build_inventory(
            [("plugin::a", "sample", {"testcases": [case]})], [], manual=[], optional={}
        )

    before = inventory(np.asarray([1], np.float64))
    after = inventory(np.asarray([1], np.float32))
    differences, _ = compare_inventories(before, after)
    assert differences == ["Changed cases: plugin::a::0"]
    assert before["contracts"][0]["input_values"][0]["array_dtype"] == "float64"
    assert len(before["contracts"][0]["input_values"][0]["sha256"]) == 64
    assert before["unresolved_recipes"][0][0] == 0


def test_factory_contract_fingerprinting_never_instantiates_models() -> None:
    import numpy as np
    from jax2onnx.plugins.plugin_system import construct_and_call, with_requested_dtype

    calls = []

    def constructor(*, dtype):
        calls.append(dtype)
        raise AssertionError("Inventory must not instantiate this model")

    def inventory(dtype):
        recipe = construct_and_call(
            constructor, dtype=with_requested_dtype()
        ).with_dtype(dtype)
        return build_inventory(
            [
                (
                    "plugin::a",
                    "sample",
                    {"testcases": [{"testcase": "a", "callable": recipe}]},
                )
            ],
            [],
            manual=[],
            optional={},
        )

    before, after = inventory(np.float64), inventory(np.float32)
    assert not calls
    assert compare_inventories(before, after)[0] == ["Changed cases: plugin::a::0"]
    assert before["contracts"][0]["callable"]["dtype"] == {"dtype": "float64"}


def test_numpy_scalar_contract_preserves_typed_storage_and_nan_payloads() -> None:
    unresolved: list[str] = []

    def contract(value):
        return canonical_contract(value, unresolved=unresolved)

    assert contract(1.0) != contract(np.float64(1.0))
    assert contract(np.float32(1.0)) != contract(np.float64(1.0))
    assert contract(np.float64(0.0)) != contract(np.float64(-0.0))
    nan_values = np.asarray(
        [0x7FF8000000000001, 0x7FF8000000000002, 0xFFF8000000000001],
        dtype=np.uint64,
    ).view(np.float64)
    assert np.isnan(nan_values).all()
    assert (
        len({json.dumps(contract(value), sort_keys=True) for value in nan_values}) == 3
    )
    for value in nan_values:
        assert contract(value) == {
            "scalar_dtype": "float64",
            "dtype_str": value.dtype.str,
            "bytes_hex": value.tobytes().hex(),
        }
    assert not unresolved


def test_inventory_compare_detects_python_to_typed_scalar_and_storage_changes() -> None:
    def inventory(value):
        return build_inventory(
            [
                (
                    "plugin::a",
                    "sample",
                    {"testcases": [{"testcase": "scalar", "input_values": [value]}]},
                )
            ],
            [],
            manual=[],
            optional={},
        )

    nan_values = np.asarray(
        [0x7FF8000000000001, 0x7FF8000000000002], dtype=np.uint64
    ).view(np.float64)
    for before, after in (
        (1.0, np.float64(1.0)),
        (np.float64(0.0), np.float64(-0.0)),
        tuple(nan_values),
    ):
        differences, _ = compare_inventories(inventory(before), inventory(after))
        assert differences == ["Changed cases: plugin::a::0"]
