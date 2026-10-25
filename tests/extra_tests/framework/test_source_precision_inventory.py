# tests/extra_tests/framework/test_source_precision_inventory.py

"""Migration identities preserve metadata collisions and missing optional scope."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from scripts.source_precision_inventory import (
    DEFAULT_BASELINE,
    build_inventory,
    retained_missing_cells,
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
    for variable in (
        "JAX2ONNX_MAXTEXT_SRC",
        "JAX2ONNX_MAXTEXT_MODELS",
        "JAX2ONNX_MAXDIFFUSION_SRC",
        "JAX2ONNX_MAXDIFFUSION_MODELS",
    ):
        monkeypatch.delenv(variable, raising=False)
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


@pytest.mark.parametrize(
    "family,config_directory",
    [
        ("maxtext", "src/maxtext/configs/models"),
        ("maxdiffusion", "src/maxdiffusion/configs"),
    ],
)
def test_optional_config_discovery_retains_configured_external_and_local_sources(
    family: str,
    config_directory: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in (
        "JAX2ONNX_MAXTEXT_SRC",
        "JAX2ONNX_MAXTEXT_MODELS",
        "JAX2ONNX_MAXDIFFUSION_SRC",
        "JAX2ONNX_MAXDIFFUSION_MODELS",
    ):
        monkeypatch.delenv(variable, raising=False)
    repository = tmp_path / "repository"
    local_configs = repository / "tmp" / family / config_directory
    local_configs.mkdir(parents=True)
    (local_configs / "local.yml").write_text("local: true\n")
    external = tmp_path / "external" / family
    external_configs = external / config_directory
    external_configs.mkdir(parents=True)
    (external_configs / "external.yml").write_text("external: true\n")
    (external_configs / "other.yml").write_text("other: true\n")
    local_discovery = {
        "checkout": f"tmp/{family}",
        "present": True,
        "configs": ["local.yml"],
        "registration_exclusions": [],
    }
    unconfigured = optional_inventory(repository)[family]
    assert unconfigured["source_environment_configured"] is False
    assert unconfigured["model_filter"] == "<plugin-default>"
    assert unconfigured["discovery"] == [local_discovery]
    source_variable = f"JAX2ONNX_{family.upper()}_SRC"
    models_variable = f"JAX2ONNX_{family.upper()}_MODELS"
    monkeypatch.setenv(source_variable, str(external))
    monkeypatch.setenv(models_variable, "external")
    optional = optional_inventory(repository)
    configured = optional[family]
    assert configured["source_environment_configured"] is True
    assert configured["model_filter"] == "external"
    assert {item["checkout"]: item for item in configured["discovery"]} == {
        f"tmp/{family}": local_discovery,
        str(external): {
            "checkout": str(external),
            "present": True,
            "configs": ["external.yml", "other.yml"],
            "registration_exclusions": [],
        },
    }
    other_family = "maxdiffusion" if family == "maxtext" else "maxtext"
    assert optional[other_family]["source_environment_configured"] is False
    assert configured["source_audit"] == "not-run"
    assert configured["numerical_acceptance"] == "unverified"
    monkeypatch.delenv(source_variable)
    monkeypatch.delenv(models_variable)
    assert optional_inventory(repository)[family] == unconfigured


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


def test_generated_metadata_enrichment_preserves_raw_case_and_factory_identity() -> (
    None
):
    from jax2onnx.plugins.plugin_system import construct_and_call, with_requested_dtype
    from tests import t_generator

    def constructor(*, dtype):
        pytest.fail("Metadata extraction must not construct a model")

    recipe = construct_and_call(constructor, dtype=with_requested_dtype())
    values = [np.asarray([[1, 2], [3, 4], [5, 6]], np.float32)]
    raw = {
        "testcase": "factory",
        "description": "Original case description",
        "callable": recipe,
        "input_shapes": [("B", 2)],
        "input_values": values,
        "input_dtypes": [np.float32],
    }
    original = dict(raw)
    metadata = {
        "context": "primitives.synthetic",
        "component": "factory",
        "jax_doc": "jax documentation",
        "onnx": [{"component": "Identity"}],
        "source": "synthetic.py",
        "since": "0.1",
        "description": "Parent description for generated entries",
        "children": ["synthetic-child"],
        "testcases": [raw],
    }
    expected = {
        **original,
        **{key: value for key, value in metadata.items() if key != "testcases"},
    }
    first = t_generator.extract_from_metadata([metadata])[0]
    second = t_generator.extract_from_metadata([metadata])[0]
    assert first is not raw and second is not raw and first is not second
    assert raw == original
    assert raw["description"] == "Original case description"
    assert first == second == expected
    assert first["callable"] is second["callable"] is raw["callable"] is recipe
    assert first["input_values"] is second["input_values"] is values
    assert first["input_values"][0] is values[0]
    assert first["children"] is metadata["children"]

    variants = t_generator.generate_test_params(first)
    assert [entry["testcase"] for entry in variants] == [
        "factory_dynamic",
        "factory_dynamic_f64",
        "factory",
        "factory_f64",
    ]
    assert [entry["input_shapes"] for entry in variants] == [
        [("B", 2)],
        [("B", 2)],
        [(3, 2)],
        [(3, 2)],
    ]
    for entry, dtype in zip(
        variants, (np.float32, np.float64, np.float32, np.float64), strict=True
    ):
        assert entry["description"] == metadata["description"]
        assert entry["input_values"][0].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(entry["input_values"][0], values[0])
        contract = canonical_contract(entry["callable"], unresolved=[])
        assert contract["dtype"] == {"dtype": np.dtype(dtype).name}
    assert raw == original and raw["callable"] is recipe


def test_generated_collection_preserves_all_raw_contracts_and_exact_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax
    import onnxruntime
    from jax2onnx import user_interface
    from jax2onnx.plugins.plugin_system import construct_and_call
    from scripts import source_precision_probe as probe
    from scripts.source_precision_inventory import raw_registrations
    from tests import t_generator

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata collection/preflight must not trace or execute models")

    monkeypatch.setattr(jax, "make_jaxpr", forbidden)
    monkeypatch.setattr(user_interface, "_to_onnx", forbidden)
    monkeypatch.setattr(t_generator, "to_onnx", forbidden)
    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    monkeypatch.setattr(type(construct_and_call(forbidden)), "instantiate", forbidden)

    registrations, _ = raw_registrations()
    baseline = json.loads(probe.DEFAULT_BASELINE.read_text())
    plan = probe.build_probe_plan(baseline)
    cases = frozenset(probe.FIRST_DIAGNOSTIC_CASES)
    probe.preflight_cases(plan, registrations, cases)

    def snapshot():
        return {
            (origin, ordinal): (
                canonical_contract(case, unresolved=[]),
                id(case),
                id(case.get("callable")),
                id(case.get("input_values")),
                tuple(id(value) for value in case.get("input_values") or ()),
            )
            for origin, _, metadata in registrations
            for ordinal, case in enumerate(metadata.get("testcases", ()))
        }

    before = snapshot()
    baseline_identities = {(row.origin, row.ordinal) for row in plan.rows}
    assert len(baseline_identities) == 1852
    assert before.keys() <= baseline_identities
    assert baseline_identities - before.keys() <= {
        (origin, ordinal)
        for origin, condition in baseline["conditional_registrations"].items()
        for ordinal in condition.get("ordinals", [0])
    }
    extracted = t_generator.load_plugin_metadata()
    generated = [
        variant
        for entry in extracted
        for variant in t_generator.generate_test_params(entry)
    ]
    assert len(extracted) == len(before) and len(generated) > len(extracted)
    assert snapshot() == before
    probe.preflight_cases(plan, registrations, cases)
    assert (
        len(
            [
                cell
                for cell in plan.cells
                if probe.SourceProvenance(cell.origin, cell.ordinal) in cases
            ]
        )
        == 36
    )


@pytest.fixture
def optional_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Small stored recipes and config bytes; no native factory is evaluated."""
    from scripts import source_precision_inventory as inventory_module

    def git_result(command, **kwargs):
        if command[-1] == "--show-toplevel":
            return SimpleNamespace(stdout=str(Path(command[2]).parents[1]))
        return SimpleNamespace(stdout="1" * 40 if command[-1] == "HEAD" else "")

    monkeypatch.setattr(inventory_module.subprocess, "run", git_result)

    def create(
        root_name="first", *, family="maxtext", labels=("tiny", "other"), excluded=()
    ):
        root = tmp_path / root_name
        configs = root / (
            "src/maxtext/configs/models"
            if family == "maxtext"
            else "src/maxdiffusion/configs"
        )
        configs.mkdir(parents=True, exist_ok=True)
        for label in labels:
            (configs / (label + ".yml")).write_text("size: 4\n")
        for label in excluded:
            (configs / (label + ".yml")).write_text("excluded: true\n")
        if family == "maxtext":
            (configs.parent / "base.yml").write_text("dtype: float32\n")
        filters = {
            "default_models": [
                labels[0] + ".yml",
                *(label + ".yml" for label in excluded),
            ],
            "skip_patterns": ["quant"] if excluded else [],
        }
        source = (
            tmp_path
            / f"repository/jax2onnx/plugins/examples/{family}/{family}_models.py"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            f"DEFAULT_MODELS = {tuple(filters['default_models'])!r}\nskip_patterns = {filters['skip_patterns']!r}\n"
        )
        observations = {
            "source_environment_configured": True,
            "registration_available": True,
            "import_error": None,
            "resolved_checkout": str(root),
            "resolved_config_directory": str(configs),
            "registration_source_sha256": hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
            "model_filter": "all",
            "registration_filters": filters,
            "registration_source": f"jax2onnx/plugins/examples/{family}/{family}_models.py",
            "source_audit": "not-run",
            "numerical_acceptance": "unverified",
            "discovery": [],
        }
        registrations = []
        for label in labels:
            module = f"jax2onnx.plugins.examples.{family}.{family}_models"
            component = (
                ("MaxText" if family == "maxtext" else "MaxDiffusion") + "_" + label
            )
            kwargs = {"seed": 42}
            if family == "maxtext":
                kwargs["config_path"] = str(configs / (label + ".yml"))
            factory = (
                "get_maxtext_model_callable"
                if family == "maxtext"
                else "get_maxdiffusion_unet"
            )
            registrations.append(
                (
                    f"example::examples.{family}::{component}",
                    module,
                    {
                        "context": f"examples.{family}",
                        "component": component,
                        "testcases": [
                            {
                                "testcase": family + "_" + label,
                                "callable": {
                                    "factory": {"callable": module + "." + factory},
                                    "kwargs": kwargs,
                                },
                                "input_shapes": [(1, 4)],
                                "skip_numeric_validation": family == "maxtext",
                            }
                        ],
                    },
                )
            )
        return build_inventory(
            registrations, [], manual=[], optional={family: observations}
        )

    return create


def _omit_optional_cases(inventory, *, keep=frozenset(), unavailable=False):
    result = copy.deepcopy(inventory)
    original = result["registrations"]
    omitted = {
        row[0]
        for row in original
        if row[2] in {"examples.maxtext", "examples.maxdiffusion"}
        and row[0] not in keep
    }
    indices = [index for index, row in enumerate(original) if row[0] not in omitted]
    remap = {before: after for after, before in enumerate(indices)}
    kept_rows = [row for row in result["cases"] if row[0] in remap]
    result["registrations"] = [original[index] for index in indices]
    result["cases"] = [tuple([remap[row[0]], *row[1:]]) for row in kept_rows]
    result["conditional_registrations"] = {
        key: value
        for key, value in result["conditional_registrations"].items()
        if key not in omitted
    }
    for family, record in result["optional"].items():
        baseline = record["baseline"]
        baseline["members"] = [
            row for row in baseline["members"] if row[0] not in omitted
        ]
        baseline["configs"] = {
            key: value
            for key, value in baseline["configs"].items()
            if key.rsplit("::", 1)[0] not in omitted
        }
        record["registered_case_rows"] = len(baseline["members"])
        if unavailable:
            record.update(
                source_environment_configured=False,
                registration_available=False,
                resolved_checkout=None,
                resolved_config_directory=None,
            )
            baseline.update(ref=None, config_tree=[], configs={})
    return result


def test_all_fifty_optional_baseline_cells_survive_unconfigured_observation() -> None:
    expected = json.loads(DEFAULT_BASELINE.read_text())
    optional_origins = {
        origin
        for family in ("maxtext", "maxdiffusion")
        for origin, _ in expected["optional"][family]["baseline"]["members"]
    }
    assert len(optional_origins) == 50 and len(expected["cases"]) == 1852
    actual = _omit_optional_cases(expected, unavailable=True)
    before = copy.deepcopy(expected)
    differences, observations = compare_inventories(expected, actual)
    assert differences == []
    assert any("Retained 45 baseline maxtext" in item for item in observations)
    assert any("Retained 5 baseline maxdiffusion" in item for item in observations)
    assert len(retained_missing_cells(expected, actual)) == 50
    assert actual["registrations"] == [
        row for row in expected["registrations"] if row[0] not in optional_origins
    ]
    empty_index = next(
        index
        for index, row in enumerate(actual["registrations"])
        if row[0] == "plugin::name"
    )
    assert all(row[0] != empty_index for row in actual["cases"])
    del actual["registrations"][empty_index]
    actual["cases"] = [
        (row[0] - (row[0] > empty_index), *row[1:]) for row in actual["cases"]
    ]
    differences, _ = compare_inventories(expected, actual)
    assert differences == ["Missing registrations: plugin::name"]
    assert expected == before


def test_optional_selected_disappearance_and_default_filter_retention(
    optional_fixture,
) -> None:
    expected = optional_fixture()
    tiny = next(row[0] for row in expected["registrations"] if row[0].endswith("_tiny"))
    actual = _omit_optional_cases(expected, keep={tiny})
    differences, _ = compare_inventories(expected, actual)
    assert any("Missing selected optional case" in item for item in differences)
    actual["optional"]["maxtext"]["model_filter"] = "<plugin-default>"
    differences, observations = compare_inventories(expected, actual)
    assert differences == []
    assert any("model filter <plugin-default>" in item for item in observations)
    assert len(retained_missing_cells(expected, actual)) == 1


@pytest.mark.parametrize(
    "change",
    [
        "configured_missing",
        "import_error",
        "placeholder",
        "unknown_origin",
        "filter_policy",
        "unknown_filter",
        "condition",
    ],
)
def test_optional_failures_and_unknown_policies_remain_drift(
    optional_fixture, change
) -> None:
    expected = optional_fixture()
    actual = copy.deepcopy(expected)
    record = actual["optional"]["maxtext"]
    if change == "configured_missing":
        actual = _omit_optional_cases(actual)
        actual["optional"]["maxtext"]["registration_available"] = False
    elif change == "import_error":
        record["import_error"] = "configured dependency import failed"
    elif change in {"placeholder", "unknown_origin"}:
        row = actual["registrations"][0]
        actual["registrations"][0] = (
            row[0] + ("_Environment_Error" if change == "placeholder" else "_new"),
            *row[1:],
        )
    elif change == "filter_policy":
        record["registration_filters"]["skip_patterns"] = ["tiny"]
    elif change == "unknown_filter":
        record["model_filter"] = "not-a-baseline-model"
    else:
        origin = actual["registrations"][0][0]
        actual["conditional_registrations"][origin]["requires"] = "optional:unknown"
    differences, _ = compare_inventories(expected, actual)
    assert differences


def test_optional_config_relocation_preserves_distinct_raw_hashes(
    optional_fixture,
) -> None:
    expected = optional_fixture("first")
    actual = optional_fixture("second")
    differences, observations = compare_inventories(expected, actual)
    assert differences == []
    assert len(observations) == 2 and all(
        "Verified config relocation" in item for item in observations
    )
    first = expected["optional"]["maxtext"]["baseline"]["configs"]
    second = actual["optional"]["maxtext"]["baseline"]["configs"]
    assert all(
        first[key]["raw_contract_sha256"] != second[key]["raw_contract_sha256"]
        for key in first
    )
    assert retained_missing_cells(expected, actual) == set()


@pytest.mark.parametrize(
    "change",
    [
        "yaml",
        "base",
        "ref",
        "relative",
        "seed",
        "unrelated_string",
        "outside",
        "symlink",
    ],
)
def test_optional_portability_is_limited_to_verified_config_location(
    optional_fixture, change, tmp_path: Path
) -> None:
    expected = optional_fixture("first")
    actual = optional_fixture("second")
    record = actual["optional"]["maxtext"]["baseline"]
    key = next(iter(record["configs"]))
    case = next(
        row
        for row in actual["cases"]
        if actual["registrations"][row[0]][0] + "::0" == key
    )
    contract = actual["contracts"][case[5]]
    config = record["configs"][key]
    if change in {"yaml", "base"}:
        from scripts.source_precision_inventory import _optional_baseline

        observations = actual["optional"]["maxtext"]
        changed_file = (
            (Path(observations["resolved_config_directory"]).parent / "base.yml")
            if change == "base"
            else Path(config["observed_absolute_path"])
        )
        changed_file.write_text("changed: true\n")
        observations["baseline"] = _optional_baseline(
            "maxtext",
            observations,
            actual["registrations"],
            actual["cases"],
            actual["contracts"],
        )
        assert observations["baseline"]["config_tree"] != record["config_tree"]
    elif change == "ref":
        record["ref"] = "2" * 40
    elif change == "relative":
        config["relative_path"] = "elsewhere/" + Path(config["relative_path"]).name
    elif change in {"seed", "unrelated_string"}:
        contract["callable"]["kwargs"][
            "seed" if change == "seed" else "unrelated_path"
        ] = (43 if change == "seed" else str(tmp_path / "unrelated"))
        config["raw_contract_sha256"] = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    else:
        from scripts.source_precision_inventory import _optional_baseline

        outside = tmp_path / "outside.yml"
        outside.write_text("size: 4\n")
        if change == "symlink":
            linked = Path(config["observed_absolute_path"])
            linked.unlink()
            linked.symlink_to(outside)
        else:
            contract["callable"]["kwargs"]["config_path"] = str(outside)
        observations = actual["optional"]["maxtext"]
        record = _optional_baseline(
            "maxtext",
            observations,
            actual["registrations"],
            actual["cases"],
            actual["contracts"],
        )
        observations["baseline"] = record
        assert record["errors"]
    differences, _ = compare_inventories(expected, actual)
    assert differences


def test_maxdiffusion_labels_keep_distinct_identities_and_shared_raw_factory(
    optional_fixture,
) -> None:
    expected = optional_fixture(
        family="maxdiffusion", labels=("base", "v1", "v2", "v3", "v4")
    )
    assert len(expected["cases"]) == 5 and len(expected["contracts"]) == 1
    assert "config_path" not in expected["contracts"][0]["callable"]["kwargs"]
    actual = optional_fixture(
        "second", family="maxdiffusion", labels=("base", "v1", "v2", "v3", "v4")
    )
    assert compare_inventories(expected, actual) == ([], [])


def test_refresh_refuses_retained_loss_but_allows_complete_relocation(
    optional_fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import source_precision_inventory as inventory_module

    expected = optional_fixture()
    output = tmp_path / "baseline.json"
    output.write_text(inventory_json(expected))
    original_bytes = output.read_bytes()
    actual = _omit_optional_cases(expected, unavailable=True)
    monkeypatch.setattr(inventory_module, "collect_inventory", lambda: actual)
    monkeypatch.setattr(inventory_module, "unavailable_registrations", frozenset)
    assert inventory_module.main(["--output", str(output), "--refresh"]) == 1
    assert output.read_bytes() == original_bytes
    actual = copy.deepcopy(expected)
    actual["optional"]["maxtext"]["model_filter"] = "tiny"
    assert inventory_module.main(["--output", str(output), "--refresh"]) == 1
    assert output.read_bytes() == original_bytes
    actual = optional_fixture("relocated")
    assert inventory_module.main(["--output", str(output), "--refresh"]) == 0
    assert json.loads(output.read_text())["optional"]["maxtext"]["baseline"][
        "checkout"
    ].endswith("relocated")


@pytest.mark.parametrize(
    "state",
    [
        "unconfigured_missing",
        "unconfigured_import_error",
        "installed",
        "configured_error",
        "unconfigured_partial",
        "available_unverifiable",
    ],
)
def test_optional_resolver_state_matches_registration_outcome(
    optional_fixture, state, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from scripts import source_precision_inventory as inventory_module

    expected = optional_fixture()
    before = expected["optional"]["maxtext"]
    source = tmp_path / "repository" / before["registration_source"]
    available = state in {"installed", "available_unverifiable"}
    package = (
        None
        if state in {"unconfigured_missing", "available_unverifiable"}
        else Path(before["resolved_config_directory"]).parents[1]
    )
    module = SimpleNamespace(
        __file__=str(source),
        MAXTEXT_PKG_PATH=package,
        MODELS_DIR=(
            Path(before["resolved_config_directory"]) if package is not None else None
        ),
        MAXTEXT_AVAILABLE=available,
        _MAXTEXT_IMPORT_ERROR=(
            ImportError("missing optional helper dependency")
            if "error" in state or state == "unconfigured_partial"
            else None
        ),
    )
    monkeypatch.setitem(
        sys.modules, "jax2onnx.plugins.examples.maxtext.maxtext_models", module
    )
    monkeypatch.delenv("JAX2ONNX_MAXTEXT_SRC", raising=False)
    monkeypatch.setenv("JAX2ONNX_MAXTEXT_MODELS", "all")
    if state == "configured_error":
        monkeypatch.setenv("JAX2ONNX_MAXTEXT_SRC", before["resolved_checkout"])
    keep = {row[0] for row in expected["registrations"]} if available else set()
    if state == "unconfigured_partial":
        keep = {expected["registrations"][0][0]}
    actual = _omit_optional_cases(expected, keep=keep)
    observations = optional_inventory(tmp_path / "repository")["maxtext"]
    observations["registered_case_rows"] = len(actual["cases"])
    observations["baseline"] = inventory_module._optional_baseline(
        "maxtext",
        observations,
        actual["registrations"],
        actual["cases"],
        actual["contracts"],
    )
    actual["optional"] = {"maxtext": observations}
    differences, diagnostics = compare_inventories(expected, actual)
    if state in {"configured_error", "unconfigured_partial", "available_unverifiable"}:
        assert differences
    else:
        assert differences == []
        assert observations["source_environment_configured"] is False
        assert observations["registration_available"] is available
    if state == "unconfigured_import_error":
        assert any("missing optional helper dependency" in item for item in diagnostics)
        assert any("Retained 2 baseline maxtext" in item for item in diagnostics)
    if state == "installed":
        assert observations["resolved_checkout"] == before["resolved_checkout"]
        assert len(observations["baseline"]["members"]) == 2


@pytest.mark.parametrize(
    "selection",
    [
        "default_excluded",
        "explicit_excluded",
        "fabricated_exclusion",
        "stale_selected",
        "selected_omission",
    ],
)
def test_optional_filters_require_exact_recorded_labels_and_observed_selection(
    optional_fixture, selection
) -> None:
    expected = optional_fixture(excluded=("quant-real",))
    tiny = next(row[0] for row in expected["registrations"] if row[0].endswith("_tiny"))
    actual = _omit_optional_cases(expected, keep={tiny})
    if selection == "default_excluded":
        model_filter = "<plugin-default>"
    elif selection == "explicit_excluded":
        model_filter = "tiny,quant-real"
    elif selection == "fabricated_exclusion":
        model_filter = "tiny,nonexistent-quant"
    elif selection == "stale_selected":
        model_filter = "tiny"
        actual = copy.deepcopy(expected)
    else:
        model_filter = "all"
    actual["optional"]["maxtext"]["model_filter"] = model_filter
    differences, observations = compare_inventories(expected, actual)
    if selection in {"default_excluded", "explicit_excluded"}:
        assert differences == []
        assert any("Retained 1 baseline maxtext" in item for item in observations)
    else:
        assert differences
    if selection == "fabricated_exclusion":
        assert any("Unknown optional model filter" in item for item in differences)
    if selection == "stale_selected":
        assert any("Unexpected filtered optional cases" in item for item in differences)
