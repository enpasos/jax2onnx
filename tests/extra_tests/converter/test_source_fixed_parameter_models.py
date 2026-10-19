# tests/extra_tests/converter/test_source_fixed_parameter_models.py

"""Exact original model contracts and fresh first barriers, never support claims."""

from collections.abc import Iterator
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import user_interface as ui
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.precision import (
    CheckStatus,
    get_precision_report,
    precision_report_to_json,
)
from scripts.source_precision_inventory import canonical_contract, raw_registrations
from tests.extra_tests.converter.source_precision_support import (
    SourceSettings,
    _global_source_configuration,
)


# Reviewed immutable raw contract identities; no generated f32/f64 variants.
_CONTRACTS = (
    (
        "example::examples.gpt::GPT",
        0,
        "c5709eafd7a65c680110b5f8b351db49534522eabf7273eb31467eccb975246d",
    ),
    (
        "example::examples.gpt::GPT_Embeddings",
        0,
        "4b8fd79db25a4dbce61069df37d5da8d0cce81f5ddf1480295e92c775a15964d",
    ),
    (
        "example::examples.gpt::GPT_MLP",
        0,
        "faf258485b7a1bf4a97e1a56b87e9fb472ad8eba84b973b8e303dd925c69e21a",
    ),
    (
        "example::examples.gpt::GPT_TransformerBlock",
        0,
        "7dd3d5813b788033f8a1957f69c7c777b3a16035474724f1aa5747fd372f5ca4",
    ),
    (
        "example::examples.gpt::GPT_TransformerStack",
        0,
        "3c744db92fdc624842ffc5f1d08e6122a78b988d72bb2c2231f5c13d88e3041a",
    ),
    (
        "example::examples.linen::LinenMLP",
        2,
        "c302629899c4bbc09191f12419e672e5ffbb01b11d4af27653fe15d8ec6e892c",
    ),
    (
        "example::examples.nnx::MLP",
        2,
        "d3853463d2292865810e762ec68e580bb20e11f2220176af2c99da80b2b37163",
    ),
    (
        "example::examples.onnx_functions::onnx_functions_013",
        0,
        "c1b7edb2fd0ad67a4e7f729cccc8b1cdae54322224b4e27f5543a462377ed7b6",
    ),
    (
        "example::examples.onnx_functions::onnx_functions_014",
        0,
        "087c57edf57e96f20daae82e7cb733d919afb08bb865f8a998d826831076f7f2",
    ),
    (
        "example::examples.onnx_functions::onnx_functions_015",
        0,
        "147b3a5de49876d467d84acd4182502fb4de6125cc5f9c87b846145b9bcdf67e",
    ),
    (
        "example::examples.vit::TransformerBlock",
        0,
        "a8db4b1b02c0d08b0541684095c90aa2b05b199e7747180d6f08f162d2dee8d2",
    ),
    (
        "example::examples.vit::TransformerBlockFlatten",
        0,
        "2311a5123b155243546182795ac05a5885adc61a0ad1589a4d5838d546463e45",
    ),
    (
        "example::examples.vit::TransformerStack",
        0,
        "0d8c5ea4e90413ca221f67af527bb4d89a263003628561006b210a775448f338",
    ),
    (
        "example::examples.vit::TransformerStackFlatten",
        0,
        "2d3ee0b954a3b0f90f526c1869e19c4c528a8e0825ac095fe9cecf3d68aa0416",
    ),
    (
        "example::examples.vit::VisionTransformer",
        0,
        "b07c88dcdd1851761605375bc824ccf9e5dbb5e23e3e0d162f74eed92884def8",
    ),
    (
        "example::examples.vit::VisionTransformer",
        1,
        "a4524257941ee4aba8830a83a4a2f3bcc195472d5d627a52e67d52da50886509",
    ),
    (
        "example::examples.vit::VisionTransformerFlatten",
        0,
        "28b1abd03759d5d281267d270186049aaf6a2ce3ae1daaeb6a1806773875458e",
    ),
    (
        "example::examples.vit::VisionTransformerFlatten",
        1,
        "0dd340184236cd76db7206425d0cfeb77a38efa318ab80d4b73c834e9a55adc8",
    ),
    (
        "plugin::eqx.nn.dropout",
        2,
        "7f9d6b14ae54a436c7ff39fefedcd1f9b7d3a5bc4be6b539737ed27735c2d2c5",
    ),
    (
        "plugin::linen.dropout",
        1,
        "2e554d4de13a9cb58623ce4c54a8d69456d65e3da529593e92656e212a72b290",
    ),
    (
        "plugin::linen.spectral_norm",
        0,
        "f2d55351772070cb40ad41a8df27ff5aa3fd3005369224887177d5bbfcb9a80e",
    ),
    (
        "plugin::nnx.dropout",
        1,
        "6c7e00b4e31e5d0932bf442f774e8687a0c85364f5ca5ad3c928820cb6fe030d",
    ),
)


# Only this exact immutable raw contract now has an artifact-only expectation.
# Its None criterion and runtime/numerical UNVERIFIED statuses remain unchanged.
_EXPECTED_ARTIFACT_CONTRACT = (
    "example::examples.gpt::GPT_MLP",
    0,
    "faf258485b7a1bf4a97e1a56b87e9fb472ad8eba84b973b8e303dd925c69e21a",
)


@pytest.fixture(autouse=True)
def _configuration() -> Iterator[None]:
    previous = _global_source_configuration()
    try:
        yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _error(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        **{
            name: getattr(exc, name)
            for name in ("phase", "category", "primitive", "location")
            if hasattr(exc, name)
        },
    }


def _state(target: Any) -> tuple[dict[str, np.ndarray], str]:
    module = (
        target if isinstance(target, nnx.Module) else getattr(target, "_model", None)
    )
    if not isinstance(module, nnx.Module):
        return {}, "not-an-NNX-module"
    leaves, tree = jax.tree_util.tree_flatten(nnx.state(module))
    arrays = {}
    for index, value in enumerate(leaves):
        if not hasattr(value, "dtype"):
            continue
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jax.random.key_data(value)
        arrays[f"leaf{index}"] = np.array(value, copy=True)
    return arrays, json.dumps(
        {
            "tree": str(tree),
            "leaves": [
                {"dtype": str(value.dtype), "shape": list(value.shape)}
                for value in leaves
                if hasattr(value, "dtype")
            ],
            "typed_key_storage": "jax.random.key_data; original key dtype recorded per leaf",
        },
        sort_keys=True,
    )


def _instantiate(recipe: Any) -> Any:
    return (
        recipe.instantiate()
        if getattr(recipe, "__jax2onnx_factory__", False)
        else recipe
    )


def _compiled(target: Any, arguments: tuple[Any, ...], params: dict[str, Any]) -> Any:
    if isinstance(target, nnx.Module):
        return nnx.jit(lambda module, *xs: module(*xs, **params))(target, *arguments)
    if isinstance(getattr(target, "_model", None), nnx.Module):
        return nnx.jit(
            lambda module, *xs: type(target)(module, target._rngs)(*xs, **params)
        )(target._model, *arguments)
    return jax.jit(lambda *xs: target(*xs, **params))(*arguments)


@pytest.mark.parametrize("index", range(len(_CONTRACTS)))
def test_exact_original_parameter_model_first_barrier(
    index: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin, ordinal, expected_digest = _CONTRACTS[index]
    expected_artifact = (
        origin,
        ordinal,
        expected_digest,
    ) == _EXPECTED_ARTIFACT_CONTRACT
    registrations, _ = raw_registrations()
    metadata = next(item for name, _, item in registrations if name == origin)
    case = metadata["testcases"][ordinal]
    contract = canonical_contract(
        {
            key: value
            for key, value in case.items()
            if key not in {"testcase", "description"}
        },
        unresolved=[],
    )
    contract_blob = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(contract_blob).hexdigest() == expected_digest
    root = Path(os.environ.get("JAX2ONNX_NAMED_MODEL_EVIDENCE_DIR", tmp_path))
    folder = root / f"{index:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    before = SourceSettings.observe()
    record: dict[str, Any] = {
        "index": index,
        "origin": origin,
        "ordinal": ordinal,
        "testcase": case["testcase"],
        "contract": contract,
        "contract_sha256": expected_digest,
        "settings": asdict(before),
        "hidden_global_settings": _global_source_configuration(),
        "versions": {"python": platform.python_version(), "jax": jax.__version__},
        "criterion": None,
        "expected_export_outcome": (
            "artifact-checked" if expected_artifact else "first-barrier"
        ),
        "native_reference_acceptance": "UNVERIFIED",
        "runtime": "UNVERIFIED",
        "numerical_acceptance": "UNVERIFIED",
        "product_support": "UNVERIFIED",
        "source_sha256": {
            str(Path(module.__file__).resolve()): _digest(Path(module.__file__))
            for module in (ui, conversion_api)
        },
    }
    recipe = case["callable"]
    params = dict(case.get("input_params") or {})
    dtypes = case.get("input_dtypes", [np.float32] * len(case["input_shapes"]))
    specs = [
        jax.ShapeDtypeStruct(tuple(shape), dtype)
        for shape, dtype in zip(case["input_shapes"], dtypes, strict=True)
    ]
    concrete = [
        tuple(1 if isinstance(dim, str) else dim for dim in spec.shape)
        for spec in specs
    ]
    arguments = tuple(
        jnp.asarray(np.zeros(shape, dtype=spec.dtype))
        for shape, spec in zip(concrete, specs, strict=True)
    )
    np.savez(
        folder / "feeds.npz",
        **{f"in_{i}": np.asarray(value) for i, value in enumerate(arguments)},
        **{name: np.asarray(value) for name, value in params.items()},
    )
    record["feed_sha256"] = _digest(folder / "feeds.npz")
    record["native"] = {}
    # Each reference starts from a separately reconstructed identical factory state.
    # No patched Dropout implementation participates in these references.
    for mode in ("eager", "jit"):
        entry: dict[str, Any] = {}
        try:
            target = _instantiate(recipe)
            state, tree = _state(target)
            np.savez(folder / f"{mode}-state-before.npz", **state)
            entry["state_tree"] = tree
            output = (
                target(*arguments, **params)
                if mode == "eager"
                else _compiled(target, arguments, params)
            )
            leaves, output_tree = jax.tree_util.tree_flatten(jax.device_get(output))
            np.savez(
                folder / f"{mode}-reference.npz",
                **{f"y{i}": np.asarray(value) for i, value in enumerate(leaves)},
            )
            state, _ = _state(target)
            np.savez(folder / f"{mode}-state-after.npz", **state)
            entry.update(
                status="evaluated",
                output_tree=str(output_tree),
                reference_sha256=_digest(folder / f"{mode}-reference.npz"),
                state_before_sha256=_digest(folder / f"{mode}-state-before.npz"),
                state_after_sha256=_digest(folder / f"{mode}-state-after.npz"),
            )
        except Exception as exc:
            entry.update(status="original-error", error=_error(exc))
        record["native"][mode] = entry
    try:
        target = _instantiate(recipe)
        declarations = [
            conversion_api._source_named_input_spec(name, value, opset=23)
            for name, value in params.items()
        ]
        # Concrete shapes isolate the original Python/runtime prerequisite from B admission.
        native_specs = [
            jax.ShapeDtypeStruct(shape, spec.dtype)
            for shape, spec in zip(concrete, specs, strict=True)
        ]

        def original(*values: Any) -> Any:
            return target(
                *values[: len(specs)],
                **dict(zip(params, values[len(specs) :], strict=True)),
            )

        if isinstance(target, nnx.Module):
            native_shape = nnx.eval_shape(
                lambda module, *values: module(
                    *values[: len(specs)],
                    **dict(zip(params, values[len(specs) :], strict=True)),
                ),
                target,
                *native_specs,
                *declarations,
            )
            record["native_dynamic_trace"] = {
                "status": "traced",
                "transform": "nnx.eval_shape graph argument",
                "output": str(native_shape),
            }
        elif isinstance(getattr(target, "_model", None), nnx.Module):
            native_shape = nnx.eval_shape(
                lambda module, *values: type(target)(module, target._rngs)(
                    *values[: len(specs)],
                    **dict(zip(params, values[len(specs) :], strict=True)),
                ),
                target._model,
                *native_specs,
                *declarations,
            )
            record["native_dynamic_trace"] = {
                "status": "traced",
                "transform": "nnx.eval_shape original bridge graph argument",
                "output": str(native_shape),
            }
        else:
            native_trace = jax.make_jaxpr(original)(*native_specs, *declarations)
            record["native_dynamic_trace"] = {
                "status": "traced",
                "transform": "jax.make_jaxpr pure callable",
                "root_primitives": [
                    eqn.primitive.name for eqn in native_trace.jaxpr.eqns
                ],
            }
    except Exception as exc:
        record["native_dynamic_trace"] = {
            "status": "original-error",
            "error": _error(exc),
        }
    record["native_reference_domain"] = (
        "zero tensors; symbolic batch concretized to 1; progress-only, no numerical acceptance"
    )
    if origin in {"plugin::eqx.nn.dropout", "plugin::linen.spectral_norm"}:
        alternate = (
            {"inference": np.asarray(False)}
            if origin == "plugin::eqx.nn.dropout"
            else {"update_stats": True}
        )
        try:
            _instantiate(recipe)(*arguments, **alternate)
        except Exception as exc:
            record["retained_original_runtime_prerequisite"] = {
                "arguments": {key: bool(value) for key, value in alternate.items()},
                "error": _error(exc),
            }
        else:
            pytest.fail("Original missing-key/mutable-state prerequisite disappeared")
    # Freeze exact original references, errors, state, inputs and None criterion
    # before attempting the exported computation.
    (folder / "prefreeze.json").write_text(json.dumps(record, sort_keys=True, indent=2))
    record["prefreeze_sha256"] = _digest(folder / "prefreeze.json")
    traces = []
    real_trace = conversion_api._trace_to_jaxpr

    def observe(**kwargs: Any) -> Any:
        traced = real_trace(**kwargs)
        traces.append(traced)
        return traced

    monkeypatch.setattr(conversion_api, "_trace_to_jaxpr", observe)
    try:
        target = _instantiate(recipe)
        export_params = params
        if origin == "plugin::linen.spectral_norm":
            # The original supported contract intentionally fixes this setting.
            target = partial(target, update_stats=False)
            export_params = {}
            record["external_preparation"] = "partial(original, update_stats=False)"
        artifact = ui._to_onnx(
            target,
            specs,
            input_params=export_params,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="proto",
            opset=23,
        )
        payload = artifact.SerializeToString()
        (folder / "model.onnx").write_bytes(payload)
        onnx.checker.check_model(onnx.load_model_from_string(payload), full_check=True)
        report = get_precision_report(artifact)
        assert report is not None and report.profile == "source-static-v1"
        (folder / "report.json").write_text(precision_report_to_json(report))
        if expected_artifact:
            assert report.opset == 23
            checks = {check.category: check.status for check in report.checks}
            assert checks["representation"] is CheckStatus.VALIDATED
            assert checks["lowering"] is CheckStatus.VALIDATED
            assert checks["runtime-execution"] is CheckStatus.UNVERIFIED
            assert checks["numerical-acceptance"] is CheckStatus.UNVERIFIED
            assert len(report.inputs) == 2
            for summary, dtype, shape in zip(
                report.inputs,
                ("float32", "bool"),
                (("B", 1024, 768), ()),
                strict=True,
            ):
                for properties in (
                    summary.supplied,
                    summary.normalized,
                    summary.traced,
                    summary.exported,
                ):
                    assert properties.dtype == dtype and properties.shape == shape
            assert len(report.traced_outputs) == len(report.exported_outputs) == 1
            for properties in (*report.traced_outputs, *report.exported_outputs):
                assert properties.dtype == "float32"
                assert properties.shape == ("B", 1024, 768)
            assert [value.name for value in artifact.graph.input] == [
                "in_0",
                "deterministic",
            ]
            assert len(artifact.graph.output) == 1
            for value in (artifact.graph.input[0], artifact.graph.output[0]):
                tensor = value.type.tensor_type
                assert tensor.elem_type == onnx.TensorProto.FLOAT
                assert tensor.shape.dim[0].dim_param == "B"
                assert tuple(dim.dim_value for dim in tensor.shape.dim[1:]) == (
                    1024,
                    768,
                )
            control = artifact.graph.input[1].type.tensor_type
            assert control.elem_type == onnx.TensorProto.BOOL
            assert not control.shape.dim
        record["export"] = {
            "status": "artifact-checked",
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "inputs": [value.name for value in artifact.graph.input],
        }
    except Exception as exc:
        record["export"] = {"status": "first-barrier", "error": _error(exc)}
    record["traces"] = [
        {
            "root_primitives": [eqn.primitive.name for eqn in trace.jaxpr.eqns],
            "frozen_params": trace.frozen_params,
            "bindings": [
                {
                    "trace_index": binding.trace_index,
                    "keyword_name": binding.keyword_name,
                    "positional_index": binding.positional_index,
                    "dtype": np.dtype(binding.spec.dtype).name,
                    "shape": [str(dim) for dim in binding.spec.shape],
                    "weak_type": binding.spec.weak_type,
                    "actual_var_identity_matches": binding.var
                    is trace.jaxpr.invars[binding.trace_index],
                }
                for binding in trace.source_inputs
            ],
        }
        for trace in traces
    ]
    assert SourceSettings.observe() == before
    assert all(
        binding["actual_var_identity_matches"]
        for trace in record["traces"]
        for binding in trace["bindings"]
    )
    (folder / "ledger.json").write_text(json.dumps(record, sort_keys=True, indent=2))
    assert record["criterion"] is None
    assert all(
        record[field] == "UNVERIFIED"
        for field in (
            "native_reference_acceptance",
            "runtime",
            "numerical_acceptance",
            "product_support",
        )
    )
    assert record["export"]["status"] == record["expected_export_outcome"]
    if expected_artifact:
        assert record["export"]["inputs"] == ["in_0", "deterministic"]
        assert "error" not in record["export"]
    else:
        # Every other original contract retains its required first barrier.
        assert record["export"]["error"]["type"] in {
            "SourcePrecisionError",
            "TracerBoolConversionError",
            "TracerIntegerConversionError",
            "ValueError",
            "TypeError",
        }


def test_original_mlp_is_next_dependency_and_static_014_stays_five() -> None:
    assert _CONTRACTS[6][:2] == ("example::examples.nnx::MLP", 2)
    registrations, _ = raw_registrations()
    metadata = next(item for name, _, item in registrations if name == _CONTRACTS[8][0])
    assert metadata["testcases"][0]["input_shapes"] == [(5, 10, 3)]
