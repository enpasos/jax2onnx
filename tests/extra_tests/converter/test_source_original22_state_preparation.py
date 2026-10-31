# tests/extra_tests/converter/test_source_original22_state_preparation.py

"""Supplementary finite follow-through of the 22 immutable original owners.

The original raw criteria and first-barrier ledger remain unchanged. New FP32-input
sample-family bounds below are candidates declared before any target observation;
they do not inherit criterion67cd or the six existing hand-authored MLP bounds.

Direct Dropout masks are observable and checked exactly. Full-owner internal
masks are not exposed by the native return contract: their direct equality is
UNVERIFIED here. Existing source PRNG/schema and Function/If/Loop audits remain
separate required checks. This module adds no graph instrumentation controller.
Authoring this candidate does not establish native, export or CPU acceptance.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
import hashlib
import json
from pathlib import Path
from types import FunctionType
from typing import Any
from unittest.mock import patch

from flax import nnx
from flax.errors import ModifyScopeVariableError
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
from onnxruntime.capi import _pybind_state  # type: ignore[import-untyped]
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat.jax import native_transformed_execution_context
from jax2onnx._source_state_preparation import (
    PreparedStatefulCall,
    prepare_linen_call,
    prepare_native_control_call,
    prepare_nnx_call,
    prepare_pure_control_call,
)
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.precision import CheckStatus, get_precision_report
from scripts.source_precision_inventory import canonical_contract
from tests.extra_tests.converter.source_eqx18_default_anchor import Eqx18DefaultAnchor
from tests.extra_tests.converter.source_original22_phase_observation import (
    phase as _phase,
)
from tests.extra_tests.converter.source_precision_support import (
    SourceProvenance,
    _input_data,
    raw_source_case,
    raw_source_scenarios,
    source_configuration,
)
from tests.extra_tests.converter.test_source_dropout_parameters import DropoutFlow
from tests.extra_tests.converter.test_source_fixed_parameter_models import (
    _CONTRACTS,
    _instantiate,
)
from tests.extra_tests.converter.test_source_mlp_runtime_acceptance import (
    _port,
    _runtime_configuration,
)

_FLAGS = (True, False, False, True)
_ROLES = ("original-eager", "original-jit-first", "same-original-jit-warmed")
_NNX = (*range(6, 18), 21)
_LINEN = (5, 19)
_GPT = tuple(range(5))
_DIRECT_DROPOUT = (18, 19, 21)
# Declared once, before model construction or observations. These are new finite
# sample criteria; no raw scenario or historical acceptance criterion is changed.
# They cover the original FP32 input families in both source contexts; an
# original FP64 result from x64-on construction retains its exact native dtype.
_FAMILY_CRITERIA = {
    "zero-rate-gpt": (1e-5, 1e-5),
    "raw-generated-mlp": (1e-5, 1e-5),
    "original-nested-functions": (1e-5, 1e-5),
    "original-vit": (1e-5, 1e-5),
    "original-dropout": (1e-5, 1e-5),
    "static-spectral-norm": (1e-5, 1e-5),
}
_CHUNK = 1 << 18


def _family(index: int) -> str:
    if index in _GPT:
        return "zero-rate-gpt"
    if index in (5, 6):
        return "raw-generated-mlp"
    if index in (7, 8, 9):
        return "original-nested-functions"
    if index in _DIRECT_DROPOUT:
        return "original-dropout"
    return "static-spectral-norm" if index == 20 else "original-vit"


def _contract(case: Any, *, default_anchor: Eqx18DefaultAnchor | None = None) -> str:
    unresolved: list[str] = []
    value = canonical_contract(
        {
            key: value
            for key, value in case.items()
            if key not in {"testcase", "description", "context", "component"}
        },
        unresolved=unresolved,
    )
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(blob).hexdigest()
    if default_anchor is None:
        assert not unresolved
    else:
        default_anchor.validate_raw(case, digest, unresolved)
    return digest


def _host(value: Any) -> np.ndarray:
    return np.array(jax.device_get(value), copy=True)


def _physical(value: Any) -> np.ndarray:
    return _host(
        jax.random.key_data(value)
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
        else value
    )


def _hash(value: Any) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(repr(value.shape).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _exact(actual: Any, reference: Any) -> None:
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.dtype == reference.dtype and actual.shape == reference.shape
    assert actual.tobytes() == reference.tobytes()


def _same_words(actual: Any, reference: Any) -> None:
    assert len(actual) == len(reference)
    for value, expected in zip(actual, reference, strict=True):
        assert np.asarray(value).dtype == np.asarray(expected).dtype == np.uint32
        _exact(value, expected)


def _module(target: Any) -> Any:
    return target if isinstance(target, nnx.Module) else getattr(target, "_model", None)


def _state_entries(target: Any, selection: Any = ...) -> dict[tuple[Any, ...], Any]:
    """Independent native observation; no production state accessor is used."""
    module = _module(target)
    if module is None:
        return {}
    assert isinstance(module, nnx.Module)
    return dict(nnx.to_flat_state(nnx.state(module, selection)))


def _snapshot(target: Any, selection: Any = ...) -> tuple[Any, ...]:
    entries = []
    for path, variable in _state_entries(target, selection).items():
        leaves, tree = jax.tree.flatten(variable)
        values = []
        for leaf in leaves:
            assert not isinstance(leaf, jax.core.Tracer)
            key = jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            values.append(
                (
                    str(leaf.dtype),
                    tuple(leaf.shape),
                    str(jax.random.key_impl(leaf)) if key else None,
                    _hash(_physical(leaf)),
                )
            )
        entries.append((path, str(tree), tuple(values)))
    # The actual Linen bridge key is a fixed binding, not an advancing stream or
    # an invented additional RNG state port.
    if hasattr(target, "_rngs"):
        entries.append(("fixed-bridge-key", _hash(_physical(target._rngs))))
    return tuple(entries)


def _identities(target: Any) -> tuple[Any, ...]:
    module = _module(target)
    if module is None:
        return ()
    return tuple(
        (tuple(path), id(value))
        for path, value in nnx.iter_graph(module)
        if isinstance(value, (nnx.Module, nnx.Variable))
    )


def _observed_words(target: Any, state_specs: Any) -> tuple[np.ndarray, ...]:
    entries = _state_entries(target, nnx.RngState)
    covered: set[tuple[Any, ...]] = set()
    words = []
    for spec in state_specs:
        # Production declares the port paths; values and alias equality come
        # directly from independent native state, including its real key impl.
        paths = tuple(tuple(path) for path in spec.paths)
        assert paths
        present = [path for path in paths if path in entries]
        assert present, f"No independent native RNG entry for {paths!r}"
        aliases = []
        for path in present:
            leaves = jax.tree.leaves(entries[path])
            assert len(leaves) == 1
            aliases.append(_physical(leaves[0]))
        for alias in aliases[1:]:
            _exact(alias, aliases[0])
        value = aliases[0]
        assert value.dtype == np.uint32
        assert tuple(value.shape) == tuple(spec.shape)
        assert value.dtype == np.dtype(spec.dtype)
        words.append(value)
        covered.update(present)
    assert covered == set(entries), "Every original RNG stream must be explicit"
    return tuple(words)


def _finite(value: Any) -> None:
    value = np.asarray(value)
    assert value.dtype in (np.dtype(np.float32), np.dtype(np.float64))
    flat = value.reshape(-1)
    for start in range(0, flat.size, _CHUNK):
        assert np.isfinite(flat[start : start + _CHUNK]).all()


def _outputs(value: Any) -> tuple[tuple[np.ndarray, ...], Any]:
    leaves, tree = jax.tree.flatten(value)
    arrays = tuple(_host(leaf) for leaf in leaves)
    assert arrays
    for array in arrays:
        _finite(array)
    return arrays, tree


def _compare(actual: Any, expected: Any, index: int) -> None:
    assert len(actual) == len(expected)
    rtol, atol = _FAMILY_CRITERIA[_family(index)]
    for value, reference in zip(actual, expected, strict=True):
        value, reference = np.asarray(value), np.asarray(reference)
        assert value.dtype == reference.dtype
        assert value.shape == reference.shape
        _finite(value)
        _finite(reference)
        a, r = value.reshape(-1), reference.reshape(-1)
        for start in range(0, a.size, _CHUNK):
            actual64 = a[start : start + _CHUNK].astype(np.float64)
            expected64 = r[start : start + _CHUNK].astype(np.float64)
            assert np.all(
                np.abs(actual64 - expected64) <= atol + rtol * np.abs(expected64)
            ), "New a priori family sample bound failed"


def _direct_mask(index: int, flag: bool, x: Any, actual: Any, expected: Any) -> None:
    if index not in _DIRECT_DROPOUT:
        return
    assert len(actual) == len(expected) == 1
    value, reference = np.asarray(actual[0]), np.asarray(expected[0])
    x = np.asarray(x)
    assert x.shape == value.shape == reference.shape and np.any(x != 0)
    # Here, and only here, the original result directly exposes the whole mask.
    # Check mask/control selection independently of the finite numeric bound.
    _exact(value == 0, reference == 0)
    if flag:
        _exact(value, x)
    else:
        nonzero = x != 0
        assert np.any(value[nonzero] == 0) and np.any(value[nonzero] != 0)
    _exact(value, reference)


@dataclass(frozen=True)
class _Step:
    output_paths: tuple[Path, ...]
    tree: Any
    before: tuple[np.ndarray, ...]
    after: tuple[np.ndarray, ...]


def _store_outputs(folder: Path, values: Any) -> tuple[Path, ...]:
    # Content addressing avoids retaining several full GPT outputs in RAM or
    # writing identical zero-rate/eager/warmed outputs repeatedly.
    folder.mkdir(exist_ok=True)
    paths = []
    for value in values:
        path = folder / f"{_hash(value)}.npy"
        if not path.exists():
            np.save(path, value, allow_pickle=False)
        paths.append(path)
    return tuple(paths)


def _load_outputs(step: _Step) -> tuple[np.ndarray, ...]:
    return tuple(
        np.load(path, allow_pickle=False, mmap_mode="r") for path in step.output_paths
    )


def _parameter(index: int) -> str:
    return "inference" if index == 18 else "deterministic"


def _flags(index: int) -> tuple[Any, ...]:
    return (None,) if index == 20 else _FLAGS


def _bound_original(
    case: Any, index: int, key: Any, *, default_anchor: Eqx18DefaultAnchor | None = None
) -> Any:
    original = _instantiate(case["callable"])
    if index != 18:
        assert default_anchor is None
        return original
    assert default_anchor is not None
    # Retain the actual raw lambda and its exact default Dropout model. The key
    # is an explicit caller binding; no adapter-created counter/split is used.
    assert isinstance(original, FunctionType) and original.__name__ == "<lambda>"
    target = partial(original, key=key)
    assert target.func is original and target.keywords["key"] is key
    assert target.func.__defaults__ is original.__defaults__
    default_anchor.bind(case, target, key)
    return target


def _jit_original(index: int, template: Any) -> Callable[..., Any]:
    """Compile actual native owners, independently of every preparation method."""
    if index == 18:

        def pure(x: Any, *, flag: bool) -> Any:
            return template(x, inference=flag)

        compiled = jax.jit(pure, static_argnames=("flag",))
        return lambda target, x, flag: compiled(x, flag=flag)
    if isinstance(template, nnx.Module):

        def native(module: Any, x: Any, *, flag: bool) -> Any:
            return module(x, deterministic=flag)

        compiled = nnx.jit(native, static_argnames=("flag",))
        return lambda target, x, flag: compiled(target, x, flag=flag)
    bridge_type = type(template)
    assert isinstance(template._model, nnx.Module)
    if index == 20:

        def spectral(module: Any, rngs: Any, x: Any) -> Any:
            return bridge_type(module, rngs)(x, update_stats=False)

        compiled = nnx.jit(spectral)
        return lambda target, x, flag: compiled(target._model, target._rngs, x)

    def bridge(module: Any, rngs: Any, x: Any, *, flag: bool) -> Any:
        return bridge_type(module, rngs)(x, deterministic=flag)

    compiled = nnx.jit(bridge, static_argnames=("flag",))
    return lambda target, x, flag: compiled(target._model, target._rngs, x, flag=flag)


def _eager_original(index: int, target: Any, x: Any, flag: Any) -> Any:
    if index == 20:
        return target(x, update_stats=False)
    return target(x, **{_parameter(index): flag})


def _native_references(
    case: Any,
    index: int,
    key: Any,
    x: Any,
    prepared: Any,
    baseline: Any,
    folder: Path,
    *,
    default_anchor: Eqx18DefaultAnchor | None = None,
) -> tuple[dict[str, tuple[_Step, ...]], list[Any]]:
    """First/warmed roles share one compiled callable, with distinct fresh owners."""
    template = _bound_original(case, index, key, default_anchor=default_anchor)
    compiled = _jit_original(index, template)
    state_specs = () if prepared is None else prepared.state_specs
    owners = [template]
    references = {}
    for role in _ROLES:
        target = _bound_original(case, index, key, default_anchor=default_anchor)
        owners.append(target)
        assert _snapshot(target) == baseline
        before_fixed = _snapshot(target, nnx.Not(nnx.RngState))
        identities = _identities(target)
        steps = []
        for flag in _flags(index):
            if default_anchor is not None:
                default_anchor.validate(case)
            before = () if prepared is None else _observed_words(target, state_specs)
            result = (
                _eager_original(index, target, x, flag)
                if role == _ROLES[0]
                else compiled(target, x, flag)
            )
            output, tree = _outputs(result)
            after = () if prepared is None else _observed_words(target, state_specs)
            assert _snapshot(target, nnx.Not(nnx.RngState)) == before_fixed
            assert _identities(target) == identities
            if index == 20:
                assert _snapshot(target) == baseline
            if flag is True or index in _GPT or index == 20:
                _same_words(after, before)
            paths = _store_outputs(folder / "native-values", output)
            step = _Step(paths, tree, before, after)
            steps.append(step)
            if role != _ROLES[0]:
                reference = references[_ROLES[0]][len(steps) - 1]
                assert tree == reference.tree
                expected = _load_outputs(reference)
                _compare(output, expected, index)
                _direct_mask(index, flag, x, output, expected)
                _same_words(before, reference.before)
                _same_words(after, reference.after)
        if index in _GPT:
            # Native zero-rate short circuits preserve output and every stream
            # under both actual BOOL values, without inferred flag rewriting.
            for step in steps[1:]:
                for actual, expected in zip(
                    _load_outputs(step), _load_outputs(steps[0]), strict=True
                ):
                    _exact(actual, expected)
        references[role] = tuple(steps)
    if index != 18:
        assert len({id(_module(owner)) for owner in owners}) == len(owners)
    return references, owners


def _prepare(index: int, target: Any) -> PreparedStatefulCall:
    if index in _GPT:
        return prepare_native_control_call(target, control_parameter="deterministic")
    if index in _NNX:
        return prepare_nnx_call(
            target, control_parameter="deterministic", mutable=nnx.RngState
        )
    if index in _LINEN:
        return prepare_linen_call(target, control_parameter="deterministic")
    assert index == 18
    return prepare_pure_control_call(target, control_parameter="inference")


def _prepared_native(
    prepared: PreparedStatefulCall,
    index: int,
    x: Any,
    references: Any,
) -> Any:
    initial = tuple(_host(value) for value in prepared.initial_words)

    def original_contract(values: Any, flag: Any, *words: Any) -> Any:
        return prepared(values, flag, *words)

    compiled = jax.jit(original_contract)
    for role in ("concrete", "eager", "jit"):
        # The second replay resets explicit words and exercises the already
        # compiled callable; the canonical original remains untouched.
        for _ in range(2):
            words = initial
            for position, flag in enumerate(_FLAGS):
                reference = references[_ROLES[0]][position]
                _same_words(words, reference.before)
                if role == "concrete":
                    result = prepared.concrete(flag, x, *words)
                else:
                    runner = prepared if role == "eager" else compiled
                    result = runner(x, jnp.asarray(flag, np.bool_), *words)
                assert isinstance(result, tuple) and len(result) == 1 + len(initial)
                output, tree = _outputs(result[0])
                for native in _ROLES:
                    reference = references[native][position]
                    assert tree == reference.tree
                    expected = _load_outputs(reference)
                    _compare(output, expected, index)
                    _direct_mask(index, flag, x, output, expected)
                    _same_words(result[1:], reference.after)
                words = tuple(_host(value) for value in result[1:])
    return compiled  # Keep this callable alive through target execution.


def _audit_artifact(model: Any, index: int, prepared: Any) -> None:
    report = get_precision_report(model)
    assert report is not None and report.profile == "source-static-v1"
    checks = {check.category: check.status for check in report.checks}
    assert checks["representation"] is checks["lowering"] is CheckStatus.VALIDATED
    assert (
        checks["runtime-execution"]
        is checks["numerical-acceptance"]
        is CheckStatus.UNVERIFIED
    )
    flow = DropoutFlow(model)
    assert not flow.controls, "SOURCE must retain native random computation"
    assert "root/features" in flow.reach(flow.outputs)
    if index == 20:
        assert [value.name for value in model.graph.input] == ["features"]
        return
    names = ["features", "control", *prepared.state_names]
    assert [value.name for value in model.graph.input] == names
    _port(model.graph.input[1], np.bool_, ())
    live = flow.reach([flow.outputs[0]])
    if index in _GPT:
        assert "root/control" not in live
    else:
        assert "root/control" in live
        assert any(
            "root/control" in flow.reach([conditional["predicate"]])
            for conditional in flow.conditionals
        )
    if index in (7, 8, 9):
        assert flow.calls, "Original named Function boundaries must survive"


def _session(blob: bytes, optimize: bool) -> Any:
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(blob, options, providers=["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]
    observed = session.get_session_options()
    assert observed.enable_mem_pattern is False
    assert observed.intra_op_num_threads == observed.inter_op_num_threads == 1
    assert observed.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert observed.graph_optimization_level == options.graph_optimization_level
    return session


def _target_session_options(session: Any, optimize: bool) -> dict[str, Any]:
    observed = session.get_session_options()
    return dict(
        optimize=optimize,
        providers=list(session.get_providers()),
        enable_mem_pattern=observed.enable_mem_pattern,
        enable_cpu_mem_arena=observed.enable_cpu_mem_arena,
        intra_op_num_threads=observed.intra_op_num_threads,
        inter_op_num_threads=observed.inter_op_num_threads,
        execution_mode=str(observed.execution_mode),
        graph_optimization_level=str(observed.graph_optimization_level),
    )


@pytest.mark.parametrize("index", range(len(_CONTRACTS)))
@pytest.mark.parametrize("caller_x64", (False, True), ids=("raw-x64-off", "raw-x64-on"))
def test_original22_source_state_preparation_cpu_feedback_and_reset(
    index: int, caller_x64: bool, tmp_path: Path, record_property: Any
) -> None:
    origin, ordinal, digest = _CONTRACTS[index]
    case = raw_source_case(origin, ordinal)
    default_anchor = (
        Eqx18DefaultAnchor.capture(case, origin, ordinal, digest)
        if index == 18
        else None
    )
    assert _contract(case, default_anchor=default_anchor) == digest
    scenario = raw_source_scenarios(case, SourceProvenance(origin, ordinal))[
        int(caller_x64)
    ]
    assert scenario.criteria is None and scenario.require_reviewed_criteria
    assert scenario.input_specs is not None and len(scenario.input_specs) == 1
    raw_params = dict(case.get("input_params") or {})
    assert set(raw_params) == {"update_stats" if index == 20 else _parameter(index)}
    symbolic = any(isinstance(dim, str) for dim in scenario.input_specs[0].shape)
    batches = (2, 5) if symbolic else (int(scenario.input_specs[0].shape[0]),)
    receipt: dict[str, Any] = dict(
        index=index,
        origin=origin,
        ordinal=ordinal,
        raw_contract_sha256=digest,
        supplemental_native_default_anchor=(
            None if default_anchor is None else default_anchor.evidence()
        ),
        raw_criterion=None,
        caller_x64=caller_x64,
        family=_family(index),
        supplementary_fp32_input_sample_rtol_atol=_FAMILY_CRITERIA[_family(index)],
        previous_mlp_six_bounds="unchanged; different hand-authored inputs",
        prior_raw_first_barrier_ledger="unchanged",
        internal_masks=(
            "exact direct masks/control"
            if index in _DIRECT_DROPOUT
            else "UNVERIFIED internal masks; original return contract has no mask tape"
        ),
        status="UNVERIFIED",
        exports=0,
        sessions=0,
        target_session_options=[],
        target_runs=0,
        batches=batches,
        references={},
    )

    def checkpoint() -> None:
        (tmp_path / "original22-receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n"
        )

    checkpoint()
    with (
        _runtime_configuration(caller_x64),
        source_configuration(scenario),
        native_transformed_execution_context(),
    ):
        # Explicitly bound once by the caller for the raw Eqx lambda. The
        # preparation layer neither creates this key nor advances it.
        key = jax.random.key(20261021, impl="threefry2x32") if index == 18 else None
        with _phase("factory", index=index, caller_x64=caller_x64):
            target = _bound_original(case, index, key, default_anchor=default_anchor)
        with _phase("initial-state-checks", index=index, caller_x64=caller_x64):
            baseline, identities = _snapshot(target), _identities(target)
            bridge_key = getattr(target, "_rngs", None)
            if default_anchor is not None:
                default_anchor.validate(case)
                receipt["supplemental_native_default_anchor"] = (
                    default_anchor.evidence()
                )
        with _phase("preparation", index=index, caller_x64=caller_x64):
            prepared = None if index == 20 else _prepare(index, target)
            initial = (
                ()
                if prepared is None
                else tuple(_host(v) for v in prepared.initial_words)
            )
            if prepared is not None:
                assert isinstance(prepared, PreparedStatefulCall)
                assert (
                    len(initial)
                    == len(prepared.state_names)
                    == len(prepared.state_specs)
                )
                assert len(set(prepared.state_names)) == len(initial)
                _same_words(initial, _observed_words(target, prepared.state_specs))
                _same_words(prepared.words_from_target(target), initial)
                clone = prepared.fresh_original()
                assert _snapshot(clone) == baseline
                if _module(target) is not None:
                    assert _module(clone) is not _module(target)
        export_target = partial(target, update_stats=False) if index == 20 else prepared
        if index == 20:
            assert export_target.func is target and export_target.args == ()
            assert export_target.keywords == raw_params == {"update_stats": False}

        def unchanged() -> None:
            assert _snapshot(target) == baseline
            assert _identities(target) == identities
            assert getattr(target, "_rngs", None) is bridge_key
            if prepared is not None:
                _same_words(_observed_words(target, prepared.state_specs), initial)
                _same_words(prepared.initial_words, initial)
            assert _contract(case, default_anchor=default_anchor) == digest
            assert scenario.criteria is None
            if default_anchor is not None:
                receipt["supplemental_native_default_anchor"] = (
                    default_anchor.evidence()
                )

        feeds: dict[int, Any] = {}
        references: dict[int, Any] = {}
        owners, compiled_owners = [], []
        attempts = []

        def denied(*args: Any, **kwargs: Any) -> Any:
            attempts.append(True)
            pytest.fail(
                "Every independent native reference must precede target sessions"
            )

        try:
            with (
                patch.object(ort, "InferenceSession", denied),
                patch.object(_pybind_state, "InferenceSession", denied),
            ):
                specs = None
                for batch in batches:
                    shapes = tuple(
                        tuple(
                            batch if isinstance(dim, str) else dim for dim in spec.shape
                        )
                        for spec in scenario.input_specs
                    )
                    selected = replace(scenario, concrete_shapes=shapes)
                    declared, arguments = _input_data(
                        case, selected, (origin, str(ordinal), case["testcase"])
                    )
                    assert len(arguments) == 1
                    if specs is None:
                        specs = declared
                    else:
                        assert str(specs) == str(declared)
                    x = arguments[0]
                    feeds[batch] = _host(x)
                    if index == 18:
                        raw = _instantiate(case["callable"])
                        with pytest.raises(RuntimeError, match="key"):
                            raw(x, inference=False)
                    if index == 20:
                        raw = _instantiate(case["callable"])
                        with pytest.raises(ModifyScopeVariableError):
                            raw(x, update_stats=True)
                    with _phase(
                        "native-references",
                        index=index,
                        caller_x64=caller_x64,
                        batch=batch,
                    ):
                        refs, held = _native_references(
                            case,
                            index,
                            key,
                            x,
                            prepared,
                            baseline,
                            tmp_path,
                            default_anchor=default_anchor,
                        )
                    references[batch] = refs
                    owners.extend(held)
                    if prepared is not None:
                        if default_anchor is not None:
                            default_anchor.validate(case)
                        with _phase(
                            "prepared-native",
                            index=index,
                            caller_x64=caller_x64,
                            batch=batch,
                        ):
                            compiled_owners.append(
                                _prepared_native(prepared, index, x, refs)
                            )
                    receipt["references"][str(batch)] = {
                        role: [
                            dict(
                                output_hashes=[path.stem for path in step.output_paths],
                                output_tree=str(step.tree),
                                before=[_hash(v) for v in step.before],
                                after=[_hash(v) for v in step.after],
                            )
                            for step in refs[role]
                        ]
                        for role in _ROLES
                    }
                    unchanged()
                assert specs is not None and not attempts
                receipt["native_references"] = "frozen before export/target sessions"
                checkpoint()
                export_specs = list(specs)
                names = ["features"]
                if prepared is not None:
                    export_specs.extend(
                        [jax.ShapeDtypeStruct((), np.bool_)]
                        + [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in initial]
                    )
                    names.extend(["control", *prepared.state_names])
                # Static SpectralNorm uses exactly the same original receiver and
                # a static False partial; no dynamic-True or mutable-stats claim.
                if default_anchor is not None:
                    default_anchor.validate(case)
                with _phase("source-export", index=index, caller_x64=caller_x64):
                    model = ui._to_onnx(
                        export_target,
                        export_specs,
                        input_params=None,
                        input_names=names,
                        precision_policy=PrecisionPolicy.SOURCE,
                        return_mode="proto",
                        opset=23,
                    )
                receipt["exports"] += 1
                assert isinstance(model, onnx.ModelProto)
                path = tmp_path / "original22.onnx"
                with _phase("artifact-save", index=index, caller_x64=caller_x64):
                    onnx.save_model(model, path)
                with _phase("artifact-reload", index=index, caller_x64=caller_x64):
                    model = onnx.load_model(path)
                with _phase("schema-check", index=index, caller_x64=caller_x64):
                    onnx.checker.check_model(model, full_check=True)
                with _phase("artifact-audit", index=index, caller_x64=caller_x64):
                    _audit_artifact(model, index, prepared)
                with _phase("port-source-checks", index=index, caller_x64=caller_x64):
                    input_shape = tuple(
                        str(dim) if not isinstance(dim, int) else dim
                        for dim in specs[0].shape
                    )
                    _port(model.graph.input[0], specs[0].dtype, input_shape)
                    first = references[batches[0]][_ROLES[0]][0]
                    native_output = _load_outputs(first)
                    count = len(native_output)
                    assert len(model.graph.output) == count + len(initial)
                    for port, value in zip(
                        model.graph.output[:count], native_output, strict=True
                    ):
                        shape = ("B", *value.shape[1:]) if symbolic else value.shape
                        _port(port, value.dtype, shape)
                    for position, value in enumerate(initial):
                        _port(model.graph.input[position + 2], value.dtype, value.shape)
                        _port(
                            model.graph.output[count + position],
                            value.dtype,
                            value.shape,
                        )
                    unchanged()
            assert not attempts
            with _phase("serialization", index=index, caller_x64=caller_x64):
                blob = model.SerializeToString()
            receipt["artifact_sha256"] = hashlib.sha256(blob).hexdigest()
            receipt["same_artifact_all_batches"] = True
            for optimize in (False, True):
                if default_anchor is not None:
                    default_anchor.validate(case)
                with _phase(
                    "cpu-session",
                    index=index,
                    caller_x64=caller_x64,
                    optimize=optimize,
                ):
                    session = _session(blob, optimize)
                receipt["sessions"] += 1
                receipt["target_session_options"].append(
                    _target_session_options(session, optimize)
                )
                assert [value.name for value in session.get_inputs()] == names
                for batch, x in feeds.items():
                    for _ in range(2):
                        words = tuple(value.copy() for value in initial)
                        # State outputs can alias inputs; retain every feed and
                        # returned array throughout the complete feedback replay.
                        replay_owners: list[Any] = [words]
                        for position, flag in enumerate(_flags(index)):
                            values = (
                                (x,)
                                if prepared is None
                                else (x, np.asarray(flag, np.bool_), *words)
                            )
                            feed = dict(zip(names, values, strict=True))
                            replay_owners.append(feed)
                            if default_anchor is not None:
                                default_anchor.validate(case)
                            with _phase(
                                "cpu-run",
                                index=index,
                                caller_x64=caller_x64,
                                batch=batch,
                                optimize=optimize,
                                replay=_,
                                position=position,
                            ):
                                outputs = tuple(session.run(None, feed))
                            replay_owners.append(outputs)
                            receipt["target_runs"] += 1
                            with _phase(
                                "runtime-comparisons",
                                index=index,
                                caller_x64=caller_x64,
                                batch=batch,
                                optimize=optimize,
                                replay=_,
                                position=position,
                            ):
                                actual, after = outputs[:count], outputs[count:]
                                assert len(outputs) == count + len(initial)
                                for role in _ROLES:
                                    reference = references[batch][role][position]
                                    _same_words(words, reference.before)
                                    expected = _load_outputs(reference)
                                    _compare(actual, expected, index)
                                    _direct_mask(index, flag, x, actual, expected)
                                    _same_words(after, reference.after)
                                    # Reconstruct only at the export boundary. The
                                    # native preparation itself retains this tree.
                                    restored = jax.tree.unflatten(
                                        reference.tree, list(actual)
                                    )
                                    assert (
                                        jax.tree.structure(restored) == reference.tree
                                    )
                                if flag is True or index in _GPT:
                                    _same_words(after, words)
                                words = after
                with _phase(
                    "runtime-source-checks",
                    index=index,
                    caller_x64=caller_x64,
                    optimize=optimize,
                ):
                    unchanged()
            assert receipt["exports"] == 1 and receipt["sessions"] == 2
            assert receipt["target_runs"] == 4 * len(batches) * len(_flags(index))
            receipt["status"] = "supplementary finite outputs/state validated"
            receipt["full_B_to_F_acceptance"] = "UNVERIFIED"
            record_property("source-original22", json.dumps(receipt, sort_keys=True))
        except Exception as exc:
            receipt["status"] = "failed; acceptance UNVERIFIED"
            receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            checkpoint()
            unchanged()
