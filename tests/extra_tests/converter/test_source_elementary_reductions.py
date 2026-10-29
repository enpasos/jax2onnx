# tests/extra_tests/converter/test_source_elementary_reductions.py

"""Finite native/consumed SOURCE elementary and reduction witnesses.

Criteria below belong to these declared witnesses. Raw corpus None criteria
remain None. All native eager/fresh-JIT references run outside activation of
SOURCE's private tracing carriers. No operator precision is changed for tests.
"""

from collections.abc import Callable, Iterator
from copy import copy, deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.precision import (
    CheckStatus,
    _array_evidence,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.source_precision_corpus_oracles import (
    _GATHER_GRAPH_TEMPLATE_V2,
    _gather_graph_signature,
    fingerprint,
)
from scripts.source_precision_inventory import canonical_contract
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _body,
    _globals,
    _settings,
)


_FINITE = {
    "float32": SampleCriteria("tolerant", rtol=1e-5, atol=1e-7, equal_nan=True),
    "float64": SampleCriteria("tolerant", rtol=1e-7, atol=1e-12, equal_nan=True),
}
_EXACT = SampleCriteria("exact", equal_nan=True)
_BITS = SampleCriteria("bitwise")
_GATHER = SampleCriteria("tolerant", rtol=1e-7, atol=1e-7, equal_nan=False)
_DIRECT = (
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "log",
    "floor",
    "ceil",
    "abs",
    "sign",
    "is_finite",
    "logistic",
    "log1p",
    "expm1",
)
_F64 = ("sin", "cos", "log", "log1p", "expm1")
_POWERS = (0, 1, 2, 3, 4, -1, -3, 17)
_REDUCTIONS = ("reduce_sum", "reduce_prod", "reduce_min", "reduce_max")
_ORIGINAL_COMPOSITIONS = (
    "jax.nn.relu",
    "jax.nn.relu6",
    "jax.nn.hard_sigmoid",
    "jax.nn.sigmoid",
    "jax.nn.silu",
    "jax.nn.soft_sign",
    "jax.nn.softmax",
    "jax.nn.log_softmax",
    "jax.nn.softplus",
    "jax.nn.elu",
    "jax.nn.selu",
    "jax.nn.leaky_relu",
    "jax.nn.gelu",
    "jax.numpy.mean",
    "jax.numpy.sum",
    "jax.numpy.prod",
    "jax.numpy.all",
    "jax.numpy.any",
)

_ORIGINAL_CASES = tuple((component, 0) for component in _ORIGINAL_COMPOSITIONS) + (
    ("div", 6),
    ("div", 7),
    ("div", 8),
    ("jax.nn.gelu", 6),
)

_GATHER_PRODUCT_GRAPH_SHA256 = (
    "573d12d2c96984c2bcb79af2bfceae7ff7e8b6a59f2b3ace0f037dd646bbc8ba",
    "c5f73dfeb54bcc3fa4567590d9fd0a616ef2a87874b04e37773953a29552b748",
)

# Current typed witnesses are separate from the unchanged historical tuple.
# The old ordinal1 graph is not reconstructed from its historical fingerprint.
_CURRENT_GATHER_PRODUCT_GRAPH_SHA256 = (
    _GATHER_PRODUCT_GRAPH_SHA256[0],
    "c941a4e1e7d1648361023d74713429e3b828b4ce1c7e4023ff438d1b86588468",
)


def _current_gather_graph_witness(ordinal: int) -> dict[str, Any]:
    """Exactly the reviewed V2 graph, with the original ordinal1 I32 indices."""
    assert type(ordinal) is int and ordinal in (0, 1)
    template = _GATHER_GRAPH_TEMPLATE_V2
    assert fingerprint(template) == _GATHER_PRODUCT_GRAPH_SHA256[0]
    assert template["inputs"] == [[11, [4, 3]], [7, [2]]]
    assert template["constants"][:2] == [
        {"dtype": 7, "shape": [], "bytes": "0000000000000000"},
        {"dtype": 7, "shape": [], "bytes": "0400000000000000"},
    ]
    assert template["nodes"][:4] == [
        {
            "op": "Less",
            "inputs": ["input1", "constant0"],
            "outputs": [[9, [2]]],
            "attributes": {},
        },
        {
            "op": "Add",
            "inputs": ["input1", "constant1"],
            "outputs": [[7, [2]]],
            "attributes": {},
        },
        {
            "op": "Where",
            "inputs": ["node0:0", "node1:0", "input1"],
            "outputs": [[7, [2]]],
            "attributes": {},
        },
        {
            "op": "Cast",
            "inputs": ["node2:0"],
            "outputs": [[6, [2]]],
            "attributes": {"to": [2, 6]},
        },
    ]
    expected = deepcopy(template)
    if ordinal == 0:
        return expected
    expected["inputs"][1][0] = 6
    for index, value in enumerate((0, 4)):
        expected["constants"][index] = {
            "dtype": 6,
            "shape": [],
            "bytes": value.to_bytes(4, "little", signed=True).hex(),
        }
    for index in (1, 2):
        expected["nodes"][index]["outputs"] = [[6, [2]]]
    # This exact I64-to-I32 Cast becomes an I32 identity in source ordinal1.
    # All remaining edges and final output roles are remapped together.
    expected["nodes"].pop(3)

    def role(value: str) -> str:
        if not value.startswith("node"):
            return value
        producer, port = value.split(":")
        index = int(producer.removeprefix("node"))
        if index == 3:
            assert port == "0"
            return "node2:0"
        return f"node{index - 1 if index > 3 else index}:{port}"

    for node in expected["nodes"]:
        node["inputs"] = [role(value) for value in node["inputs"]]
    for output in expected["outputs"]:
        output[0] = role(output[0])
    return expected


def _assert_current_gather_graph(signature: dict[str, Any], ordinal: int) -> None:
    assert signature == _current_gather_graph_witness(ordinal)
    assert fingerprint(signature) == _CURRENT_GATHER_PRODUCT_GRAPH_SHA256[ordinal]


def _assert_raw_contract(metadata: Any, origin: str, ordinal: int) -> None:
    inventory = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "internal_docs/precision/source_migration_inventory.json"
        ).read_text()
    )
    registrations = inventory["registrations"]
    matches = [
        row
        for row in inventory["cases"]
        if registrations[row[0]][0] == origin and row[1] == ordinal
    ]
    assert len(matches) == 1
    row = matches[0]
    unresolved: list[str] = []
    actual = canonical_contract(
        {
            key: value
            for key, value in metadata.items()
            if key not in {"testcase", "description", "context", "component"}
        },
        unresolved=unresolved,
    )
    # The persisted contract is JSON. Integer-key callback maps are represented
    # by tuple pairs in memory and array pairs after JSON loading; compare the
    # complete serialized form without discarding any field or payload.
    assert json.dumps(actual, sort_keys=True, separators=(",", ":")) == json.dumps(
        inventory["contracts"][row[5]], sort_keys=True, separators=(",", ":")
    )
    assert (
        metadata["testcase"] == row[2] and metadata.get("opset_version", 23) == row[4]
    )


@pytest.fixture(autouse=True)
def _restore_configuration() -> Iterator[None]:
    hidden, effective = _globals(), _settings()
    try:
        with (
            jax.enable_x64(False),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in hidden.items():
            jax.config.update(name, value)
        assert (_globals(), _settings()) == (hidden, effective)


def _arrays(value: Any) -> list[np.ndarray]:
    return [np.asarray(item) for item in jax.tree.leaves(value)]


def _export(
    fn: Callable[..., Any],
    feeds: tuple[tuple[Any, ...], ...],
    *,
    specs: list[Any] | None = None,
    opset: int = 23,
    carrier: bool = False,
) -> tuple[Any, Any]:
    if specs is None:
        specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in feeds[0]]
    native = jax.make_jaxpr(fn)(*specs)
    # Each feed gets its own fresh JIT wrapper. Native disagreement is retained
    # as two separate roles; target comparisons never demand dual-bit equality.
    references = [
        (_arrays(fn(*values)), _arrays(jax.jit(fn)(*values))) for values in feeds
    ]
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(fn)(*specs)
    if not carrier:
        assert typed_trace_manifest(activated) == typed_trace_manifest(native)
    else:
        assert [
            (str(var.aval.dtype), str(var.aval.shape), var.aval.weak_type)
            for var in activated.outvars
        ] == [
            (str(var.aval.dtype), str(var.aval.shape), var.aval.weak_type)
            for var in native.outvars
        ]
    before = _globals(), _settings()
    proto = ui._to_onnx(
        fn,
        specs,
        opset=opset,
        return_mode="proto",
        precision_policy=PrecisionPolicy.SOURCE,
    )
    assert (_globals(), _settings()) == before
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    for value, var in zip(report.traced_outputs, native.outvars, strict=True):
        assert value.dtype == np.dtype(var.aval.dtype).name
    return proto, references


def _accept(proto: Any, feeds: Any, references: Any, criteria: SampleCriteria) -> None:
    for mode in ("disabled", "all"):
        for role in (0, 1):
            result = validate_source_samples(
                proto,
                samples=[
                    SourceSample(values, refs[role], criteria)
                    for values, refs in zip(feeds, references, strict=True)
                ],
                optimization=mode,
            )
            try:
                result.require_sample_acceptance()
            except SampleAcceptanceError as exc:
                agreement = [
                    [
                        a.dtype == b.dtype
                        and a.shape == b.shape
                        and a.tobytes() == b.tobytes()
                        for a, b in zip(refs[0], refs[1], strict=True)
                    ]
                    for refs in references
                ]
                raise AssertionError(
                    f"target mode={mode}, native role={role}, eager/JIT bit agreement={agreement}: {exc}"
                ) from exc


# Copied as integer storage words from the unchanged original seven-value
# PRIMARY fixtures, saved/1 and saved/2 in the failure-diagnostics-v1 epoch.
# Index 3 is the smallest positive subnormal (word 1), never an ordinary zero.
_SIGN_FIXTURE_WORDS = {
    "float32": {
        "input": (
            0xBF800000,
            0x80000000,
            0,
            1,
            0x00800000,
            0x7F800000,
            0x7FC00000,
        ),
        "sign": (
            0xBF800000,
            0x80000000,
            0,
            0,
            0x3F800000,
            0x3F800000,
            0x7FC00000,
        ),
        "reciprocal": (
            0xBF800000,
            0xFF800000,
            0x7F800000,
            0x7F800000,
            0x3F800000,
            0x3F800000,
            0x7FC00000,
        ),
        "one": 0x3F800000,
    },
    "float64": {
        "input": (
            0xBFF0000000000000,
            0x8000000000000000,
            0,
            1,
            0x0010000000000000,
            0x7FF0000000000000,
            0x7FF8000000000000,
        ),
        "sign": (
            0xBFF0000000000000,
            0x8000000000000000,
            0,
            0,
            0x3FF0000000000000,
            0x3FF0000000000000,
            0x7FF8000000000000,
        ),
        "reciprocal": (
            0xBFF0000000000000,
            0xFFF0000000000000,
            0x7FF0000000000000,
            0x7FF0000000000000,
            0x3FF0000000000000,
            0x3FF0000000000000,
            0x7FF8000000000000,
        ),
        "one": 0x3FF0000000000000,
    },
}
_SIGN_BOOLEAN_WORDS = (1, 1, 1, 1, 1, 1, 0)
_SIGN_EXPECTED_FAILURE = (
    "Output 0: values fail tolerant comparison; "
    "Output 1: values fail tolerant comparison"
)
_SIGN_FIXTURE_PROVENANCE = {
    "epoch": "/tmp/jax2onnx-source-elementary-reductions-failure-diagnostics-v1",
    "float32": {
        "directory": "saved/1",
        "inputs.npz": "554c3d323de11560a7e0eaad7832da432b306aa4bf2ea4b7f4023737987c01ee",
        "native-1.npz": "99c174af2b4dce1255b83e62c93d5a2df4c4dcc41fd2be65faa5a30174fc68ca",
        "native-2.npz": "99c174af2b4dce1255b83e62c93d5a2df4c4dcc41fd2be65faa5a30174fc68ca",
        "target-1.npz": "8957c7a0613d831938d27ffbb7b1c348f0fbda997bfc9991ceecccccbf9d14ba",
    },
    "float64": {
        "directory": "saved/2",
        "inputs.npz": "3c078981ea64360757ac5b37c041353e1144c05b25fb70141af8454d3c6acbc9",
        "native-1.npz": "db63a86311d26409ba8a7d9fbfa427ca75126fa94ac6265d4a99011601d6be23",
        "native-2.npz": "db63a86311d26409ba8a7d9fbfa427ca75126fa94ac6265d4a99011601d6be23",
        "target-1.npz": "1a9d5592e833966730cd9eb285c328f58ac2ed7cadb0e5e678e39ccf317e14d1",
    },
}


def _sign_fixture(dtype_name: str) -> tuple[Any, Any, Any]:
    """Return frozen input/native arrays and the exact failed target candidate."""
    words = _SIGN_FIXTURE_WORDS[dtype_name]
    dtype = np.dtype(dtype_name)
    unsigned = np.dtype(f"uint{dtype.itemsize * 8}")

    def decode(name: str) -> np.ndarray:
        return np.asarray(words[name], dtype=unsigned).view(dtype)

    inputs = (decode("input"),)
    native = (
        decode("sign"),
        decode("reciprocal"),
        np.asarray(_SIGN_BOOLEAN_WORDS, dtype=np.bool_),
    )
    target = tuple(value.copy() for value in native)
    for output in target[:2]:
        output.view(unsigned)[3] = words["one"]
    return inputs, native, target


def _sign_word_record(name: str, value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    raw = array.tobytes(order="C")
    return {
        "array_evidence": asdict(_array_evidence(name, array)),
        "dtype_str": array.dtype.str,
        "shape": list(array.shape),
        "bytes_hex": raw.hex(),
        "words": [
            int.from_bytes(raw[offset : offset + array.itemsize], "little")
            for offset in range(0, len(raw), array.itemsize)
        ],
        "word_byte_order": "little",
    }


def _sign_evidence(outputs: Any) -> tuple[Any, ...]:
    return tuple(
        _array_evidence(str(index), value) for index, value in enumerate(outputs)
    )


def _sign_assert_inputs(inputs: Any) -> str:
    assert len(inputs) == 1, "Sign diagnostic input count differs"
    value = np.asarray(inputs[0])
    assert value.dtype.name in _SIGN_FIXTURE_WORDS, "Sign diagnostic input dtype"
    dtype_name = value.dtype.name
    frozen_inputs, _, _ = _sign_fixture(dtype_name)
    assert _sign_evidence(inputs) == _sign_evidence(
        frozen_inputs
    ), "Sign diagnostic input storage differs, including subnormal word 1"
    unsigned = np.dtype(f"uint{value.dtype.itemsize * 8}")
    assert value.view(unsigned)[3] == 1, "Sign diagnostic input word must be 1"
    return dtype_name


def _sign_assert_native_controls(outputs: Any, dtype_name: str) -> None:
    """Require every unaffected native word, including NaNs and both zeros."""
    _, frozen_native, _ = _sign_fixture(dtype_name)
    assert len(outputs) == 3, "Sign native output count differs"
    for index, (actual, frozen) in enumerate(zip(outputs, frozen_native, strict=True)):
        actual = np.asarray(actual)
        assert actual.dtype == frozen.dtype, "Sign native output dtype differs"
        assert actual.shape == frozen.shape, "Sign native output shape differs"
        positions = range(7) if index == 2 else (0, 1, 2, 4, 5, 6)
        for position in positions:
            assert (
                actual[position : position + 1].tobytes()
                == frozen[position : position + 1].tobytes()
            ), f"Sign native unaffected word differs: output {index}, index {position}"


def _sign_assert_known_failure(
    inputs: Any, native_outputs: Any, target_outputs: Any
) -> None:
    """Match complete run hashes to precisely the saved two-word mismatch."""
    dtype_name = _sign_assert_inputs(inputs)
    _sign_assert_native_controls(native_outputs, dtype_name)
    _, frozen_native, frozen_target = _sign_fixture(dtype_name)
    assert _sign_evidence(native_outputs) == _sign_evidence(
        frozen_native
    ), "Sign native-reference classification differs from the frozen failure"
    assert tuple(target_outputs) == _sign_evidence(
        frozen_target
    ), "Sign target differs from the exact frozen failure fixture"


def _sign_source_pins() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        Path(__file__).resolve(),
        Path(_array_evidence.__code__.co_filename).resolve(),
        root / "jax2onnx/converter/source_profile.py",
        root / "jax2onnx/plugins/jax/lax/sign.py",
        root / "jax2onnx/plugins/jax/lax/_source_elementary.py",
    )
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _accept_sign_diagnostic(
    proto: Any,
    feeds: Any,
    references: Any,
    criteria: SampleCriteria,
    tmp_path: Path,
) -> None:
    """Record four original comparisons, then classify each exact outcome."""
    packet = tmp_path / "original-sign-diagnostic"
    packet.mkdir()
    artifact = proto.SerializeToString(deterministic=True)
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    (packet / "artifact.onnx").write_bytes(artifact)
    source_pins = _sign_source_pins()
    state_before = (_globals(), _settings())
    export_report = get_precision_report(proto)
    context = {
        "purpose": "original Sign numerical comparison and strict failure diagnostic",
        "artifact": {"path": "artifact.onnx", "sha256": artifact_sha256},
        "source_files": source_pins,
        "frozen_fixture_provenance": _SIGN_FIXTURE_PROVENANCE,
        "source_report": None if export_report is None else asdict(export_report),
        "native_state_before": state_before,
        "criteria": asdict(criteria),
        "comparator": "validate_source_samples: actual-first; original frozen criteria",
        "feeds": [
            [_sign_word_record(str(i), value) for i, value in enumerate(values)]
            for values in feeds
        ],
        "native_roles": [
            {
                role_name: [
                    _sign_word_record(str(i), output)
                    for i, output in enumerate(refs[role])
                ]
                for role, role_name in enumerate(("eager", "fresh-jit"))
            }
            for refs in references
        ],
        "known_failure_target_word_candidates": {
            dtype_name: [
                _sign_word_record(str(i), output)
                for i, output in enumerate(_sign_fixture(dtype_name)[2])
            ]
            for dtype_name in _SIGN_FIXTURE_WORDS
        },
        "target_word_binding": (
            "No target is rerun. Full SourceSampleEvidence.outputs hashes must "
            "bind either the known-failure candidate or the current exact native "
            "words for a validated result; candidate words alone are not observations."
        ),
    }
    (packet / "context.json").write_text(json.dumps(context, indent=2, sort_keys=True))
    observed = []
    # Complete all four already-budgeted comparisons and save each unaltered
    # report before any diagnostic assertion can stop this function.
    for mode in ("disabled", "all"):
        for role in (0, 1):
            result = validate_source_samples(
                proto,
                samples=[
                    SourceSample(values, refs[role], criteria)
                    for values, refs in zip(feeds, references, strict=True)
                ],
                optimization=mode,
            )
            report = asdict(result)
            report["_artifact"] = context["artifact"]
            receipt = {
                "mode": mode,
                "native_role": ("eager", "fresh-jit")[role],
                "report": report,
                "report_status": result.status,
                "sample_statuses": [sample.status for sample in result.samples],
                "context": "context.json",
                "source_files": source_pins,
                "artifact": context["artifact"],
            }
            (packet / f"{mode}-{role}.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True)
            )
            observed.append((mode, role, result))
    state_after = (_globals(), _settings())
    (packet / "native-state-after.json").write_text(
        json.dumps(state_after, indent=2, sort_keys=True)
    )

    assert len(feeds) == len(references) == 1
    assert state_before == state_after
    assert source_pins == _sign_source_pins()
    assert hashlib.sha256(proto.SerializeToString(deterministic=True)).hexdigest() == (
        artifact_sha256
    )
    dtype_name = _sign_assert_inputs(feeds[0])
    expected_criteria = {
        "float32": SampleCriteria("tolerant", rtol=1e-5, atol=1e-7, equal_nan=True),
        "float64": SampleCriteria("tolerant", rtol=1e-7, atol=1e-12, equal_nan=True),
    }[dtype_name]
    assert criteria == expected_criteria
    assert len(references[0]) == 2
    for native_outputs in references[0]:
        _sign_assert_native_controls(native_outputs, dtype_name)

    outcomes = []
    for mode, role, result in observed:
        assert result.artifact_sha256 == artifact_sha256
        assert [(check.category, check.status) for check in result.checks] == [
            ("artifact-load", CheckStatus.VALIDATED),
            ("onnx-schema", CheckStatus.VALIDATED),
        ]
        assert result.target is not None
        assert (
            result.target.optimization
            == {
                "disabled": "ORT_DISABLE_ALL",
                "all": "ORT_ENABLE_ALL",
            }[mode]
        )
        assert result.target.providers == ("CPUExecutionProvider",)
        assert result.target.intra_op_threads == result.target.inter_op_threads == 1
        assert result.target.execution_mode == "ORT_SEQUENTIAL"
        assert result.target.session_config == ()
        assert len(result.samples) == 1
        sample = result.samples[0]
        assert sample.index == 0
        assert sample.inputs == _sign_evidence(feeds[0])
        assert sample.references == _sign_evidence(references[0][role])
        assert sample.criteria == criteria
        assert len(sample.outputs) == 3
        assert len(sample.checks) == 2
        runtime, numerical = sample.checks
        assert runtime.category == "runtime-execution"
        assert runtime.status is CheckStatus.VALIDATED
        assert numerical.category == "numerical-acceptance"
        if result.status is CheckStatus.VALIDATED:
            assert sample.status is numerical.status is CheckStatus.VALIDATED
            result.require_sample_acceptance()
            # These Sign/reciprocal fixtures have discrete exact outputs. Full
            # equality also binds every unaffected zero, NaN and Boolean word;
            # index 3 may follow a genuinely changed native implementation.
            assert sample.outputs == _sign_evidence(
                references[0][role]
            ), "Validated Sign target must retain the exact native fixture words"
            bound_target_words = references[0][role]
            outcomes.append((mode, role, "actual-sample-acceptance"))
        else:
            assert result.status is sample.status is CheckStatus.FAILED
            assert numerical.status is CheckStatus.FAILED
            assert numerical.detail == _SIGN_EXPECTED_FAILURE
            # A FAILED result requires both independent native roles to retain
            # the complete saved pattern, including the original reciprocal.
            for native_outputs in references[0]:
                _sign_assert_known_failure(feeds[0], native_outputs, sample.outputs)
            with pytest.raises(SampleAcceptanceError, match="^failed:"):
                result.require_sample_acceptance()
            assert result.status is numerical.status is CheckStatus.FAILED
            bound_target_words = _sign_fixture(dtype_name)[2]
            outcomes.append((mode, role, "known-numerical-failure-exposed"))
        # Complete words are bound to the already-compared run only after its
        # full output evidence matches. The passive epoch observer separately
        # saves actual returned target arrays; there is no second execution.
        (packet / f"{mode}-{role}-bound-words.json").write_text(
            json.dumps(
                {
                    "artifact": context["artifact"],
                    "source_files": source_pins,
                    "sample_sha256": sample.sample_sha256,
                    "mode": mode,
                    "native_role": ("eager", "fresh-jit")[role],
                    "target_settings": asdict(result.target),
                    "original_numerical_status": numerical.status,
                    "criteria": asdict(criteria),
                    "inputs": [
                        _sign_word_record(str(i), value)
                        for i, value in enumerate(feeds[0])
                    ],
                    "references": [
                        _sign_word_record(str(i), value)
                        for i, value in enumerate(references[0][role])
                    ],
                    "target_word_witnesses": [
                        _sign_word_record(str(i), value)
                        for i, value in enumerate(bound_target_words)
                    ],
                    "reported_actual_outputs": [
                        asdict(value) for value in sample.outputs
                    ],
                    "binding": "Full dtype/shape/SHA evidence of this compared run equals the complete target word witnesses above; raw returned arrays are retained by the passive epoch observer.",
                },
                indent=2,
                sort_keys=True,
            )
        )
    assert (
        hashlib.sha256((packet / "artifact.onnx").read_bytes()).hexdigest()
        == artifact_sha256
    )
    (packet / "diagnostic-outcomes.json").write_text(json.dumps(outcomes, indent=2))


@pytest.mark.parametrize(
    "corruption",
    (
        "normal_word",
        "signed_zero_word",
        "nan_payload",
        "boolean_word",
        "dtype",
        "shape",
        "mismatch_position",
    ),
)
def test_sign_failure_diagnostic_rejects_other_corruption(corruption: str) -> None:
    inputs, native, target = _sign_fixture("float32")
    _sign_assert_known_failure(inputs, native, _sign_evidence(target))
    changed = [value.copy() for value in target]
    if corruption == "normal_word":
        changed[0].view(np.uint32)[0] ^= np.uint32(1)
    elif corruption == "signed_zero_word":
        changed[0].view(np.uint32)[1] = 0
    elif corruption == "nan_payload":
        changed[0].view(np.uint32)[6] ^= np.uint32(1)
    elif corruption == "boolean_word":
        changed[2][0] = False
    elif corruption == "dtype":
        changed[0] = changed[0].astype(np.float64)
    elif corruption == "shape":
        changed[0] = changed[0].reshape(1, 7)
    elif corruption == "mismatch_position":
        for output in changed[:2]:
            output[[2, 3]] = output[[3, 2]]
    else:
        raise AssertionError(f"Unfrozen Sign corruption control: {corruption}")
    with pytest.raises(AssertionError, match="Sign target differs"):
        _sign_assert_known_failure(inputs, native, _sign_evidence(changed))


def _direct_values(operation: str, dtype: Any) -> np.ndarray:
    if operation in {"asin", "acos", "atanh"}:
        data = [-0.75, -0.25, -0.0, 0.0, 0.25, 0.75]
    elif operation == "acosh":
        data = [1, 1.25, 2, 3, 4, 8]
    elif operation in {"log", "log1p"}:
        data = [np.finfo(dtype).tiny, 2**-30, 2**-12, 0.25, 1, 4]
    elif operation == "expm1":
        data = [-0.5, -(2**-30), -0.0, 0.0, 2**-30, 0.5]
    else:
        data = [-0.75, -0.25, -0.0, 0.0, 0.25, 0.75]
    return np.asarray(data, dtype=dtype)


@pytest.mark.parametrize("operation", _DIRECT)
def test_direct_elementary_value_and_native_consumers(operation: str) -> None:
    function = getattr(jax.lax, operation)
    values = (_direct_values(operation, np.float32),)

    def model(x: Any) -> Any:
        y = function(x)
        return (
            y,
            y.astype(jnp.float64) if jax.config.x64_enabled else y.astype(jnp.float32),
            jnp.sum(y.astype(jnp.float32)),
            y > np.float32(0.125),
        )

    proto, references = _export(model, (values,))
    _accept(proto, (values,), references, _FINITE["float32"])


@pytest.mark.parametrize("operation", _F64)
def test_f64_retains_values_past_float32_mantissa_and_small_boundaries(
    operation: str,
) -> None:
    with jax.enable_x64(True):
        function = getattr(jax.lax, operation)
        values = (_direct_values(operation, np.float64),)

        def model(x: Any) -> Any:
            y = function(x)
            return y, y.astype(jnp.float32), jnp.sum(y), y > np.float64(2**-31)

        proto, references = _export(model, (values,))
        _accept(proto, (values,), references, _FINITE["float64"])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("operation", ["log1p", "expm1", "sign"])
def test_stable_specials_nan_infinity_tiny_and_consumed_zero_signs(
    dtype: Any, operation: str, tmp_path: Path
) -> None:
    with jax.enable_x64(dtype is np.float64):
        smallest = np.nextafter(np.asarray(0, dtype), np.asarray(1, dtype)).item()
        values = (
            np.asarray(
                [-1, -0.0, 0.0, smallest, np.finfo(dtype).tiny, np.inf, np.nan], dtype
            ),
        )

        def model(x: Any) -> Any:
            y = getattr(jax.lax, operation)(x)
            return y, np.asarray(1, dtype) / y, jnp.isfinite(y)

        proto, references = _export(model, (values,))
        if operation == "sign":
            _accept_sign_diagnostic(
                proto, (values,), references, _FINITE[np.dtype(dtype).name], tmp_path
            )
        else:
            _accept(proto, (values,), references, _FINITE[np.dtype(dtype).name])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", list(jax.lax.RoundingMethod))
def test_round_tie_modes_large_integrals_and_signed_zero(
    dtype: Any, method: Any
) -> None:
    with jax.enable_x64(dtype is np.float64):
        threshold = 2 ** (24 if dtype is np.float32 else 53)
        values = (
            np.asarray(
                [
                    -np.inf,
                    -threshold,
                    -2.5,
                    -1.5,
                    -0.5,
                    -0.0,
                    0.0,
                    0.5,
                    1.5,
                    2.5,
                    threshold,
                    np.inf,
                ],
                dtype,
            ),
        )

        def model(x: Any) -> Any:
            rounded = jax.lax.round(x, method)
            return rounded, np.asarray(1, dtype) / rounded

        proto, references = _export(model, (values,))
        _accept(proto, (values,), references, _BITS)


@pytest.mark.parametrize("exponent", _POWERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("opset", [21, 23])
def test_integer_pow_uses_native_multiplication_association(
    exponent: int, dtype: Any, opset: int
) -> None:
    with jax.enable_x64(dtype is np.float64):
        values = (np.asarray([-2, -1.125, -0.0, 0.0, 0.75, 1.00001, 2], dtype),)

        def model(x: Any) -> Any:
            y = jax.lax.integer_pow(x, exponent)
            return y, y.astype(jnp.float32), np.asarray(1, dtype) / y

        proto, references = _export(model, (values,), opset=opset)
        assert all(
            node.op_type not in {"Pow", "Reciprocal"} for node in proto.graph.node
        )
        _accept(proto, (values,), references, _BITS)


@pytest.mark.parametrize("operation", _REDUCTIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_value_reductions_axes_dtype_accumulator_and_consumers(
    operation: str, dtype: Any
) -> None:
    with jax.enable_x64(dtype is np.float64):
        values = (np.asarray([[[1, 2], [-3, 4]], [[5, -6], [7, 8]]], dtype),)
        function = getattr(jax.lax, operation)

        def model(x: Any) -> Any:
            scalar = function(x, (0, 1, 2))
            multi = function(x, (0, 2))
            identity = function(x, ())
            return (
                scalar,
                multi,
                identity,
                jnp.sum(x, axis=(0, 2), keepdims=True),
                multi.astype(jnp.float32),
                multi > np.asarray(2, dtype),
            )

        proto, references = _export(model, (values,))
        _accept(proto, (values,), references, _EXACT)


@pytest.mark.parametrize("operation", ["reduce_and", "reduce_or", "reduce_xor"])
def test_boolean_reduction_is_separate_from_integer_bitwise(operation: str) -> None:
    values = (np.asarray([[True, False, True], [False, False, True]], np.bool_),)
    function = getattr(jax.lax, operation)

    def model(x: Any) -> Any:
        y = function(x, (1,))
        return y, function(x, (0, 1)), function(x, ()), y.astype(jnp.int32)

    proto, references = _export(model, (values,))
    _accept(proto, (values,), references, _BITS)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32])
def test_integer_arithmetic_modulo_truncation_exceptions_and_consumed_values(
    dtype: Any,
) -> None:
    with jax.enable_x64(dtype is np.int64):
        info = np.iinfo(dtype)
        unsigned = np.issubdtype(dtype, np.unsignedinteger)
        x = np.asarray([info.min, info.max, 7, 0, 1], dtype)
        y = np.asarray([info.max if unsigned else -1, 2, 3, 0, 0], dtype)

        def model(a: Any, b: Any) -> Any:
            quotient, remainder = jax.lax.div(a, b), jax.lax.rem(a, b)
            product = jax.lax.mul(a, b)
            return (
                quotient,
                remainder,
                product,
                jax.lax.neg(a),
                quotient == np.asarray(info.max if unsigned else -1, dtype),
                product.astype(jnp.float32),
            )

        proto, references = _export(model, ((x, y),))
        _accept(proto, ((x, y),), references, _BITS)


def _elementary_body(x: Any) -> Any:
    y = jnp.sin(x) + jnp.log1p(jnp.abs(x))
    return jnp.sum(y, axis=1), jnp.max(y, axis=0)


@onnx_function(name="SourceElementaryReductions")
def _named(x: Any) -> Any:
    return _elementary_body(x)


@pytest.mark.parametrize("scope", ["named", "jit", "cond"])
def test_actual_symbolic_B2_B5_one_artifact_function_jit_cond_and_removed_batch(
    scope: str,
) -> None:
    (batch,) = export.symbolic_shape("B")
    specs = [jax.ShapeDtypeStruct((batch, 3), np.float32)]
    fn = (
        (lambda x: _named(x))
        if scope == "named"
        else (
            jax.jit(_elementary_body)
            if scope == "jit"
            else lambda x: jax.lax.cond(
                jnp.sum(x) > np.float32(0),
                _elementary_body,
                lambda y: tuple(-value for value in _elementary_body(y)),
                x,
            )
        )
    )
    feeds = tuple(
        (np.linspace(-0.5, 0.75, b * 3, dtype=np.float32).reshape(b, 3),)
        for b in (2, 5)
    )
    proto, references = _export(fn, feeds, specs=specs, carrier=True)
    if scope == "named":
        assert any(
            function.name == "SourceElementaryReductions"
            for function in proto.functions
        )
        function = next(
            function
            for function in proto.functions
            if function.name == "SourceElementaryReductions"
        )
        assert len(function.input) == 1 and len(function.output) == 2
        assert any(node.op_type == "Sin" for node in function.node)
    _accept(proto, feeds, references, _FINITE["float32"])


def _empty_reduction_model(x: Any) -> Any:
    truth = x != np.float32(0)
    return tuple(getattr(jax.lax, op)(x, (1,)) for op in _REDUCTIONS) + tuple(
        getattr(jax.lax, op)(truth, (1,))
        for op in ("reduce_and", "reduce_or", "reduce_xor")
    )


def test_empty_static_reduction_identities_and_consumers() -> None:
    feeds = ((np.empty((2, 0, 3), np.float32),),)
    proto, references = _export(_empty_reduction_model, feeds)
    _accept(proto, feeds, references, _BITS)


def test_original_symbolic_zero_tail_witness_remains_an_open_support_obligation() -> (
    None
):
    # Keep the actual v1 failing witness. This is a temporary audited-profile
    # restriction, not a proof of ONNX/native impossibility or a migrated case.
    (batch,) = export.symbolic_shape("B")
    with pytest.raises(SourcePrecisionError, match="positive static tails"):
        ui._to_onnx(
            _empty_reduction_model,
            [jax.ShapeDtypeStruct((batch, 0, 3), np.float32)],
            opset=23,
            return_mode="proto",
            precision_policy=PrecisionPolicy.SOURCE,
        )


@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16])
def test_low_precision_native_values_consumers_and_schema_without_cpu_claim(
    dtype: Any,
) -> None:
    values = (np.asarray([-0.5, -0.0, 0.0, 0.5], dtype),)

    def model(x: Any) -> Any:
        y = jax.lax.sin(x)
        return y, y.astype(jnp.float32), jnp.sum(y), jax.lax.sign(x)

    proto, references = _export(model, (values,))
    assert all(ref[0].dtype == np.dtype(dtype) for ref in references[0])
    assert all(
        out.dtype == np.dtype(dtype).name
        for out in (
            get_precision_report(proto).traced_outputs[0],
            get_precision_report(proto).traced_outputs[2],
        )
    )


def test_uint64_high_values_are_typed_schema_and_native_accumulation_controls() -> None:
    with jax.enable_x64(True):
        values = (np.asarray([2**63, 2**63 + 1, 2**64 - 1], np.uint64),)

        def model(x: Any) -> Any:
            return (
                jax.lax.reduce_sum(x, (0,)),
                jax.lax.reduce_prod(x, (0,)),
                jax.lax.min(x, np.uint64(2**63 + 1)),
                x.astype(jnp.float64),
            )

        proto, references = _export(model, (values,))
        assert references[0][0][0].dtype == np.dtype(np.uint64)
        assert references[0][0][0].item() == 0
        assert get_precision_report(proto).traced_outputs[0].dtype == "uint64"


def test_uint32_sum_preserves_modular_result_after_bounded_int64_work() -> None:
    values = (np.asarray([2**32 - 1, 2**32 - 1, 5], np.uint32),)

    def model(x: Any) -> Any:
        result = jax.lax.reduce_sum(x, (0,))
        return result, result == np.uint32(3), result.astype(jnp.float32)

    proto, references = _export(model, (values,))
    _accept(proto, (values,), references, _BITS)


@pytest.mark.parametrize("kind", ["integer", "boolean"])
def test_native_discrete_extrema_abs_sign_and_broadcast_consumers(kind: str) -> None:
    dtype = np.int32 if kind == "integer" else np.bool_
    values = (
        np.asarray(
            (
                [[-(2**31), -1, 0], [2**31 - 1, 1, 2]]
                if kind == "integer"
                else [[True, False, True], [False, False, True]]
            ),
            dtype,
        ),
    )

    def model(x: Any) -> Any:
        minimum, maximum = jax.lax.min(x, np.asarray(0, dtype)), jax.lax.max(
            x, np.asarray(0, dtype)
        )
        return (
            (
                minimum,
                maximum,
                jax.lax.abs(x),
                jax.lax.sign(x),
                jax.lax.clamp(np.int32(-1), x, np.int32(1)),
            )
            if kind == "integer"
            else (minimum, maximum, jnp.all(x, axis=1), jnp.any(x, axis=1))
        )

    proto, references = _export(model, (values,))
    _accept(proto, (values,), references, _BITS)


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64],
    ids=lambda dtype: np.dtype(dtype).name,
)
def test_integral_extrema_all_width_schemas_exact_16bit_bridge_and_empty_noop(
    dtype: Any,
) -> None:
    with jax.enable_x64(np.dtype(dtype).itemsize == 8):
        info = np.iinfo(dtype)
        values = (
            np.asarray([[info.min, info.max, 0], [info.max, 1, info.min]], dtype),
            np.empty((2, 0), dtype),
        )

        def model(x: Any, empty: Any) -> Any:
            minimum = jax.lax.reduce_min(x, (1,))
            maximum = jax.lax.reduce_max(x, (1,))
            return (
                minimum,
                maximum,
                jax.lax.reduce_min(x, (0, 1)),
                jax.lax.reduce_max(x, (0, 1)),
                jax.lax.reduce_min(x, ()),
                jax.lax.reduce_max(x, ()),
                jax.lax.reduce_min(empty, (1,)),
                jax.lax.reduce_max(empty, (1,)),
                minimum.astype(jnp.float32),
                maximum.astype(jnp.float32),
                minimum == np.asarray(info.min, dtype),
                maximum == np.asarray(info.max, dtype),
            )

        proto, references = _export(model, (values,))
        assert references[0][0][2].item() == info.min
        assert references[0][0][3].item() == info.max
        assert np.all(references[0][0][6] == info.max)
        assert np.all(references[0][0][7] == info.min)
        typed = onnx.shape_inference.infer_shapes(proto, strict_mode=True)
        types = {
            value.name: value.type.tensor_type.elem_type
            for value in (
                *typed.graph.input,
                *typed.graph.value_info,
                *typed.graph.output,
            )
        }
        reductions = [
            node
            for node in typed.graph.node
            if node.op_type in {"ReduceMin", "ReduceMax"}
        ]
        assert len(reductions) == 4
        for node in reductions:
            emitted = np.dtype(
                onnx.helper.tensor_dtype_to_np_dtype(types[node.input[0]])
            ).name
            schema_type = "tensor(" + emitted + ")"
            allowed = onnx.defs.get_schema(node.op_type, 23).type_constraints[0]
            assert schema_type in allowed.allowed_type_strs
            if np.dtype(dtype).itemsize == 2:
                assert types[node.input[0]] == onnx.TensorProto.INT32
                assert types[node.output[0]] == onnx.TensorProto.INT32
            else:
                assert types[node.input[0]] == onnx.helper.np_dtype_to_tensor_dtype(
                    np.dtype(dtype)
                )
        for index in range(8):
            assert typed.graph.output[index].type.tensor_type.elem_type == (
                onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
            )
        # Other widths are schema/native retention controls; their target kernel
        # availability remains separate. Execute the two corrected work paths.
        if np.dtype(dtype).itemsize == 2:
            _accept(proto, (values,), references, _BITS)


@pytest.mark.parametrize("opset", [16, 23])
def test_fp16_is_finite_keeps_data_and_uses_only_required_predicate_bridge(
    opset: int,
) -> None:
    tiny = np.nextafter(np.float16(0), np.float16(1))
    values = (
        np.asarray(
            [-np.inf, -65504, -tiny, -0.0, 0.0, tiny, 65504, np.inf, np.nan],
            np.float16,
        ),
    )

    def model(x: Any) -> Any:
        predicate = jax.lax.is_finite(x)
        return x, predicate, predicate.astype(jnp.int32)

    proto, references = _export(model, (values,), opset=opset)
    assert (
        next(item.version for item in proto.opset_import if item.domain == "") == opset
    )
    assert proto.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    bridges = [
        node
        for node in proto.graph.node
        if node.op_type == "Cast"
        and any(
            attr.name == "to" and attr.i == onnx.TensorProto.FLOAT
            for attr in node.attribute
        )
    ]
    assert len(bridges) == (1 if opset == 16 else 0)
    if bridges:
        assert bridges[0].input[0] == proto.graph.input[0].name
        assert all(
            node.op_type in {"IsInf", "IsNaN"}
            for node in proto.graph.node
            if bridges[0].output[0] in node.input
        )
    _accept(proto, (values,), references, _BITS)


@pytest.mark.parametrize("component,ordinal", _ORIGINAL_CASES)
@pytest.mark.parametrize("x64", [False, True])
def test_original_composition_artifacts_retain_unmodified_contract_and_raw_callback_first(
    component: str, ordinal: int, x64: bool, tmp_path: Path
) -> None:
    metadata = source.raw_source_case(f"plugin::{component}", ordinal)
    _assert_raw_contract(metadata, f"plugin::{component}", ordinal)
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance(f"plugin::{component}", ordinal)
    )[int(x64)]
    if any(
        any(isinstance(dim, str) for dim in spec.shape) for spec in scenario.input_specs
    ):
        scenario = replace(
            scenario,
            concrete_shapes=tuple(
                tuple(2 if dim == "B" else dim for dim in spec.shape)
                for spec in scenario.input_specs
            ),
        )
    prepared = source.prepare_source_case(metadata, scenario)
    assert prepared.fn is metadata["callable"] and prepared.criteria is None
    with source.source_configuration(scenario):
        compiled = _arrays(jax.jit(prepared.fn)(*prepared.arguments))
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(prepared.fn)(*prepared.input_specs)
        assert typed_trace_manifest(activated) == typed_trace_manifest(
            prepared.original_jaxpr
        )
        proto = source._export_prepared(prepared)
    callback = metadata.get("post_check_onnx_graph")
    try:
        raw_status = (
            "absent" if callback is None else "passed" if callback(proto) else "failed"
        )
        detail = ""
    except Exception as exc:
        raw_status, detail = "error", f"{type(exc).__name__}: {exc}"
    # A raw fused-op callback may fail on the preserved native expression.
    # Keep that failure and the whole serialized artifact for ROOT's actual
    # product graph review. This diagnostic cohort claims no numerical or
    # replacement-structural acceptance and never changes the raw criterion.
    onnx.save(proto, tmp_path / "original-product.onnx")
    (tmp_path / "original-evidence.json").write_text(
        json.dumps(
            {
                "origin": f"plugin::{component}",
                "ordinal": ordinal,
                "caller_x64": x64,
                "raw_rtol": metadata.get("rtol"),
                "raw_atol": metadata.get("atol"),
                "raw_numeric_skip": metadata.get("skip_numeric_validation"),
                "source_criteria": None,
                "raw_callback": raw_status,
                "raw_callback_detail": detail,
                "numerical_acceptance": "unverified",
                "native_eager_jit_bit_equal": [
                    a.dtype == b.dtype
                    and a.shape == b.shape
                    and a.tobytes() == b.tobytes()
                    for a, b in zip(prepared.reference_outputs, compiled, strict=True)
                ],
            }
        )
    )
    assert len(compiled) == len(prepared.reference_outputs)
    assert (
        next(
            check.status
            for check in get_precision_report(proto).checks
            if check.category == "numerical-acceptance"
        )
        is CheckStatus.UNVERIFIED
    )


@pytest.mark.parametrize("operation", ["reduce_sum", "integer_pow", "round", "sin"])
@pytest.mark.parametrize(
    "mutation", ["parameter", "dtype", "shape", "weakness", "placement", "effects"]
)
def test_malformed_typed_contracts_reach_the_validator(
    operation: str, mutation: str
) -> None:
    fn = (
        (lambda x: jax.lax.reduce_sum(x, (0,)))
        if operation == "reduce_sum"
        else (
            (lambda x: jax.lax.integer_pow(x, 2))
            if operation == "integer_pow"
            else (lambda x: jax.lax.round(x)) if operation == "round" else jax.lax.sin
        )
    )
    native = _body(jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((3,), np.float32))).eqns[0]
    inputs = [SimpleNamespace(aval=copy(var.aval)) for var in native.invars]
    outputs = [SimpleNamespace(aval=copy(native.outvars[0].aval))]
    params, effects = dict(native.params), ()
    if mutation == "parameter":
        params.update(
            {"axes": (True,)}
            if operation == "reduce_sum"
            else (
                {"y": True}
                if operation == "integer_pow"
                else (
                    {"rounding_method": 1}
                    if operation == "round"
                    else {"accuracy": "unknown"}
                )
            )
        )
    elif mutation == "dtype":
        outputs[0].aval = jax.core.ShapedArray(outputs[0].aval.shape, np.float64)
    elif mutation == "shape":
        outputs[0].aval = jax.core.ShapedArray((7,), np.float32)
    elif mutation == "weakness":
        outputs[0].aval = jax.core.ShapedArray(
            outputs[0].aval.shape, np.float32, weak_type=True
        )
    elif mutation == "placement":
        params["out_sharding"] = object()
    else:
        effects = ("effect",)
    eqn = SimpleNamespace(
        primitive=native.primitive,
        invars=inputs,
        outvars=outputs,
        params=params,
        effects=effects,
    )
    graph = SimpleNamespace(
        invars=inputs, constvars=(), outvars=outputs, eqns=[eqn], effects=effects
    )
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(graph, 23)


@pytest.mark.parametrize("axes", [(0.0,), (np.bool_(True),), (0, 0), (-1,), (2,)])
def test_reduction_rejects_float_boolean_duplicate_and_noncanonical_axes(
    axes: Any,
) -> None:
    x = SimpleNamespace(aval=jax.core.ShapedArray((2, 3), np.float32))
    y = SimpleNamespace(aval=jax.core.ShapedArray((3,), np.float32))
    eqn = SimpleNamespace(
        primitive=SimpleNamespace(name="reduce_sum"),
        invars=[x],
        outvars=[y],
        params={"axes": axes},
        effects=(),
    )
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(
            SimpleNamespace(
                invars=[x], constvars=[], outvars=[y], eqns=[eqn], effects=()
            ),
            23,
        )


def test_physical_drift_reaches_the_owning_lowerer() -> None:
    from jax2onnx.plugins.jax.lax._source_elementary import lower_source_elementary

    x = SimpleNamespace(aval=jax.core.ShapedArray((3,), np.float32))
    y = SimpleNamespace(aval=jax.core.ShapedArray((3,), np.float32))
    bad = ir.Value(
        name="drift", type=ir.TensorType(ir.DataType.DOUBLE), shape=ir.Shape((3,))
    )
    target = ir.Value(
        name="out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,))
    )
    ctx = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        fresh_name=lambda name: name,
        require_value_for_var=lambda var: bad,
        allocate_value_for_var=lambda var, **kwargs: target,
    )
    with pytest.raises(ValueError, match="canonical representation"):
        lower_source_elementary(
            ctx, SimpleNamespace(invars=[x], outvars=[y], params={}), operation="log1p"
        )


@pytest.mark.parametrize("ordinal", [0, 1])
def test_original_gather_trigonometric_anchor_preserves_raw_criterion_and_both_native_roles(
    ordinal: int, tmp_path: Path
) -> None:
    metadata = source.raw_source_case("plugin::gather", ordinal)
    _assert_raw_contract(metadata, "plugin::gather", ordinal)
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::gather", ordinal)
    )[1]
    prepared = source.prepare_source_case(metadata, scenario)
    assert scenario.criteria is prepared.criteria is None
    assert metadata["callable"] is prepared.fn
    with source.source_configuration(scenario):
        proto = source._export_prepared(prepared)
        eager = prepared.reference_outputs
        compiled = _arrays(jax.jit(prepared.fn)(*prepared.arguments))
    onnx.save(proto, tmp_path / "original-gather-product.onnx")
    np.savez(
        tmp_path / "original-native-values.npz",
        input0=np.asarray(prepared.arguments[0]),
        input1=np.asarray(prepared.arguments[1]),
        eager=np.asarray(eager[0]),
        jit=np.asarray(compiled[0]),
    )
    callback = metadata["post_check_onnx_graph"]
    try:
        raw_status = "passed" if callback(proto) else "failed"
        raw_detail = ""
    except Exception as exc:
        raw_status, raw_detail = "error", f"{type(exc).__name__}: {exc}"
    (tmp_path / "raw-callback.json").write_text(
        json.dumps(
            {
                "origin": "plugin::gather",
                "ordinal": ordinal,
                "raw_criteria": None,
                "raw_callback": raw_status,
                "detail": raw_detail,
            }
        )
    )
    assert raw_status == "failed"
    # Freeze the complete type/shape/def-use/constant signatures only after
    # inspecting both actual product graphs. The old diagnostic mapping and
    # its historical raw callback failure remain independent evidence.
    current_signature = _gather_graph_signature(proto)
    _assert_current_gather_graph(current_signature, ordinal)
    # This small typed product witness does not replace the historical frozen
    # diagnostic counterpart or claim that a failing raw callback passed.
    assert proto.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
    assert proto.graph.input[1].type.tensor_type.elem_type == (
        onnx.TensorProto.INT64 if ordinal == 0 else onnx.TensorProto.INT32
    )
    assert any(node.op_type == "Sin" for node in proto.graph.node)
    assert any(node.op_type == "Gather" for node in proto.graph.node)
    assert not any(
        node.op_type == "Cast"
        and any(
            attr.name == "to" and attr.i == onnx.TensorProto.FLOAT
            for attr in node.attribute
        )
        for node in proto.graph.node
    )
    # Compare both predicates on each identical target result. Historical
    # allclose is reference-first/equal_nan=True; SOURCE is actual-first with
    # the separately frozen equal_nan=False. Neither mutates raw None.
    targets = {}
    for mode, optimization in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = optimization
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            proto.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        actual = session.run(
            None,
            {
                value.name: np.asarray(argument)
                for value, argument in zip(
                    proto.graph.input, prepared.all_arguments, strict=True
                )
            },
        )
        assert len(actual) == 1
        targets[mode] = actual[0]
        for role, reference in enumerate((eager, compiled)):
            assert len(actual) == len(reference)
            for target, expected in zip(actual, reference, strict=True):
                assert _gather_sample_accepts(
                    np.asarray(target), np.asarray(expected)
                ), (
                    mode,
                    role,
                )
    np.savez(tmp_path / "original-gather-target-values.npz", **targets)


def _gather_sample_accepts(actual: np.ndarray, reference: np.ndarray) -> bool:
    from jax2onnx.precision import _compare

    return _compare(actual, reference, _GATHER) is None and bool(
        np.allclose(reference, actual, rtol=1e-7, atol=1e-7, equal_nan=True)
    )


def test_gather_conjunction_rejects_asymmetric_and_nan_predicate_counterexamples() -> (
    None
):
    from jax2onnx.precision import _compare

    reference = np.asarray([1.0], np.float64)
    actual = np.asarray([0.99999980000001], np.float64)
    assert _compare(actual, reference, _GATHER) is None
    assert not np.allclose(reference, actual, rtol=1e-7, atol=1e-7, equal_nan=True)
    assert not _gather_sample_accepts(actual, reference)
    nan = np.asarray([np.nan], np.float64)
    assert np.allclose(nan, nan, rtol=1e-7, atol=1e-7, equal_nan=True)
    assert not _gather_sample_accepts(nan, nan)
    assert not _gather_sample_accepts(reference.astype(np.float32), reference)
    assert not _gather_sample_accepts(reference.reshape(1, 1), reference)


@pytest.mark.parametrize("ordinal", [0, 1])
def test_original_gather_keeps_genuine_x64_off_configuration_conflict(
    ordinal: int,
) -> None:
    metadata = source.raw_source_case("plugin::gather", ordinal)
    _assert_raw_contract(metadata, "plugin::gather", ordinal)
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::gather", ordinal)
    )[0]
    with pytest.raises((SourcePrecisionError, source.SourcePreparationError)):
        source.prepare_source_case(metadata, scenario)


def test_exact_original_gelu18_keeps_erfc_owning_opset_gap_visible() -> None:
    metadata = source.raw_source_case("plugin::jax.nn.gelu", 5)
    _assert_raw_contract(metadata, "plugin::jax.nn.gelu", 5)
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::jax.nn.gelu", 5)
    )[0]
    prepared = source.prepare_source_case(metadata, scenario)
    assert metadata["opset_version"] == 18
    with (
        source.source_configuration(scenario),
        pytest.raises(SourcePrecisionError) as captured,
    ):
        source._export_prepared(prepared)
    assert captured.value.opset == 18 and captured.value.primitive == "erfc"


def test_frozen_complete_original_selector_keeps_every_contract_and_remaining_obligation() -> (
    None
):
    components = set(
        "abs acos acosh add asin asinh atan atan2 atanh cbrt ceil clamp cos cosh div exp exp2 expm1 floor integer_pow is_finite log log1p logistic max min mul neg pow reduce_and reduce_max reduce_min reduce_or reduce_prod reduce_sum reduce_xor rem round rsqrt sign sin sinh sqrt sub tan tanh all amax amin any clip isfinite maximum mean minimum prod sum elu gelu hard_sigmoid leaky_relu log_softmax relu relu6 selu sigmoid silu soft_sign softmax softplus".split()
    )
    inventory = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "internal_docs/precision/source_migration_inventory.json"
        ).read_text()
    )
    selected_registrations = {
        index
        for index, (_, _, context, component) in enumerate(inventory["registrations"])
        if context in {"primitives.lax", "primitives.jnp", "primitives.nn"}
        and component in components
    }
    rows = []
    for row in inventory["cases"]:
        origin = inventory["registrations"][row[0]][0]
        if (
            row[0] in selected_registrations
            or origin == "plugin::gather"
            and row[1] in (0, 1)
        ):
            rows.append(
                [origin, row[1], row[2], row[4], inventory["contracts"][row[5]]]
            )
    assert (
        len(selected_registrations) == 97 and len(rows) == 318 and 2 * len(rows) == 636
    )
    assert (
        hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == "a49953f0d8b9dc472948c749af8abca31165f4df14b7cfe2391a9adf9f769d89"
    )


def test_consumed_scalar_and_saved_target_corruption_are_detected(
    tmp_path: Path,
) -> None:
    def model(x: Any) -> Any:
        value = x * np.float32(2)
        return value, value > np.float32(0.75), jnp.sum(value)

    values = (np.asarray([0.25, 0.5], np.float32),)
    proto, references = _export(model, (values,))
    _accept(proto, (values,), references, _BITS)
    corrupted = onnx.load_model_from_string(proto.SerializeToString())
    coefficient = next(
        tensor
        for tensor in corrupted.graph.initializer
        if onnx.numpy_helper.to_array(tensor).shape == ()
        and onnx.numpy_helper.to_array(tensor).dtype == np.dtype(np.float32)
        and onnx.numpy_helper.to_array(tensor).item() == 2
    )
    coefficient.CopyFrom(
        onnx.numpy_helper.from_array(np.asarray(3, np.float32), name=coefficient.name)
    )
    rejected = validate_source_samples(
        corrupted,
        samples=[SourceSample(values, references[0][0], _BITS)],
        optimization="disabled",
    )
    with pytest.raises(SampleAcceptanceError):
        rejected.require_sample_acceptance()
    from jax2onnx.precision import _compare

    saved = tmp_path / "saved-target.npy"
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        proto.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {proto.graph.input[0].name: values[0]})
    np.save(saved, actual[0])
    original_digest = hashlib.sha256(saved.read_bytes()).hexdigest()
    payload = np.load(saved)
    payload[0] += np.float32(1)
    np.save(saved, payload)
    assert hashlib.sha256(saved.read_bytes()).hexdigest() != original_digest
    assert _compare(np.load(saved), references[0][0][0], _BITS) is not None


def test_current_gather_typed_witness_preserves_templates_and_historical_pins() -> None:
    assert _GATHER_PRODUCT_GRAPH_SHA256 == (
        "573d12d2c96984c2bcb79af2bfceae7ff7e8b6a59f2b3ace0f037dd646bbc8ba",
        "c5f73dfeb54bcc3fa4567590d9fd0a616ef2a87874b04e37773953a29552b748",
    )
    original = deepcopy(_GATHER_GRAPH_TEMPLATE_V2)
    ordinal0 = _current_gather_graph_witness(0)
    ordinal1 = _current_gather_graph_witness(1)
    assert ordinal0 == original and ordinal0 is not _GATHER_GRAPH_TEMPLATE_V2
    _assert_current_gather_graph(ordinal0, 0)
    _assert_current_gather_graph(ordinal1, 1)
    assert fingerprint(ordinal0) == _CURRENT_GATHER_PRODUCT_GRAPH_SHA256[0]
    assert fingerprint(ordinal1) == _CURRENT_GATHER_PRODUCT_GRAPH_SHA256[1]
    assert _CURRENT_GATHER_PRODUCT_GRAPH_SHA256[1] == (
        "c941a4e1e7d1648361023d74713429e3b828b4ce1c7e4023ff438d1b86588468"
    )
    assert ordinal1["inputs"] == [[11, [4, 3]], [6, [2]]]
    assert len(ordinal1["nodes"]) == 25 and len(ordinal1["constants"]) == 14
    assert ordinal1["constants"][2:] == original["constants"][2:]
    assert ordinal1["nodes"][5]["inputs"] == ["node2:0", "node4:0"]
    assert ordinal1["outputs"] == [["node24:0", [11, [2, 3]]]]
    assert _GATHER_GRAPH_TEMPLATE_V2 == original
    for invalid_ordinal in (-1, 2, True):
        with pytest.raises(AssertionError):
            _current_gather_graph_witness(invalid_ordinal)


@pytest.mark.parametrize(
    "corruption",
    [
        "input-type",
        "integer-literal",
        "normalization-type",
        "where-arms",
        "layout-edge",
        "gather-axis",
        "floating-operand",
        "trigonometric-edge",
        "copy-arms",
        "selector-type",
        "copy-axis",
        "copy-shape",
        "zero-fill-word",
        "extra-node",
    ],
)
def test_current_gather_typed_witness_rejects_whole_signature_corruptions(
    corruption: str,
) -> None:
    original = deepcopy(_GATHER_GRAPH_TEMPLATE_V2)
    damaged = _current_gather_graph_witness(1)
    if corruption == "input-type":
        damaged["inputs"][1][0] = 7
    elif corruption == "integer-literal":
        damaged["constants"][1]["bytes"] = "02000000"
    elif corruption == "normalization-type":
        damaged["nodes"][2]["outputs"][0][0] = 7
    elif corruption == "where-arms":
        damaged["nodes"][2]["inputs"][1:] = list(
            reversed(damaged["nodes"][2]["inputs"][1:])
        )
    elif corruption == "layout-edge":
        damaged["nodes"][5]["inputs"][0] = "input1"
    elif corruption == "gather-axis":
        damaged["nodes"][8]["attributes"]["axis"][1] = 1
    elif corruption == "floating-operand":
        damaged["nodes"][11]["inputs"][0] = "node8:0"
    elif corruption == "trigonometric-edge":
        damaged["nodes"][12]["inputs"][0] = "node9:0"
    elif corruption == "copy-arms":
        damaged["nodes"][21]["inputs"].reverse()
    elif corruption == "selector-type":
        damaged["nodes"][18]["attributes"]["to"][1] = 7
    elif corruption == "copy-axis":
        damaged["nodes"][23]["attributes"]["axis"][1] = 1
    elif corruption == "copy-shape":
        damaged["nodes"][24]["outputs"][0][1] = [3, 2]
    elif corruption == "zero-fill-word":
        damaged["constants"][12]["bytes"] = "000000000000f03f" + "0000000000000000" * 5
    else:
        damaged["nodes"].append(deepcopy(damaged["nodes"][-1]))
    with pytest.raises(AssertionError):
        _assert_current_gather_graph(damaged, 1)
    assert fingerprint(damaged) != _CURRENT_GATHER_PRODUCT_GRAPH_SHA256[1]
    assert _GATHER_GRAPH_TEMPLATE_V2 == original
