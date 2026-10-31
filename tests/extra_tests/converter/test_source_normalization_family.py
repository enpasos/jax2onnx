# tests/extra_tests/converter/test_source_normalization_family.py

"""Original LayerNorm rows, retained native instances and shared-B consumers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
from jax import export
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.capi import _pybind_state as ort_state
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter import conversion_api
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import validate_source_jaxpr
from jax2onnx.converter.source_representation import normalized_shape_dimensions
from jax2onnx.plugins.plugin_system import FunctionPlugin
from jax2onnx.precision import (
    CheckStatus,
    PRECISION_METADATA_KEY,
    SampleCriteria,
    get_precision_report,
)
from scripts import source_precision_probe as probe
from scripts.source_precision_inventory import canonical_contract
from tests.extra_tests.converter import source_precision_support as source_support
from tests.extra_tests.converter.source_precision_corpus_oracles import (
    CorpusCase,
    NORMALIZATION_CASES,
    NORMALIZATION_COUNTERPARTS,
    _NormalizationExpressions,
    counterpart_for,
    fingerprint,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.source_precision_support import (
    PRESERVE_FACTORY_DTYPE,
    SourcePreparationError,
    SourceProvenance,
    prepare_source_case,
    prepared_source_reference,
    prepare_source_reference,
    raw_source_case,
    raw_source_scenarios,
    run_source_case,
    source_configuration,
    validate_prepared_source_samples,
    value_evidence,
)
from tests.extra_tests.converter.precision_numerical_support import (
    normalization_input,
    normalization_reference,
)
from tests.extra_tests.converter.test_source_square_variance import (
    _BOUNDS,
    _assert_result,
    _exercise,
    _fresh,
    _payload,
    _square_precision_witness,
    source_settings,  # noqa: F401
)


_ROOT = Path(__file__).resolve().parents[3]
_ORIGINAL_DEFAULTS = {
    "plugin::nnx.layer_norm": {6, 10, 11, 13, 14},
    "plugin::linen.layer_norm": {4, 5, 6},
    "plugin::eqx.nn.layer_norm": {4, 5, 6, 7},
}
_ORIGINAL_OPTIONS = {
    "plugin::nnx.layer_norm": {0, 1, 2, 3, 4, 5, 7, 8, 9, 12},
    "plugin::linen.layer_norm": {0, 1, 2, 3},
    "plugin::eqx.nn.layer_norm": {0, 1, 2, 3},
}
_ROW_COUNTS = {
    "plugin::nnx.layer_norm": 15,
    "plugin::linen.layer_norm": 7,
    "plugin::eqx.nn.layer_norm": 8,
}
_PATTERNS = ("cancellation", "constant", "signed-zero-nan")
_FAULTS = (
    "weight",
    "bias",
    "epsilon",
    "axes",
    "square",
    "cast",
    "swapped-producer",
    "disconnected-output",
    "batch-scope",
    "model-opset",
    "report-opset",
    "raw-contract",
    "stored-trace",
    "weak-type",
    "artifact-after-audit",
)


@pytest.mark.parametrize(
    "metadata",
    [
        "complete",
        "missing-shape-helper-types",
        "wrong-dtype",
        "wrong-shape",
        "wrong-axis",
        "wrong-arity",
        "constant-input",
        "unknown-shape-producer",
        "missing-arithmetic-type",
    ],
)
def test_normalization_shape_helper_metadata(metadata: str) -> None:
    """Missing helper annotations need exact INT64 def-use, never a data waiver."""
    port = onnx.helper.make_tensor_value_info
    node = onnx.helper.make_node
    data = "alias" if metadata == "missing-arithmetic-type" else "x"
    nodes = ([node("Identity", ["x"], [data])] if data != "x" else []) + [
        node("Shape", [data], ["dimensions"]),
        node("Gather", ["dimensions", "index"], ["batch"], axis=0),
        node("Unsqueeze", ["batch", "axis"], ["promoted"]),
        node("Concat", ["promoted", "tail"], ["shape"], axis=0),
        node("Reshape", [data, "shape"], ["y"]),
    ]
    helper_values = [
        port("dimensions", onnx.TensorProto.INT64, [2]),
        port("batch", onnx.TensorProto.INT64, []),
        (
            port("promoted", onnx.TensorProto.INT64, [1])
            if metadata == "complete"
            else onnx.ValueInfoProto(name="promoted")
        ),
        (
            port("shape", onnx.TensorProto.INT64, [2])
            if metadata == "complete"
            else onnx.ValueInfoProto(name="shape")
        ),
    ]
    if metadata == "wrong-dtype":
        helper_values[2].CopyFrom(port("promoted", onnx.TensorProto.INT32, [1]))
    elif metadata == "wrong-shape":
        helper_values[2].CopyFrom(port("promoted", onnx.TensorProto.INT64, [2]))
    elif metadata == "wrong-axis":
        nodes[3].attribute[0].i = 1
    elif metadata == "wrong-arity":
        nodes[2].input.pop()
    elif metadata == "unknown-shape-producer":
        nodes[2].op_type = "Identity"
    elif metadata == "constant-input":
        # x remains connected to both Shape and Reshape, so an ignored illegal
        # Constant input cannot be caught by the disconnected-input guard alone.
        nodes.insert(
            0,
            node(
                "Constant",
                ["x"],
                ["axis"],
                value=onnx.numpy_helper.from_array(np.asarray([0], np.int64)),
            ),
        )
    elif metadata == "missing-arithmetic-type":
        helper_values.append(onnx.ValueInfoProto(name=data))
    graph = onnx.helper.make_graph(
        nodes,
        "exact_shape_metadata",
        [port("x", onnx.TensorProto.FLOAT, ["B", 3])],
        [port("y", onnx.TensorProto.FLOAT, ["B", 3])],
        initializer=[
            onnx.numpy_helper.from_array(np.asarray(0, np.int64), name="index"),
            onnx.numpy_helper.from_array(np.asarray([3], np.int64), name="tail"),
        ]
        + (
            [onnx.numpy_helper.from_array(np.asarray([0], np.int64), name="axis")]
            if metadata != "constant-input"
            else []
        ),
        value_info=helper_values,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
    )
    original = model.SerializeToString()
    if metadata in {"complete", "missing-shape-helper-types"}:
        assert _NormalizationExpressions(model).outputs() == (
            ("input", "float32", ("B", 3), 0),
        )
    else:
        with pytest.raises(ValueError):
            _NormalizationExpressions(model).outputs()
    assert model.SerializeToString() == original


@contextmanager
def _deny_runtime() -> Iterator[None]:
    attempts = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        attempts.append(True)
        raise AssertionError(
            "original native preparation/reference cannot construct ORT"
        )

    with (
        patch.object(ort.InferenceSession, "__init__", forbidden),
        patch.object(ort_state, "InferenceSession", forbidden),
    ):
        yield
    assert not attempts


def _prepare(metadata: Any, scenario: Any) -> Any:
    with _deny_runtime():
        return prepare_source_case(metadata, scenario)


@contextmanager
def _retain_export(path: Path) -> Iterator[list[Any]]:
    """Save the actual once-returned artifact before any counterpart assertions."""
    original = source_support._export_prepared
    returned = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        prepared = args[0]
        modes = []
        facade = ui._to_onnx

        def actual_facade(*call_args: Any, **call_kwargs: Any) -> Any:
            # Observe the real shared facade argument; omission must fail even
            # when the effective typed DAG happens to be the same as auto.
            modes.append(call_kwargs["normalization_mode"])
            return facade(*call_args, **call_kwargs)

        with patch.object(ui, "_to_onnx", actual_facade):
            model = original(*args, **kwargs)
        assert modes == [prepared.metadata.get("normalization_mode", "auto")]
        # Retain the returned object before evidence I/O, including on a later
        # signature/counterpart failure. No second export or evaluation occurs.
        returned.append(model)
        onnx.save_model(model, path)
        reloaded = onnx.load_model(path)
        assert reloaded.SerializeToString() == model.SerializeToString()
        return model

    with patch.object(source_support, "_export_prepared", observe):
        yield returned
    assert source_support._export_prepared is original


def _counterpart(case: CorpusCase) -> Any:
    result = counterpart_for(case.origin, case.ordinal, case.contract_sha256)
    assert any(result is item for item in NORMALIZATION_COUNTERPARTS)
    return result


def _scenario(case: CorpusCase, caller_x64: bool) -> tuple[Any, Any]:
    metadata = raw_source_case(case.origin, case.ordinal)
    counterpart = _counterpart(case)
    counterpart.check_metadata(metadata)
    scenario = raw_source_scenarios(
        metadata, SourceProvenance(case.origin, case.ordinal)
    )[int(caller_x64)]
    assert scenario.model_dtype is PRESERVE_FACTORY_DTYPE
    if any(
        not isinstance(dimension, int)
        for spec in scenario.input_specs
        for dimension in spec.shape
    ):
        scenario = replace(
            scenario,
            concrete_shapes=tuple(
                tuple(2 if dimension == "B" else dimension for dimension in spec.shape)
                for spec in scenario.input_specs
            ),
        )
    return metadata, scenario


def _reviewed_criterion(prepared: Any) -> Any:
    # The complete criterion map is declared before targets. Selecting by the
    # original native result dtype preserves preexisting opaque Eqx parameters.
    # Raw missing/one-sided criteria remain unchanged in metadata and the ledger.
    bounds = _BOUNDS["layer_norm"][prepared.traced_outputs[0].dtype]
    criteria = SampleCriteria("tolerant", rtol=bounds[0], atol=bounds[1])
    return replace(
        prepared,
        scenario=replace(prepared.scenario, criteria=criteria),
        criteria=criteria,
    )


def _values(shape: tuple[int, ...], dtype: Any, pattern: str) -> np.ndarray:
    if pattern == "constant":
        return np.full(shape, 0.75, dtype=dtype)
    words = {
        "cancellation": (4096, -4096, 1, -1, 2, -2, 0.125, -0.125),
        "signed-zero-nan": (-0.0, 0.0, -2, 3, 1, -1, np.nan, 0.0),
    }[pattern]
    return np.resize(np.asarray(words, dtype=dtype), shape)


def _references(prepared: Any) -> tuple[Any, ...]:
    assert len(prepared.input_specs) == 1 and not prepared.named_specs
    source = prepared.traced_inputs[0]
    symbolic = any(isinstance(dimension, str) for dimension in source.shape)
    references = [prepared_source_reference(prepared)]
    for batch in (2, 5) if symbolic else (None,):
        shape = tuple(
            batch if dimension == "B" else dimension for dimension in source.shape
        )
        for pattern in _PATTERNS:
            with _deny_runtime():
                references.append(
                    prepare_source_reference(
                        prepared,
                        label=f"B={batch}:{pattern}",
                        input_values=(_values(shape, source.dtype, pattern),),
                    )
                )
    return tuple(references)


def _native_preservation(
    prepared: Any, references: Any, directory: Path
) -> tuple[Any, ...]:
    """Compare matching native roles; eager/JIT equality is not assumed."""
    compiled_references = []
    original_constants = tuple(
        _payload(value) for value in prepared.original_jaxpr.consts
    )
    with _deny_runtime(), source_configuration(prepared.scenario):
        for index, reference in enumerate(references):
            outputs, tree = jax.tree.flatten(
                jax.jit(_fresh(prepared.fn))(*reference.arguments)
            )
            assert str(tree) == prepared.output_tree
            copied = tuple(np.asarray(value).copy() for value in outputs)
            for value in copied:
                value.setflags(write=False)
            compiled_references.append(
                replace(
                    reference,
                    label=reference.label + ":native-jit",
                    reference_outputs=copied,
                    reference_evidence=tuple(
                        value_evidence(value, properties)
                        for value, properties in zip(
                            copied, prepared.traced_outputs, strict=True
                        )
                    ),
                )
            )
            np.savez(
                directory / f"native-{index}.npz",
                **{
                    f"input-{i}": np.asarray(value)
                    for i, value in enumerate(reference.arguments)
                },
                **{
                    f"eager-{i}": value
                    for i, value in enumerate(reference.reference_outputs)
                },
                **{f"jit-{i}": value for i, value in enumerate(copied)},
            )
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(_fresh(prepared.fn))(*prepared.all_input_specs)
            assert json.loads(
                json.dumps(typed_trace_manifest(activated))
            ) == json.loads(prepared.original_trace_json)
            assert (
                tuple(_payload(value) for value in activated.consts)
                == original_constants
            )
            for eager, compiled in zip(references, compiled_references, strict=True):
                for runner, expected in (
                    (prepared.fn, eager),
                    (jax.jit(_fresh(prepared.fn)), compiled),
                ):
                    actual, tree = jax.tree.flatten(runner(*eager.arguments))
                    assert str(tree) == prepared.output_tree
                    assert tuple(_payload(value) for value in actual) == tuple(
                        _payload(value) for value in expected.reference_outputs
                    )
    assert (
        tuple(_payload(value) for value in prepared.original_jaxpr.consts)
        == original_constants
    )
    return tuple(compiled_references)


_NATIVE_ZERO_CONFLICT_ROWS = {
    ("plugin::linen.layer_norm", 3, False): (
        "layer_norm_default_epsilon",
        "ff13993a7b3a5172ae874ab70056b6c46839631a9b2dbee0448bfca8f43a5f8b",
    ),
    ("plugin::linen.layer_norm", 3, True): (
        "layer_norm_default_epsilon",
        "ff13993a7b3a5172ae874ab70056b6c46839631a9b2dbee0448bfca8f43a5f8b",
    ),
    ("plugin::nnx.layer_norm", 7, False): (
        "layer_norm_symbolic_batch_seq10_feat3",
        "15fd8a32e70652890df09b2f530d76b764c1a0525b8d71bc56fd4e0c87e83595",
    ),
    ("plugin::nnx.layer_norm", 7, True): (
        "layer_norm_symbolic_batch_seq10_feat3",
        "15fd8a32e70652890df09b2f530d76b764c1a0525b8d71bc56fd4e0c87e83595",
    ),
    ("plugin::nnx.layer_norm", 8, False): (
        "layer_norm_symbolic_batch_seq10_feat3_2",
        "8fc138d204be5a200657ade547ea552d3a0843caf406140a162859b5e6561681",
    ),
    ("plugin::nnx.layer_norm", 8, True): (
        "layer_norm_symbolic_batch_seq10_feat3_2",
        "8fc138d204be5a200657ade547ea552d3a0843caf406140a162859b5e6561681",
    ),
}
_NATIVE_ZERO_CONFLICT_SAMPLES = {
    "B=2:signed-zero-nan": (
        3,
        (2, 10, 3),
        (16, 40),
        "49eb7279a82f4a27d47cebe9193868037c24c70f02b02eec746f55011a2f20ff",
    ),
    "B=5:signed-zero-nan": (
        6,
        (5, 10, 3),
        (16, 40, 64, 88, 112, 136),
        "77a65a6fc3c86bdc94fbc1922f07b5178a0d2a9e4c8537cb678b11f968460906",
    ),
}


def _native_zero_conflict_mask(eager: np.ndarray, compiled: np.ndarray) -> np.ndarray:
    assert eager.dtype == compiled.dtype and eager.shape == compiled.shape
    for predicate in (np.isnan, np.isposinf, np.isneginf, np.isfinite):
        np.testing.assert_array_equal(predicate(eager), predicate(compiled))
    assert eager.dtype.kind == "f" and eager.dtype.itemsize in (2, 4, 8)
    unsigned = {2: np.uint16, 4: np.uint32, 8: np.uint64}[eager.dtype.itemsize]
    return (
        np.isfinite(eager)
        & np.isfinite(compiled)
        & (eager == 0)
        & (compiled == 0)
        & (
            np.ascontiguousarray(eager).view(unsigned)
            != np.ascontiguousarray(compiled).view(unsigned)
        )
    )


def _qualify_native_zero_reference(
    case: CorpusCase,
    caller_x64: bool,
    sample_index: int,
    label: str,
    arguments: Any,
    eager_outputs: Any,
    compiled_outputs: Any,
) -> tuple[np.ndarray, ...]:
    """Admit only the six independently surveyed native contradictions, before export."""
    assert type(caller_x64) is bool
    row = _NATIVE_ZERO_CONFLICT_ROWS.get((case.origin, case.ordinal, caller_x64))
    if row is not None:
        assert (case.testcase, case.contract_sha256) == row
    witness = _NATIVE_ZERO_CONFLICT_SAMPLES.get(label) if row is not None else None
    assert len(eager_outputs) == len(compiled_outputs)
    if witness is not None:
        index, shape, positions, input_sha256 = witness
        assert sample_index == index and len(arguments) == len(eager_outputs) == 1
        argument = np.asarray(arguments[0])
        assert argument.dtype == np.dtype("float32") and argument.shape == shape
        assert hashlib.sha256(argument.tobytes(order="C")).hexdigest() == input_sha256
    masks = []
    for output_index, (eager, compiled) in enumerate(
        zip(eager_outputs, compiled_outputs, strict=True)
    ):
        eager, compiled = np.asarray(eager), np.asarray(compiled)
        conflict = _native_zero_conflict_mask(eager, compiled)
        expected = np.zeros(eager.shape, dtype=bool)
        if witness is not None:
            assert output_index == 0 and eager.dtype == np.dtype("float32")
            assert eager.shape == shape
            expected.flat[list(positions)] = True
            for value, word in ((eager, 0), (compiled, 2147483648)):
                words = np.ascontiguousarray(value).view(np.uint32).reshape(-1)
                np.testing.assert_array_equal(words[list(positions)], word)
        np.testing.assert_array_equal(conflict, expected)
        conflict.setflags(write=False)
        masks.append(conflict)
    return tuple(masks)


def _qualify_native_zero_references(
    case: CorpusCase,
    caller_x64: bool,
    prepared: Any,
    references: Any,
    compiled_references: Any,
    directory: Path,
) -> tuple[tuple[np.ndarray, ...], ...]:
    masks, samples = [], []
    for index, (eager, compiled) in enumerate(
        zip(references, compiled_references, strict=True)
    ):
        qualified = _qualify_native_zero_reference(
            case,
            caller_x64,
            index,
            eager.label,
            eager.arguments,
            eager.reference_outputs,
            compiled.reference_outputs,
        )
        masks.append(qualified)
        outputs = []
        for a, b, mask in zip(
            eager.reference_outputs, compiled.reference_outputs, qualified, strict=True
        ):
            positions = np.flatnonzero(mask).tolist()
            outputs.append(
                {
                    "dtype": a.dtype.name,
                    "shape": list(a.shape),
                    "eager_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                    "jit_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                    "qualified_native_zero_indices": positions,
                    "qualified_native_zero_words": [
                        {
                            "flat_index": p,
                            "eager": int(a.reshape(-1).view(np.uint32)[p]),
                            "jit": int(b.reshape(-1).view(np.uint32)[p]),
                        }
                        for p in positions
                    ],
                    "jit_zero_sign_status": (
                        "NATIVE_REFERENCE_CONFLICT_QUALIFIED"
                        if positions
                        else "NO_NATIVE_ZERO_CONFLICT"
                    ),
                }
            )
        samples.append(
            {
                "index": index,
                "label": eager.label,
                "native_npz_sha256": hashlib.sha256(
                    (directory / f"native-{index}.npz").read_bytes()
                ).hexdigest(),
                "input_sha256": [
                    hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
                    for a in eager.arguments
                ],
                "outputs": outputs,
            }
        )
    np.savez(
        directory / "native-zero-masks.npz",
        **{
            f"sample-{i}-output-{j}": mask
            for i, sample in enumerate(masks)
            for j, mask in enumerate(sample)
        },
    )
    receipt = {
        "cell": [case.origin, case.ordinal, prepared.scenario.name],
        "caller_x64": caller_x64,
        "prepared_settings": asdict(prepared.settings),
        "raw_contract_sha256": case.contract_sha256,
        "raw_criteria": {
            k: prepared.metadata[k] for k in ("rtol", "atol") if k in prepared.metadata
        },
        "approved_native_survey_audit_sha256": "f5850ce870a75fecbbc4e684b201b0e3a1bd5a1a181012f5235cb02a19ec670c",
        "approved_qualification_plan_sha256": "15f7a1b4ce58552cd08b373eede82f22a38404c7bb973e30d1d1e32b54310aa8",
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_preparation_sha256": hashlib.sha256(
            Path(source_support.__file__).read_bytes()
        ).hexdigest(),
        "registration_module_sha256": hashlib.sha256(
            (
                _ROOT
                / {
                    "plugin::eqx.nn.layer_norm": "jax2onnx/plugins/equinox/eqx/nn/layer_norm.py",
                    "plugin::linen.layer_norm": "jax2onnx/plugins/flax/linen/layer_norm.py",
                    "plugin::nnx.layer_norm": "jax2onnx/plugins/flax/nnx/layer_norm.py",
                }[case.origin]
            ).read_bytes()
        ).hexdigest(),
        "original_trace_sha256": hashlib.sha256(
            prepared.original_trace_json.encode()
        ).hexdigest(),
        "original_constants": [
            hashlib.sha256(_payload(a)[2]).hexdigest()
            for a in prepared.original_jaxpr.consts
        ],
        "masks_npz_sha256": hashlib.sha256(
            (directory / "native-zero-masks.npz").read_bytes()
        ).hexdigest(),
        "samples": samples,
        "qualification_before_export": True,
        "target_dependent_exception": False,
        "native_output_hashes_are_provenance_only": True,
    }
    (directory / "native-zero-qualification.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return tuple(masks)


def _assert_qualified_reference_zeros(
    actual: Any, eager: Any, compiled: Any, qualified_mask: np.ndarray
) -> None:
    actual, eager, compiled = (
        np.asarray(actual),
        np.asarray(eager),
        np.asarray(compiled),
    )
    assert actual.dtype == eager.dtype == compiled.dtype
    assert actual.shape == eager.shape == compiled.shape == qualified_mask.shape
    assert qualified_mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(
        qualified_mask, _native_zero_conflict_mask(eager, compiled)
    )
    eager_zeros = np.isfinite(eager) & (eager == 0)
    _assert_result(actual[eager_zeros], eager[eager_zeros], "exact")
    jit_zeros = np.isfinite(compiled) & (compiled == 0) & ~qualified_mask
    _assert_result(actual[jit_zeros], compiled[jit_zeros], "exact")


@pytest.mark.parametrize(
    "control",
    [
        "valid",
        "forged-mask",
        "nonzero-subnormal",
        "reversed-native-words",
        "extra-native-conflict",
        "target-wrong-eager-sign",
        "numeric-equal-input-byte-change",
    ],
)
def test_native_zero_qualification_requires_original_witness(control: str) -> None:
    """Corrupt an actual saved native witness without constructing or running a model."""
    payload = (
        Path(__file__).parent / "fixtures/normalization_native_zero_reference.json"
    ).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "05057ab8b906a5306198b5bb352c1b18b27b931ed209e720e83892ba1bc3d784"
    )
    fixture = json.loads(payload)
    arrays = {}
    for name, declaration in fixture["arrays"].items():
        data = bytes.fromhex(declaration["raw_hex"])
        assert hashlib.sha256(data).hexdigest() == declaration["sha256"]
        arrays[name] = np.frombuffer(data, dtype=declaration["dtype"]).reshape(
            declaration["shape"]
        )
    witness = fixture["witness"]
    case = CorpusCase(
        witness["origin"],
        witness["ordinal"],
        witness["testcase"],
        witness["contract_sha256"],
    )

    def qualify(argument: Any, eager: Any, compiled: Any) -> np.ndarray:
        return _qualify_native_zero_reference(
            case,
            witness["caller_x64"],
            witness["sample_index"],
            witness["label"],
            (argument,),
            (eager,),
            (compiled,),
        )[0]

    argument, eager, compiled = (arrays[name] for name in ("input", "eager", "jit"))
    mask = qualify(argument, eager, compiled)
    _assert_qualified_reference_zeros(eager, eager, compiled, mask)
    if control == "valid":
        return
    if control == "forged-mask":
        forged = mask.copy()
        forged.reshape(-1)[4] = True
        forged.setflags(write=False)
        with pytest.raises(AssertionError):
            _assert_qualified_reference_zeros(eager, eager, compiled, forged)
        return
    if control == "target-wrong-eager-sign":
        target = eager.copy()
        target.view(np.uint32).reshape(-1)[16] = 2147483648
        for reference in (eager, compiled):
            _assert_result(target, reference, _BOUNDS["layer_norm"]["float32"])
        with pytest.raises(AssertionError):
            _assert_qualified_reference_zeros(target, eager, compiled, mask)
        return
    changed_argument, changed_eager, changed_compiled = (
        value.copy() for value in (argument, eager, compiled)
    )
    if control == "nonzero-subnormal":
        changed_compiled.view(np.uint32).reshape(-1)[16] = 1
    elif control == "reversed-native-words":
        changed_eager.view(np.uint32).reshape(-1)[16] = 2147483648
        changed_compiled.view(np.uint32).reshape(-1)[16] = 0
    elif control == "extra-native-conflict":
        changed_compiled.view(np.uint32).reshape(-1)[4] = 2147483648
    else:
        assert control == "numeric-equal-input-byte-change"
        assert changed_argument.view(np.uint32).reshape(-1)[0] == 2147483648
        changed_argument.view(np.uint32).reshape(-1)[0] = 0
        np.testing.assert_array_equal(changed_argument, argument)
        assert changed_argument.tobytes() != argument.tobytes()
    with pytest.raises(AssertionError):
        qualify(changed_argument, changed_eager, changed_compiled)


@pytest.mark.parametrize(
    "case",
    NORMALIZATION_CASES,
    ids=lambda case: f"{case.origin}::{case.ordinal}::{case.testcase}",
)
@pytest.mark.parametrize("caller_x64", [False, True], ids=["raw-x64-off", "raw-x64-on"])
def test_original_default_layer_norm_family_source(
    case: CorpusCase, caller_x64: bool, tmp_path: Path, record_property: Any
) -> None:
    metadata, scenario = _scenario(case, caller_x64)
    prepared = _reviewed_criterion(_prepare(metadata, scenario))
    scenario = prepared.scenario
    counterpart = _counterpart(case)
    counterpart.check_source(prepared)
    original_fn = prepared.fn
    references = _references(prepared)
    compiled_references = _native_preservation(prepared, references, tmp_path)
    native_zero_masks = _qualify_native_zero_references(
        case, caller_x64, prepared, references, compiled_references, tmp_path
    )
    path = tmp_path / "original-layer-norm.onnx"
    with _deny_runtime(), _retain_export(path) as returned:
        result = run_source_case(
            metadata,
            scenario,
            _prepared=prepared,
            _corpus_counterpart=counterpart,
            optimization=None,
        )
    assert result.audit.status is CheckStatus.VALIDATED, result.audit
    assert result.schema is CheckStatus.VALIDATED and result.corpus_structure.accepted
    assert len(returned) == 1 and result.model is returned[0]
    assert result.prepared is prepared and prepared.fn is original_fn
    model = onnx.load_model(path)
    onnx.checker.check_model(model, full_check=True)
    assert counterpart.check(prepared, model).accepted
    result = replace(result, model=model)
    payload = model.SerializeToString()
    report = get_precision_report(model)
    assert report.opset == int(metadata.get("opset_version", 23))
    observations = []
    original_init = ort.InferenceSession.__init__
    original_run = ort.InferenceSession.run
    for optimization in ("disabled", "all"):
        actual_sessions = []
        actual_runs = []

        def observe_init(session: Any, *args: Any, **kwargs: Any) -> None:
            original_init(session, *args, **kwargs)
            actual_sessions.append(session)

        def observe_run(
            session: Any, output_names: Any, feed: Any, *args: Any, **kwargs: Any
        ) -> Any:
            outputs = original_run(session, output_names, feed, *args, **kwargs)
            saved_feed = {
                name: np.asarray(value).copy() for name, value in feed.items()
            }
            saved_outputs = tuple(np.asarray(value).copy() for value in outputs)
            np.savez(
                tmp_path / f"target-{optimization}-{len(actual_runs)}.npz",
                **{f"input-{name}": value for name, value in saved_feed.items()},
                **{f"output-{i}": value for i, value in enumerate(saved_outputs)},
            )
            actual_runs.append(
                (
                    id(session),
                    saved_feed,
                    saved_outputs,
                )
            )
            return outputs

        with (
            patch.object(ort.InferenceSession, "__init__", observe_init),
            patch.object(ort.InferenceSession, "run", observe_run),
        ):
            checked = validate_prepared_source_samples(
                result, references, optimization=optimization
            )
        assert len(actual_runs) == len(references)
        assert len(actual_sessions) == 1
        assert len({session for session, _, _ in actual_runs}) == 1
        assert all(session == id(actual_sessions[0]) for session, _, _ in actual_runs)
        validation = checked.validation
        validation.require_sample_acceptance()
        assert validation.target.providers == ("CPUExecutionProvider",)
        assert (
            validation.target.intra_op_threads
            == validation.target.inter_op_threads
            == 1
        )
        for (_, feed, outputs), eager, compiled, qualified_masks in zip(
            actual_runs, references, compiled_references, native_zero_masks, strict=True
        ):
            assert len(feed) == len(eager.arguments)
            for value, argument in zip(feed.values(), eager.arguments, strict=True):
                assert _payload(value) == _payload(argument)
            for role, reference in (
                ("original-eager", eager),
                ("original-fresh-jit", compiled),
            ):
                for actual, expected in zip(
                    outputs, reference.reference_outputs, strict=True
                ):
                    _assert_result(
                        actual, expected, _BOUNDS["layer_norm"][expected.dtype.name]
                    )
                    if "rtol" in metadata and "atol" in metadata:
                        # Preserve the original harness's reference-first direction.
                        np.testing.assert_allclose(
                            expected,
                            actual,
                            rtol=metadata["rtol"],
                            atol=metadata["atol"],
                            equal_nan=True,
                        )
                    if ":constant" in reference.label:
                        assert np.all(expected == 0)
                observations.append(
                    {
                        "optimization": optimization,
                        "sample": eager.label,
                        "reference": role,
                        "status": (
                            "NATIVE_ZERO_REFERENCE_QUALIFIED"
                            if role == "original-fresh-jit"
                            and any(np.any(mask) for mask in qualified_masks)
                            else "PASSED"
                        ),
                        "numeric_and_nonfinite_checks": "PASSED",
                        "qualified_native_zero_indices": [
                            np.flatnonzero(mask).tolist() for mask in qualified_masks
                        ],
                        "dtype_bound": _BOUNDS["layer_norm"][
                            prepared.traced_outputs[0].dtype
                        ],
                        "dtype_bound_direction": "actual,reference",
                        "raw_pair": (
                            {
                                "rtol": metadata["rtol"],
                                "atol": metadata["atol"],
                                "direction": "reference,actual",
                            }
                            if "rtol" in metadata and "atol" in metadata
                            else None
                        ),
                    }
                )
            for actual, eager_output, jit_output, mask in zip(
                outputs,
                eager.reference_outputs,
                compiled.reference_outputs,
                qualified_masks,
                strict=True,
            ):
                _assert_qualified_reference_zeros(
                    actual, eager_output, jit_output, mask
                )
        assert model.SerializeToString() == payload
        assert prepared.fn is original_fn
    receipt = {
        "cell": [case.origin, case.ordinal, scenario.name],
        "raw_contract_sha256": case.contract_sha256,
        "raw_callback_status": result.raw_callback_status,
        "requested_normalization_mode": metadata.get("normalization_mode", "auto"),
        "representation_decision": [
            item for item in report.adaptations if item.startswith("normalization:")
        ],
        "source_counterpart": asdict(result.corpus_structure),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_trace": json.loads(prepared.original_trace_json),
        "native_zero_qualification_sha256": hashlib.sha256(
            (tmp_path / "native-zero-qualification.json").read_bytes()
        ).hexdigest(),
        "native_constants": [
            _payload(value)[:2] + (hashlib.sha256(_payload(value)[2]).hexdigest(),)
            for value in prepared.original_jaxpr.consts
        ],
        "samples": [
            {
                "label": reference.label,
                "inputs": [asdict(value) for value in reference.input_evidence],
                "eager_outputs": [
                    asdict(value) for value in reference.reference_evidence
                ],
                "jit_outputs": [asdict(value) for value in compiled.reference_evidence],
            }
            for reference, compiled in zip(references, compiled_references, strict=True)
        ],
        "sessions": 2,
        "runs": 2 * len(references),
        "observations": observations,
    }
    (tmp_path / "family-evidence.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    record_property(
        "normalization-family-evidence", json.dumps(receipt, sort_keys=True)
    )


def test_layer_norm_family_inventory_and_open_options(
    tmp_path: Path, record_property: Any
) -> None:
    inventory_path = _ROOT / "internal_docs/precision/source_migration_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    plan = probe.build_probe_plan(
        inventory,
        baseline_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
    )
    assert len(plan.rows) == len(inventory["cases"]) == 1852
    assert len(plan.cells) == 2 * len(plan.rows) == 3704
    # Frozen origin/ordinal + full registration/row/contract mapping of the
    # original core, independent of the newly retained optional obligations.
    core = [
        [inventory["registrations"][row[0]], row[1:5], inventory["contracts"][row[5]]]
        for row in inventory["cases"]
        if inventory["registrations"][row[0]][2]
        not in {"examples.maxtext", "examples.maxdiffusion"}
    ]
    assert len(core) == 1802
    assert (
        hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == "7fa9b3891d80bd68893268fe916e0185acb82f6f6e02161f824b939c08e3ec89"
    )

    assert len(inventory["unresolved_recipes"]) == 63
    defaults = {
        (origin, ordinal)
        for origin, ordinals in _ORIGINAL_DEFAULTS.items()
        for ordinal in ordinals
    }
    options = {
        (origin, ordinal)
        for origin, ordinals in _ORIGINAL_OPTIONS.items()
        for ordinal in ordinals
    }
    assert len(defaults) == 12 and len(options) == 18 and not defaults & options
    selected = {(case.origin, case.ordinal) for case in NORMALIZATION_CASES}
    assert selected == defaults | options
    rows = [row for row in plan.rows if row.origin in _ROW_COUNTS]
    assert len(rows) == len(selected) == 30
    assert all(
        {row.ordinal for row in rows if row.origin == origin} == set(range(count))
        for origin, count in _ROW_COUNTS.items()
    )
    assert sum((cell.origin, cell.ordinal) in defaults for cell in plan.cells) == 24
    assert sum((cell.origin, cell.ordinal) in options for cell in plan.cells) == 36
    unselected_cells = sum(cell.origin not in _ROW_COUNTS for cell in plan.cells)
    assert unselected_cells == len(plan.cells) - 2 * len(selected) == 3644
    contracts = []
    option_cells = []
    for row in rows:
        metadata = raw_source_case(row.origin, row.ordinal)
        counterpart = counterpart_for(row.origin, row.ordinal, row.contract_sha256)
        assert any(counterpart is item for item in NORMALIZATION_COUNTERPARTS)
        counterpart.check_metadata(metadata)
        unresolved: list[str] = []
        contract = canonical_contract(
            {
                key: value
                for key, value in metadata.items()
                if key not in {"testcase", "description", "context", "component"}
            },
            unresolved=unresolved,
        )
        assert fingerprint(contract) == row.contract_sha256
        scenarios = raw_source_scenarios(
            metadata, SourceProvenance(row.origin, row.ordinal)
        )
        assert all(
            scenario.criteria is None and scenario.model_dtype is PRESERVE_FACTORY_DTYPE
            for scenario in scenarios
        )
        mode = metadata.get("normalization_mode", "auto")
        assert mode == (
            "auto" if (row.origin, row.ordinal) in defaults else "prefer_native"
        )
        contracts.append(
            {
                "origin": row.origin,
                "ordinal": row.ordinal,
                "contract": contract,
                "raw_rtol": metadata.get("rtol"),
                "raw_atol": metadata.get("atol"),
                "unresolved": unresolved,
            }
        )
        if (row.origin, row.ordinal) in options:
            option_cells.extend(
                {
                    "cell": [row.origin, row.ordinal, scenario.name],
                    "requested_mode": mode,
                    "status": "scheduled-no-runtime-claim",
                }
                for scenario in scenarios
            )
    assert len(option_cells) == 36
    (tmp_path / "complete-raw-plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "raw_rows": 30,
        "raw_cells": 60,
        "selected_rows": 30,
        "selected_cells": 60,
        "preserved_default_rows": 12,
        "preserved_default_cells": 24,
        "new_option_rows": 18,
        "new_option_cells": option_cells,
        "complete_raw_contracts": contracts,
        "full_denominator": [len(plan.rows), len(plan.cells)],
        "opaque_records": 63,
        "unselected_cells_unchanged": unselected_cells,
        "acceptance": "Only retained per-cell family evidence establishes runtime results",
    }
    (tmp_path / "family-ledger.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    record_property("normalization-family-ledger", json.dumps(receipt, sort_keys=True))


@onnx_function(name="SharedBatchOriginalNormalizationConsumer")
class _NamedOriginalNormalizationConsumer:
    def __init__(self, original: Any) -> None:
        self.original = original

    def __call__(self, x: Any) -> Any:
        return x + self.original(x) + np.float32(0.125)


def _rounding_consumer(x: Any) -> Any:
    flattened = jax.lax.reshape(x, (x.shape[0], 16))
    return _square_precision_witness(flattened)


@onnx_function(name="SharedBatchRoundedVarianceConsumer")
class _NamedRoundedVarianceConsumer:
    def __call__(self, x: Any) -> Any:
        return _rounding_consumer(x)


def _consumer_structure(
    model: Any, boundary: str, captured: Any, expected: Any
) -> None:
    """Reuse the finite normalization DAG oracle on actual serialized bodies."""
    if boundary == "inline":
        assert not captured and not model.functions
        assert _NormalizationExpressions(
            model
        ).outputs() == _NormalizationExpressions.native(expected)
        return
    assert len(captured) == len(model.functions) == 1
    body = captured[0]
    function = model.functions[0]
    root_imports = {item.domain: item.version for item in model.opset_import}
    imports = {item.domain: item.version for item in function.opset_import}
    assert len(root_imports) == len(model.opset_import)
    assert len(imports) == len(function.opset_import)
    assert root_imports.get("") == imports.get("") == 23
    # Function construction retains an unused self-domain import at the
    # standard opset version, or version1 with IR 1.1. The root still resolves
    # the actual custom Function call through its own import.
    definitions = {(item.domain, item.name) for item in model.functions}
    used_domains = {node.domain for node in function.node}
    assert set(imports) <= {"", function.domain} | used_domains
    for domain, version in imports.items():
        if domain and domain not in used_domains:
            assert domain == function.domain and version in (1, imports[""])
            assert root_imports.get(domain, 0) > 0
        else:
            assert version == root_imports.get(domain)
            if domain:
                assert version > 0 and domain in {
                    item.domain for item in model.functions
                }
    for node in function.node:
        assert node.domain in imports
        if node.domain:
            assert (node.domain, node.op_type) in definitions
    assert len(function.input) == len(body.jaxpr.invars) == 1
    assert len(function.output) == len(body.jaxpr.outvars)

    def port(name: str, variable: Any) -> Any:
        return onnx.helper.make_tensor_value_info(
            name,
            onnx.helper.np_dtype_to_tensor_dtype(np.dtype(variable.aval.dtype)),
            normalized_shape_dimensions(variable.aval.shape),
        )

    graph = onnx.helper.make_graph(
        function.node,
        "actual_function_body_proof_copy",
        [
            port(name, variable)
            for name, variable in zip(function.input, body.jaxpr.invars, strict=True)
        ],
        [
            port(name, variable)
            for name, variable in zip(function.output, body.jaxpr.outvars, strict=True)
        ],
        value_info=function.value_info,
    )
    proof = onnx.helper.make_model(
        graph, opset_imports=function.opset_import, ir_version=model.ir_version
    )
    assert _NormalizationExpressions(
        proof
    ).outputs() == _NormalizationExpressions.native(body)
    producers = {name: node for node in model.graph.node for name in node.output}

    def origin(name: str, nodes: Any) -> str:
        seen = set()
        while name in nodes and nodes[name].op_type == "Identity":
            assert name not in seen
            seen.add(name)
            name = nodes[name].input[0]
        return name

    def call(graph: Any, operand: str, negative: bool) -> None:
        nodes = {name: node for node in graph.node for name in node.output}
        calls = [
            node
            for node in graph.node
            if (node.domain, node.op_type) == (function.domain, function.name)
        ]
        assert len(calls) == 1 and len(calls[0].input) == 1
        assert len(calls[0].output) == len(function.output) == len(graph.output)
        invocation = calls[0]
        assert all(
            (not node.domain and node.op_type == "Identity")
            or (node.domain, node.op_type) == (function.domain, function.name)
            or (negative and not node.domain and node.op_type == "Neg")
            for node in graph.node
        )
        actual_input = origin(invocation.input[0], nodes)
        if negative:
            before = nodes[actual_input]
            assert before.op_type == "Neg"
            assert len(before.input) == 1 and origin(before.input[0], nodes) == operand
        else:
            assert actual_input == operand
        for result, output in zip(invocation.output, graph.output, strict=True):
            actual_output = origin(output.name, nodes)
            if negative:
                after = nodes[actual_output]
                assert after.op_type == "Neg" and len(after.input) == 1
                assert origin(after.input[0], nodes) == result
            else:
                assert actual_output == result

    operand = model.graph.input[0].name
    if boundary == "function":
        call(model.graph, operand, False)
        return
    condition = producers[origin(model.graph.output[0].name, producers)]
    assert condition.op_type == "If" and len(condition.input) == 1
    assert len(condition.output) == len(model.graph.output)
    assert all(
        origin(port.name, producers) == name
        for port, name in zip(model.graph.output, condition.output, strict=True)
    )
    predicate = producers[origin(condition.input[0], producers)]
    assert (
        predicate.op_type == "Greater"
        and origin(predicate.input[0], producers) == model.graph.input[1].name
    )
    constants = {
        value.name: onnx.numpy_helper.to_array(value)
        for value in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type == "Constant":
            constants[node.output[0]] = onnx.numpy_helper.to_array(
                next(
                    attribute.t
                    for attribute in node.attribute
                    if attribute.name == "value"
                )
            )
    assert _payload(constants[predicate.input[1]]) == _payload(np.float32(0))
    branches = {attribute.name: attribute.g for attribute in condition.attribute}
    assert set(branches) == {"then_branch", "else_branch"}
    assert all(not graph.input for graph in branches.values())
    call(branches["then_branch"], operand, False)
    call(branches["else_branch"], operand, True)


_CONSUMERS = tuple(
    ("original-layernorm", rank, boundary, caller_x64)
    for rank in (3, 4)
    for boundary in ("inline", "function", "cond")
    for caller_x64 in (False, True)
) + tuple(
    ("rounded-variance", rank, boundary, True)
    for rank in (3, 4)
    for boundary in ("inline", "function", "cond")
)


@pytest.mark.parametrize("consumer,rank,boundary,caller_x64", _CONSUMERS)
def test_symbolic_normalization_function_and_conditional_consumers(
    consumer: str,
    rank: int,
    boundary: str,
    caller_x64: bool,
    tmp_path: Path,
    record_property: Any,
) -> None:
    with jax.enable_x64(caller_x64):
        if consumer == "original-layernorm":
            case = next(
                case
                for case in NORMALIZATION_CASES
                if case.origin == "plugin::nnx.layer_norm"
                and case.ordinal == (11 if rank == 3 else 13)
            )
            metadata, scenario = _scenario(case, caller_x64)
            prepared = _prepare(metadata, scenario)
            named = _NamedOriginalNormalizationConsumer(prepared.fn)

            def inline(x: Any) -> Any:
                return x + prepared.fn(x) + np.float32(0.125)

            tail = tuple(prepared.traced_inputs[0].shape[1:])
            arrays = [
                _values((batch, *tail), np.float32, "cancellation") for batch in (2, 5)
            ]
            constants = prepared.original_jaxpr.consts
            criteria = _BOUNDS["layer_norm"]["float32"]
        else:
            named = _NamedRoundedVarianceConsumer()
            inline = _rounding_consumer
            tail = (*(1 for _ in range(rank - 2)), 16)
            arrays = [
                np.resize(normalization_input(), (batch, 16)).reshape((batch, *tail))
                for batch in (2, 5)
            ]
            constants = ()
            criteria = ["exact", "exact", "exact", (3e-7, 1e-7)]
            analytic = normalization_reference(normalization_input())
            np.testing.assert_array_equal(analytic[2], [[np.float32(8388609)], [0]])
            for array in arrays:
                value = array.reshape((-1, 16))
                with _deny_runtime():
                    for runner in (
                        _square_precision_witness,
                        jax.jit(_fresh(_square_precision_witness)),
                    ):
                        outputs = runner(value)
                        for actual, expected in zip(
                            outputs[:3], analytic[:3], strict=True
                        ):
                            _assert_result(
                                actual, np.resize(expected, actual.shape), "exact"
                            )
                        _assert_result(
                            outputs[3],
                            np.resize(analytic[3], outputs[3].shape),
                            (3e-7, 1e-7),
                        )

        def state() -> Any:
            return tuple(_payload(value) for value in constants)

        original_state = state()
        if boundary == "inline":
            fn = inline
        elif boundary == "function":

            def fn(x: Any) -> Any:
                return named(x)

        else:

            def fn(x: Any, selector: Any) -> Any:
                def negative(value: Any) -> Any:
                    result = named(-value)
                    return (
                        tuple(-item for item in result)
                        if consumer == "rounded-variance"
                        else -result
                    )

                return jax.lax.cond(
                    selector > np.float32(0),
                    lambda value: named(value),
                    negative,
                    x,
                )

        feeds = (
            [(value,) for value in arrays]
            if boundary != "cond"
            else [
                (value, np.asarray(selector, np.float32))
                for value in arrays
                for selector in (1, -1)
            ]
        )
        (batch,) = export.symbolic_shape("B")
        body_specs = [jax.ShapeDtypeStruct((batch, *tail), np.float32)]
        specs = (
            body_specs
            if boundary != "cond"
            else [*body_specs, jax.ShapeDtypeStruct((), np.float32)]
        )
        with _deny_runtime():
            expected = (
                jax.make_jaxpr(_fresh(inline))(*body_specs)
                if boundary == "inline"
                else None
            )
        captured = []
        original_trace_body = FunctionPlugin._trace_function_body

        def retain_body(*args: Any, **kwargs: Any) -> Any:
            result = original_trace_body(*args, **kwargs)
            captured.append(result)
            return result

        with patch.object(FunctionPlugin, "_trace_function_body", retain_body):
            model = _exercise(
                fn,
                feeds,
                tmp_path,
                criteria,
                record_property,
                declared_specs=specs,
                function_body=(
                    (named, (arrays[0],), state) if boundary != "inline" else None
                ),
                function_body_specs=(body_specs if boundary != "inline" else None),
                function_body_feeds=(
                    [(value,) for value in arrays]
                    if boundary == "function"
                    else (
                        [(value,) for array in arrays for value in (array, -array)]
                        if boundary == "cond"
                        else None
                    )
                ),
                exact_zero_output=(3 if consumer == "rounded-variance" else None),
                boundary=boundary,
                serialized_check=lambda model, observations: _consumer_structure(
                    model, boundary, captured, expected
                ),
            )
        assert FunctionPlugin._trace_function_body is original_trace_body
        assert state() == original_state
        report = get_precision_report(model)
        assert report.inputs[0].traced.shape == ("B", *tail)


@pytest.fixture(scope="module")
def _normalization_corruption_artifact(tmp_path_factory: Any) -> Any:
    case = next(
        case
        for case in NORMALIZATION_CASES
        if case.origin == "plugin::nnx.layer_norm" and case.ordinal == 13
    )
    metadata, scenario = _scenario(case, True)
    prepared = _reviewed_criterion(_prepare(metadata, scenario))
    scenario = prepared.scenario
    counterpart = _counterpart(case)
    path = tmp_path_factory.mktemp("normalization-corruption") / "source.onnx"
    with _deny_runtime(), _retain_export(path) as returned:
        result = run_source_case(
            metadata,
            scenario,
            _prepared=prepared,
            _corpus_counterpart=counterpart,
            optimization=None,
        )
    assert result.audit.status is CheckStatus.VALIDATED, result.audit
    assert len(returned) == 1 and result.model is returned[0]
    assert counterpart.check(prepared, result.model).accepted
    return prepared, counterpart, result


@pytest.mark.parametrize("corruption", _FAULTS)
def test_source_normalization_contract_corruption(
    corruption: str, _normalization_corruption_artifact: Any
) -> None:
    prepared, counterpart, result = _normalization_corruption_artifact
    original = result.model
    original_payload = original.SerializeToString()
    model = onnx.ModelProto()
    model.CopyFrom(original)
    if corruption == "artifact-after-audit":
        tensors = list(model.graph.initializer) + [
            attribute.t
            for node in model.graph.node
            if node.op_type == "Constant"
            for attribute in node.attribute
            if attribute.name == "value"
        ]
        tensor = next(
            tensor for tensor in tensors if tensor.data_type == onnx.TensorProto.FLOAT
        )
        values = onnx.numpy_helper.to_array(tensor).copy()
        values.flat[0] += np.float32(0.25)
        tensor.CopyFrom(onnx.numpy_helper.from_array(values, name=tensor.name))
        with (
            _deny_runtime(),
            pytest.raises(SourcePreparationError, match="Artifact changed after"),
        ):
            validate_prepared_source_samples(
                replace(result, model=model),
                (prepared_source_reference(prepared),),
                optimization="disabled",
            )
        assert original.SerializeToString() == original_payload
        return
    if corruption == "raw-contract":
        prepared = replace(
            prepared, metadata={**prepared.metadata, "opset_version": 16}
        )
    elif corruption == "stored-trace":
        trace = json.loads(prepared.original_trace_json)
        trace["inputs"][0][1][0] = "Other"
        prepared = replace(
            prepared, original_trace_json=json.dumps(trace, sort_keys=True)
        )
    elif corruption == "weak-type":
        prepared = replace(
            prepared,
            traced_outputs=(replace(prepared.traced_outputs[0], weak_type=True),),
        )
    elif corruption == "model-opset":
        next(item for item in model.opset_import if item.domain == "").version = 16
    elif corruption == "report-opset":
        item = next(
            item for item in model.metadata_props if item.key == PRECISION_METADATA_KEY
        )
        report = json.loads(item.value)
        report["opset"] = 16
        item.value = json.dumps(report)
    elif corruption == "batch-scope":
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "Other"
    elif corruption == "disconnected-output":
        model.graph.output[0].name = model.graph.input[0].name
    elif corruption == "swapped-producer":
        node = next(node for node in model.graph.node if node.op_type == "Sub")
        node.input[0], node.input[1] = node.input[1], node.input[0]
    elif corruption == "cast":
        node = next(node for node in model.graph.node if node.op_type == "Cast")
        next(attribute for attribute in node.attribute if attribute.name == "to").i = (
            onnx.TensorProto.DOUBLE
        )
    else:
        tensors = [(tensor.name, tensor) for tensor in model.graph.initializer] + [
            (node.output[0], attribute.t)
            for node in model.graph.node
            if node.op_type == "Constant"
            for attribute in node.attribute
            if attribute.name == "value"
        ]
        floating = [
            (name, tensor, onnx.numpy_helper.to_array(tensor))
            for name, tensor in tensors
            if tensor.data_type == onnx.TensorProto.FLOAT
        ]
        epsilon_name, epsilon = next(
            (name, tensor)
            for name, tensor, value in floating
            if value.size == 1 and 0 < float(value.reshape(-1)[0]) < 0.001
        )
        if corruption in {"weight", "bias", "epsilon"}:
            tensor = (
                epsilon
                if corruption == "epsilon"
                else next(
                    tensor
                    for _, tensor, value in floating
                    if value.size > 1
                    and np.all(value == (1 if corruption == "weight" else 0))
                )
            )
            values = onnx.numpy_helper.to_array(tensor).copy()
            values.flat[0] += np.float32(0.25)
            tensor.CopyFrom(onnx.numpy_helper.from_array(values, name=tensor.name))
        elif corruption == "square":
            node = next(
                node
                for node in model.graph.node
                if node.op_type == "Mul" and node.input[0] == node.input[1]
            )
            node.input[1] = epsilon_name
        else:
            assert corruption == "axes"
            reduction = next(
                node for node in model.graph.node if node.op_type == "ReduceSum"
            )
            tensor = next(
                tensor for name, tensor in tensors if name == reduction.input[1]
            )
            values = onnx.numpy_helper.to_array(tensor).copy()
            values.flat[0] = 0
            tensor.CopyFrom(onnx.numpy_helper.from_array(values, name=tensor.name))
    evidence = counterpart.check(prepared, model)
    assert (
        evidence.source_structure is CheckStatus.FAILED and not evidence.accepted
    ), evidence
    assert original.SerializeToString() == original_payload


@pytest.mark.parametrize(
    ("boundary", "mode"),
    [
        pytest.param("facade", "auto", id="facade-auto"),
        pytest.param("facade", "prefer_native", id="facade-prefer_native"),
        pytest.param("facade", "force_decomposed", id="facade-force_decomposed"),
        pytest.param("common", "auto", id="common-auto"),
        pytest.param("common", "prefer_native", id="common-prefer_native"),
        pytest.param("common", "force_decomposed", id="common-force_decomposed"),
        pytest.param("diagnostic", "prefer_native", id="diagnostic-prefer_native"),
    ],
)
def test_original_layer_norm_representation_request_reaches_ir_context(
    mode: str, boundary: str
) -> None:
    case = next(
        case
        for case in NORMALIZATION_CASES
        if (case.origin, case.ordinal) == ("plugin::nnx.layer_norm", 0)
    )
    metadata, scenario = _scenario(case, False)
    prepared = _prepare(metadata, scenario)
    create = conversion_api._create_ir_context
    received = []

    def actual_context(**kwargs: Any) -> Any:
        context = create(**kwargs)
        received.append((kwargs["normalization_mode"], context.normalization_mode))
        return context

    with (
        _deny_runtime(),
        source_configuration(scenario),
        patch.object(conversion_api, "_create_ir_context", actual_context),
    ):
        if boundary in {"facade", "diagnostic"}:
            # This supplemental option does not rewrite or accept any raw row.
            model = source_support._export_prepared(
                replace(prepared, metadata={**metadata, "normalization_mode": mode}),
                diagnostic_jaxpr_validator=(
                    validate_source_jaxpr if boundary == "diagnostic" else None
                ),
            )
            report = get_precision_report(model)
            decisions = tuple(
                item for item in report.adaptations if item.startswith("normalization:")
            )
            if boundary == "diagnostic":
                assert report.profile == "source-diagnostic-v1"
                assert {check.category: check.status for check in report.checks} == {
                    "representation": CheckStatus.VALIDATED,
                    "lowering": CheckStatus.UNVERIFIED,
                    "runtime-execution": CheckStatus.UNVERIFIED,
                    "numerical-acceptance": CheckStatus.UNVERIFIED,
                }
                assert decisions == (
                    "normalization: requested=prefer_native; effective=UNVERIFIED; "
                    "source-equivalence=UNVERIFIED; reason=diagnostic export has no lowering audit",
                )
            elif mode == "auto":
                assert not decisions
            else:
                reason = (
                    "native ONNX normalization has no reviewed source-equivalence proof"
                    if mode == "prefer_native"
                    else "explicit representation requested"
                )
                assert decisions == (
                    f"normalization: requested={mode}; effective=typed-native-jax-dag; "
                    f"reason={reason}; original constants, casts, accumulation and rounding retained",
                )
        else:
            import onnx_ir as ir

            model = ir.to_proto(
                conversion_api._convert_to_ir(
                    fn=prepared.fn,
                    inputs=prepared.input_specs,
                    input_params=prepared.named_specs,
                    model_name="original_layer_norm_option",
                    opset=23,
                    enable_double_precision=False,
                    record_primitive_calls_file=None,
                    normalization_mode=mode,
                    precision_policy=PrecisionPolicy.SOURCE,
                    jaxpr_validator=validate_source_jaxpr,
                )
            )
        onnx.checker.check_model(model, full_check=True)
        assert _NormalizationExpressions(
            model
        ).outputs() == _NormalizationExpressions.native(prepared.original_jaxpr)
    assert received == [(mode, mode)]
    assert (
        raw_source_case(case.origin, case.ordinal)["normalization_mode"]
        == "prefer_native"
    )


@pytest.mark.parametrize("corruption", ["omitted", "replaced"])
def test_original_option_mode_cannot_disappear_in_shared_facade(
    corruption: str,
) -> None:
    case = next(
        case
        for case in NORMALIZATION_CASES
        if (case.origin, case.ordinal) == ("plugin::nnx.layer_norm", 0)
    )
    metadata, scenario = _scenario(case, False)
    prepared = _prepare(metadata, scenario)
    facade = ui._to_onnx
    observed = []

    def corrupt(**kwargs: Any) -> Any:
        assert kwargs["normalization_mode"] == "prefer_native"
        if corruption == "omitted":
            kwargs.pop("normalization_mode")
        else:
            kwargs["normalization_mode"] = "force_decomposed"
        observed.append(kwargs.get("normalization_mode", "auto"))
        return facade(**kwargs)

    with _deny_runtime(), patch.object(ui, "_to_onnx", corrupt):
        result = run_source_case(
            metadata,
            scenario,
            _prepared=prepared,
            _corpus_counterpart=_counterpart(case),
            optimization=None,
        )
    assert observed == ["auto" if corruption == "omitted" else "force_decomposed"]
    assert result.corpus_structure.source_structure is CheckStatus.FAILED
    assert "requested/effective representation" in result.corpus_structure.source_detail
    assert result.raw_callback_status == "failed"


@pytest.mark.parametrize(
    "corruption", ["hash-alias", "weight-present", "bias-present", "affine-flags"]
)
def test_original_eqx_opaque_collision_keeps_no_affine_configuration(
    corruption: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    affine_case = next(
        case
        for case in NORMALIZATION_CASES
        if (case.origin, case.ordinal) == ("plugin::eqx.nn.layer_norm", 0)
    )
    absent_case = next(
        case
        for case in NORMALIZATION_CASES
        if (case.origin, case.ordinal) == ("plugin::eqx.nn.layer_norm", 3)
    )
    assert affine_case.contract_sha256 == absent_case.contract_sha256
    affine_raw = raw_source_case(affine_case.origin, affine_case.ordinal)
    absent_raw = raw_source_case(absent_case.origin, absent_case.ordinal)
    counterpart = _counterpart(absent_case)
    counterpart.check_metadata(absent_raw)
    original = absent_raw["callable"]
    assert original.weight is None and original.bias is None
    assert original is not affine_raw["callable"]
    if corruption == "hash-alias":
        changed = {**absent_raw, "callable": affine_raw["callable"]}
        with pytest.raises(ValueError, match="callable/callback identity"):
            counterpart.check_metadata(changed)
        return
    copied = copy.copy(original)
    if corruption == "affine-flags":
        object.__setattr__(copied, "use_weight", True)
        object.__setattr__(copied, "use_bias", True)
        message = "configuration changed"
    else:
        object.__setattr__(
            copied,
            "weight" if corruption == "weight-present" else "bias",
            np.full((32,), 1 if corruption == "weight-present" else 0, np.float32),
        )
        message = "parameter must be absent"
    changed = {**absent_raw, "callable": copied}
    raw = source_support.raw_source_case
    monkeypatch.setattr(
        source_support,
        "raw_source_case",
        lambda origin, ordinal: (
            changed
            if (origin, ordinal) == (absent_case.origin, absent_case.ordinal)
            else raw(origin, ordinal)
        ),
    )
    with pytest.raises(ValueError, match=message):
        counterpart.check_metadata(changed)
    assert original.weight is None and original.bias is None
    assert original.use_weight is False and original.use_bias is False


@pytest.fixture(scope="module", params=[7, 8], ids=["epsilon-1e-5", "epsilon-1e-6"])
def _original_epsilon_option_artifact(request: Any, tmp_path_factory: Any) -> Any:
    case = next(
        case
        for case in NORMALIZATION_CASES
        if (case.origin, case.ordinal) == ("plugin::nnx.layer_norm", request.param)
    )
    metadata, scenario = _scenario(case, False)
    prepared = _prepare(metadata, scenario)
    counterpart = _counterpart(case)
    path = tmp_path_factory.mktemp(f"original-option-{request.param}") / "source.onnx"
    with _deny_runtime(), _retain_export(path):
        result = run_source_case(
            metadata,
            scenario,
            _prepared=prepared,
            _corpus_counterpart=counterpart,
            optimization=None,
        )
    assert result.audit.status is CheckStatus.VALIDATED, result.audit
    assert counterpart.check(prepared, result.model).accepted
    return prepared, counterpart, result.model


@pytest.mark.parametrize(
    "corruption", ["other-original-epsilon", "axis", "consumer", "report-mode"]
)
def test_actual_size_three_option_keeps_epsilon_axis_and_consumer(
    corruption: str, _original_epsilon_option_artifact: Any
) -> None:
    prepared, counterpart, original = _original_epsilon_option_artifact
    model = onnx.ModelProto()
    model.CopyFrom(original)
    saved = original.SerializeToString()
    if corruption == "report-mode":
        item = next(
            item for item in model.metadata_props if item.key == PRECISION_METADATA_KEY
        )
        report = json.loads(item.value)
        index = next(
            i
            for i, value in enumerate(report["adaptations"])
            if value.startswith("normalization:")
        )
        report["adaptations"][index] = report["adaptations"][index].replace(
            "requested=prefer_native", "requested=auto"
        )
        item.value = json.dumps(report)
    elif corruption == "consumer":
        node = next(node for node in model.graph.node if node.op_type == "Sub")
        node.input[0], node.input[1] = node.input[1], node.input[0]
    else:
        tensors = [(tensor.name, tensor) for tensor in model.graph.initializer] + [
            (node.output[0], attribute.t)
            for node in model.graph.node
            if node.op_type == "Constant"
            for attribute in node.attribute
            if attribute.name == "value"
        ]
        if corruption == "axis":
            node = next(
                node for node in model.graph.node if node.op_type == "ReduceSum"
            )
            tensor = next(tensor for name, tensor in tensors if name == node.input[1])
            values = onnx.numpy_helper.to_array(tensor).copy()
            assert tuple(values.reshape(-1)) == (2,)
            values.flat[0] = 1
        else:
            expected = np.float32(1e-5 if counterpart.case.ordinal == 7 else 1e-6)
            tensor = next(
                tensor
                for _, tensor in tensors
                if tensor.data_type == onnx.TensorProto.FLOAT
                and onnx.numpy_helper.to_array(tensor).size == 1
                and np.array_equal(
                    onnx.numpy_helper.to_array(tensor).reshape(-1), [expected]
                )
            )
            values = onnx.numpy_helper.to_array(tensor).copy()
            values.flat[0] = np.float32(1e-6 if counterpart.case.ordinal == 7 else 1e-5)
        tensor.CopyFrom(onnx.numpy_helper.from_array(values, name=tensor.name))
    evidence = counterpart.check(prepared, model)
    assert (
        evidence.source_structure is CheckStatus.FAILED and not evidence.accepted
    ), evidence
    assert original.SerializeToString() == saved
