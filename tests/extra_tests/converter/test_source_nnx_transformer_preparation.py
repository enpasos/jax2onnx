# tests/extra_tests/converter/test_source_nnx_transformer_preparation.py

"""The full original NNX Transformer under an explicit softmax-only intent.

The 1e-5 asymmetric FP32-output sample contract is frozen before observations.
It is separate from raw metadata and is not an FP64 error bound. The original
FP32 calculation is retained separately from the intentionally prepared one.
"""

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import FunctionType, MethodType
from typing import Any
from unittest.mock import patch

from flax import nnx
import jax
import numpy as np
import onnx
from onnx import inliner
import onnxruntime as ort
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._source_nnx_method_preparation import (
    _PreparedMethod,
    prepare_nnx_transformer_softmax64,
)
from jax2onnx._source_preparation import SourcePreparationError
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.converter.source_deployment import (
    ort_cpu_deployment_target,
    require_runtime_target,
)
from jax2onnx.plugins.plugin_system import (
    EXAMPLE_REGISTRY,
    FunctionPlugin,
    apply_monkey_patches,
)
from jax2onnx.precision import (
    CheckStatus,
    _configure_deployment_session,
    _observe_deployment_session,
    get_precision_report,
    precision_report_to_json,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_callable_preparation import (
    _equations,
    _fresh,
    _preserve_settings as _preserve_settings,
)


_RTOL = _ATOL = 1e-5
_CONFIG = dict(
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    num_hidden_layers=2,
    num_experts=4,
    experts_per_token=2,
    intermediate_size=16,
    vocab_size=48,
)


def _original(directory: Path) -> tuple[FunctionType, Any, Any]:
    from jax2onnx.plugins.examples.nnx import gpt_oss_flax as source

    case = EXAMPLE_REGISTRY["examples.nnx_gpt_oss::FlaxTransformer"]["testcases"][0]
    assert case["testcase"] == "gpt_oss_transformer_flax"
    assert case["run_only_f32_variant"] is True
    recipe = case["callable"]
    assert recipe._ctor is source._build_transformer_model_apply
    assert recipe._args == () and recipe._instance is None and recipe._dtype is None
    assert set(recipe._kwargs) == {
        "config",
        "sequence_length",
        "table_length",
        "init_key",
        "dtype",
    }
    assert recipe._kwargs["config"] is source._MODEL_CONFIG
    assert recipe._kwargs["sequence_length"] == 3
    assert recipe._kwargs["table_length"] == 8
    assert recipe._kwargs["init_key"].metadata["rng_trace"].seed == 123
    assert recipe._kwargs["dtype"].resolve(np.float32) is np.float32
    assert all(
        asdict(source._MODEL_CONFIG)[name] == value for name, value in _CONFIG.items()
    )
    assert len(case["input_values"]) == 1
    tokens = case["input_values"][0]
    assert tokens is source._MODEL_TOKENS
    assert np.asarray(tokens).dtype == np.int32 and tokens.shape == (3,)
    np.testing.assert_array_equal(np.asarray(tokens), np.array([0, 5, 7], np.int32))
    assert case["expected_output_shapes"] == [(3, 48)]
    resolved = recipe.with_dtype(np.float32)
    returned = []
    previous_profile = sys.getprofile()

    def observe(frame: Any, event: str, value: Any) -> None:
        if (
            event == "return"
            and frame.f_code is source._build_transformer_model_apply.__code__
        ):
            returned.append(value)
        if previous_profile is not None:
            previous_profile(frame, event, value)

    sys.setprofile(observe)
    try:
        original = resolved.instantiate()
    finally:
        sys.setprofile(previous_profile)
    assert len(returned) == 1 and returned[0] is original
    owner = inspect.getclosurevars(original).nonlocals["module"]
    assert type(owner) is source.Transformer and owner.config is source._MODEL_CONFIG
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "original-recipe.json").write_text(
        json.dumps(
            dict(
                factory=source._build_transformer_model_apply.__qualname__,
                source_sha256=hashlib.sha256(
                    Path(source.__file__).read_bytes()
                ).hexdigest(),
                config=asdict(source._MODEL_CONFIG),
                sequence_length=3,
                table_length=8,
                seed=123,
                dtype="float32",
                tokens=[0, 5, 7],
                token_dtype="int32",
                factory_calls=1,
                retained_returned_callable=True,
                metadata_run_only_f32_variant=case["run_only_f32_variant"],
            ),
            indent=2,
        )
    )
    return original, owner, tokens


def _state(owner: Any) -> tuple[Any, ...]:
    graphdef, state = nnx.split(owner)
    entries = []
    for path, variable in nnx.to_flat_state(state):
        leaves, tree = jax.tree.flatten(variable)
        entries.append(
            (
                tuple(path),
                str(tree),
                tuple(
                    (
                        np.asarray(value).dtype.str,
                        tuple(value.shape),
                        hashlib.sha256(np.asarray(value).tobytes()).hexdigest(),
                    )
                    for value in leaves
                ),
            )
        )
    identities = tuple(
        (tuple(path), id(value), type(value))
        for path, value in nnx.iter_graph(owner)
        if isinstance(value, (nnx.Module, nnx.Variable))
    )
    return graphdef, tuple(entries), identities


def _roles(root: _PreparedMethod) -> tuple[_PreparedMethod, ...]:
    found = [root]
    for name, value in root.method.__globals__.items():
        if name.startswith("__jax2onnx_nnx_child_") and type(value) is MethodType:
            child = value.__self__
            assert type(child) is _PreparedMethod
            found.extend(_roles(child))
    return tuple(found)


def _array(value: Any) -> np.ndarray:
    leaves, tree = jax.tree.flatten(jax.device_get(value))
    assert str(tree) == "PyTreeDef(*)" and len(leaves) == 1
    result = np.array(leaves[0], copy=True)
    assert result.shape == (3, 48) and result.dtype == np.float32
    assert np.isfinite(result).all()
    return result


def _within(actual: np.ndarray, reference: np.ndarray) -> bool:
    assert actual.dtype == reference.dtype == np.float32
    assert actual.shape == reference.shape == (3, 48)
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    # Port dtypes are checked before higher-precision comparison arithmetic.
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    return bool(
        np.all(difference <= _ATOL + _RTOL * np.abs(reference.astype(np.float64)))
    )


def _compare(actual: np.ndarray, reference: np.ndarray) -> None:
    assert _within(actual, reference), "frozen asymmetric FP32 sample criterion failed"


def _native(function: Any, tokens: Any, directory: Path) -> tuple[np.ndarray, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []

    def event(status: str, stage: str, **details: Any) -> None:
        events.append(dict(status=status, stage=stage, **details))
        (directory / "native-events.json").write_text(json.dumps(events, indent=2))

    phase = "create-new-jit"
    outputs: dict[str, np.ndarray] = {}
    try:
        compiled = jax.jit(_fresh(function))
        # The first execution is a new JIT; no eager/typed-trace priming occurs.
        for label, operation in (
            ("fresh", compiled),
            ("warmed", compiled),
            ("eager", function),
        ):
            phase = f"{label}-call"
            value = operation(tokens)
            # Record the actual return before conversion, validation or another call.
            event("returned", phase, value_type=type(value).__qualname__)
            phase = f"{label}-archive"
            leaves, tree = jax.tree.flatten(jax.device_get(value))
            np.savez(
                directory / f"native-{label}-returned.npz",
                **{f"leaf_{index}": leaf for index, leaf in enumerate(leaves)},
            )
            event("archived", phase, tree=str(tree), leaves=len(leaves))
            phase = f"{label}-validation"
            outputs[label] = _array(value)
            np.savez(directory / "native-outputs.npz", **outputs)
            event("validated", phase)
        phase = "fresh-vs-eager"
        _compare(outputs["fresh"], outputs["eager"])
        event("passed", phase)
        phase = "warmed-vs-eager"
        _compare(outputs["warmed"], outputs["eager"])
        event("passed", phase)
        phase = "zero-corruption-control"
        assert not _within(
            np.zeros_like(outputs["eager"]), outputs["eager"]
        ), "original sample must expose zero corruption"
        event("passed", phase)
        phase = "typed-trace"
        trace = jax.make_jaxpr(_fresh(function))(tokens)
        (directory / "typed-trace.json").write_text(
            json.dumps(typed_trace_manifest(trace), indent=2)
        )
        event("retained", phase)
    except BaseException as exc:
        event(
            "raised",
            phase,
            exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            message=str(exc),
        )
        raise
    return outputs["eager"], outputs["fresh"], outputs["warmed"]


def _softmax_trace(function: Any, tokens: Any, expected: Any) -> None:
    trace = jax.make_jaxpr(_fresh(function))(tokens)
    exps = [
        equation for equation in _equations(trace) if equation.primitive.name == "exp"
    ]
    assert len(exps) == 4  # Attention and MoE softmax in each of both original blocks.
    assert all(np.dtype(e.outvars[0].aval.dtype) == expected for e in exps)
    if expected == np.float64:
        conversions = [
            e
            for e in _equations(trace)
            if e.primitive.name == "convert_element_type"
            and np.dtype(e.params["new_dtype"]) == np.float32
        ]
        assert len(conversions) >= 4


@pytest.mark.parametrize(
    "order", [("original", "prepared", "second"), ("second", "original", "prepared")]
)
def test_full_original_transformer_native_prepared_state_and_same_warmed_jit(
    order: tuple[str, ...], tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        original, owner, tokens = _original(tmp_path)
        baseline = _state(owner)
        prepared, record = prepare_nnx_transformer_softmax64(original)
        second, second_record = prepare_nnx_transformer_softmax64(original)
        assert (
            prepared is not second
            and record.root.method is not second_record.root.method
        )
        assert (
            record.prepared_owner is not owner
            and second_record.prepared_owner is not owner
        )
        assert record.prepared_owner is not second_record.prepared_owner
        snapshots = {
            "original": baseline,
            "prepared": _state(record.prepared_owner),
            "second": _state(second_record.prepared_owner),
        }
        calls = {"original": original, "prepared": prepared, "second": second}
        for name in order:
            _native(calls[name], tokens, tmp_path / name)
            assert _state(owner) == snapshots["original"]
            assert _state(record.prepared_owner) == snapshots["prepared"]
            assert _state(second_record.prepared_owner) == snapshots["second"]
            record.assert_current()
            second_record.assert_current()
        _softmax_trace(original, tokens, np.float32)
        _softmax_trace(prepared, tokens, np.float64)
        _softmax_trace(second, tokens, np.float64)
        for name, item in (("prepared", record), ("second", second_record)):
            (tmp_path / f"{name}-record.json").write_text(item.to_json())
        assert record.to_dict()["parameter_storage"] == "retained"


def _prepared_softmax_chains(graph: Any, values: dict[str, int]) -> None:
    """Prove the four selected FP64 chains after exact def-use membership."""
    nodes = tuple(graph.node)
    producers = {}
    for node in nodes:
        for name in node.output:
            assert (
                name and name not in producers
            ), "softmax graph has duplicate SSA producer"
            producers[name] = node
    layouts = {
        "Identity": (1,),
        "Reshape": (2,),
        "Unsqueeze": (1, 2),
        "Squeeze": (1, 2),
        "Expand": (2,),
    }

    def origin(name: str) -> tuple[str, tuple[Any, ...]]:
        path, visited = [], set()
        while name in producers:
            assert name not in visited, "cycle in softmax layout provenance"
            visited.add(name)
            node = producers[name]
            if node.domain or node.op_type not in layouts:
                break
            assert len(node.input) in layouts[node.op_type] and node.input[0]
            assert tuple(node.output) == (
                name,
            ), "layout must have one exact data output"
            path.append(node)
            name = node.input[0]
        return name, tuple(path)

    def double_path(name: str, root: str) -> None:
        actual, path = origin(name)
        assert actual == root, "selected softmax path has the wrong producer"
        assert (
            values[actual] == onnx.TensorProto.DOUBLE
        ), "selected softmax root is not DOUBLE"
        for node in path:
            assert (
                values[node.input[0]]
                == values[node.output[0]]
                == onnx.TensorProto.DOUBLE
            ), "selected softmax layout changed DOUBLE computation"

    exps = [node for node in nodes if not node.domain and node.op_type == "Exp"]
    assert len(exps) == 4, "four original softmax Exps are required"
    selected = set()
    for exp in exps:
        assert len(exp.input) == len(exp.output) == 1
        assert values[exp.input[0]] == values[exp.output[0]] == onnx.TensorProto.DOUBLE
        reductions = [
            node
            for node in nodes
            if not node.domain
            and node.op_type == "ReduceSum"
            and len(node.input) in (1, 2)
            and len(node.output) == 1
            and origin(node.input[0])[0] == exp.output[0]
        ]
        assert len(reductions) == 1, "softmax Exp needs one exact ReduceSum"
        (reduction,) = reductions
        divisions = [
            node
            for node in nodes
            if not node.domain
            and node.op_type == "Div"
            and len(node.input) == 2
            and len(node.output) == 1
            and origin(node.input[0])[0] == exp.output[0]
            and origin(node.input[1])[0] == reduction.output[0]
        ]
        assert len(divisions) == 1, "softmax needs one exact Exp / ReduceSum Div"
        (division,) = divisions
        cast_backs = [
            node
            for node in nodes
            if not node.domain
            and node.op_type == "Cast"
            and len(node.input) == len(node.output) == 1
            and origin(node.input[0])[0] == division.output[0]
            and any(
                attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT
                for attribute in node.attribute
            )
        ]
        assert len(cast_backs) == 1, "softmax Div needs one exact Cast-back to FLOAT"
        (cast_back,) = cast_backs
        # Membership is established first. Unrelated FP32 reduction/division/
        # cast candidates never become part of this selected DOUBLE proof.
        double_path(reduction.input[0], exp.output[0])
        double_path(division.input[0], exp.output[0])
        double_path(division.input[1], reduction.output[0])
        double_path(cast_back.input[0], division.output[0])
        assert (
            values[reduction.output[0]]
            == values[division.output[0]]
            == onnx.TensorProto.DOUBLE
        )
        assert [
            (attribute.name, attribute.type, attribute.i)
            for attribute in cast_back.attribute
        ] == [("to", onnx.AttributeProto.INT, onnx.TensorProto.FLOAT)]
        assert values[cast_back.output[0]] == onnx.TensorProto.FLOAT
        for node in (reduction, division, cast_back):
            assert (
                node.output[0] not in selected
            ), "softmax chains must have unique correspondence"
            selected.add(node.output[0])


def _artifact(model: Any, prepared: bool) -> None:
    definitions = {(f.domain, f.name): f for f in model.functions}
    root_calls = [
        (n.domain, n.op_type)
        for n in model.graph.node
        if (n.domain, n.op_type) in definitions
    ]
    assert root_calls and any("GPTOSS_Transformer" in name for _, name in root_calls)
    reachable, todo = set(), list(root_calls)
    while todo:
        key = todo.pop()
        if key in reachable:
            continue
        reachable.add(key)
        todo.extend(
            (n.domain, n.op_type)
            for n in definitions[key].node
            if (n.domain, n.op_type) in definitions
        )
    for name in [
        "TokenEmbedding",
        "TransformerBlock",
        "AttentionCore",
        "MLPBlockCore",
        "RMSNorm",
    ]:
        assert any(name in function for _, function in reachable), name
    # Inspect a diagnostic inline copy; the tested saved artifact retains Functions.
    flat = onnx.shape_inference.infer_shapes(
        inliner.inline_local_functions(model), strict_mode=True
    )
    values = {
        v.name: v.type.tensor_type.elem_type
        for v in (*flat.graph.input, *flat.graph.value_info, *flat.graph.output)
    }
    exps = [n for n in flat.graph.node if n.op_type == "Exp"]
    assert len(exps) == 4
    wanted = onnx.TensorProto.DOUBLE if prepared else onnx.TensorProto.FLOAT
    assert all(values[n.input[0]] == values[n.output[0]] == wanted for n in exps)
    if prepared:
        _prepared_softmax_chains(flat.graph, values)
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.INT32
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert tuple(
        d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim
    ) == (3, 48)


def _prepared_softmax_fixture(layout: str) -> tuple[Any, dict[str, int]]:
    """Typed structural graph only; no model execution or numerical criterion."""
    nodes, values = [], {}
    double, floating = onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT
    inputs = {
        **{f"x_{index}": double for index in range(4)},
        "unrelated": floating,
        "shape": onnx.TensorProto.INT64,
        "axes": onnx.TensorProto.INT64,
    }
    values.update(inputs)

    def add(
        operator: str, operands: list[str], output: str, dtype: int, **attributes: Any
    ) -> None:
        nodes.append(
            onnx.helper.make_node(
                operator, operands, [output], name=output, **attributes
            )
        )
        values[output] = dtype

    def data_layout(source: str, output: str, dtype: int) -> None:
        operands = [source]
        if layout in {"Reshape", "Expand"}:
            operands.append("shape")
        elif layout in {"Unsqueeze", "Squeeze"}:
            operands.append("axes")
        add(layout, operands, output, dtype)

    for index in range(4):
        add("Exp", [f"x_{index}"], f"exp_{index}", double)
        data_layout(f"exp_{index}", f"exp_layout_{index}", double)
        add(
            "ReduceSum",
            [f"exp_layout_{index}", "axes"],
            f"sum_{index}",
            double,
            keepdims=1,
        )
        data_layout(f"sum_{index}", f"sum_layout_{index}", double)
        add(
            "Div",
            [f"exp_layout_{index}", f"sum_layout_{index}"],
            f"division_{index}",
            double,
        )
        data_layout(f"division_{index}", f"division_layout_{index}", double)
        add(
            "Cast", [f"division_layout_{index}"], f"back_{index}", floating, to=floating
        )
    data_layout("unrelated", "unrelated_layout", floating)
    add(
        "ReduceSum", ["unrelated_layout", "axes"], "unrelated_sum", floating, keepdims=1
    )
    data_layout("unrelated_sum", "unrelated_sum_layout", floating)
    add(
        "Div",
        ["unrelated_layout", "unrelated_sum_layout"],
        "unrelated_division",
        floating,
    )
    data_layout("unrelated_division", "unrelated_division_layout", floating)
    add("Cast", ["unrelated_division_layout"], "unrelated_back", floating, to=floating)
    outputs = [*(f"back_{index}" for index in range(4)), "unrelated_back"]

    def declaration(name: str) -> Any:
        return onnx.helper.make_tensor_value_info(name, values[name], None)

    graph = onnx.helper.make_graph(
        nodes,
        "selected_softmax_with_unrelated_fp32",
        [declaration(name) for name in inputs],
        [declaration(name) for name in outputs],
        value_info=[
            declaration(name)
            for name in values
            if name not in inputs and name not in outputs
        ],
    )
    return graph, values


@pytest.mark.parametrize(
    "layout", ["Identity", "Reshape", "Unsqueeze", "Squeeze", "Expand"]
)
def test_selected_softmax_provenance_ignores_unrelated_fp32_candidates(
    layout: str,
) -> None:
    graph, values = _prepared_softmax_fixture(layout)
    before = graph.SerializeToString()
    _prepared_softmax_chains(graph, values)
    assert graph.SerializeToString() == before
    assert (
        values["unrelated_layout"]
        == values["unrelated_sum_layout"]
        == values["unrelated_division_layout"]
        == onnx.TensorProto.FLOAT
    )


@pytest.mark.parametrize(
    "defect",
    [
        "selected-float-layout",
        "selected-float-division",
        "wrong-reduction",
        "wrong-div-numerator",
        "wrong-div-denominator",
        "bypassed-cast-back",
        "wrong-cast-type",
        "wrong-producer",
        "duplicate-reduction",
        "duplicate-div",
        "duplicate-cast",
        "cycle",
        "wrong-layout-port",
        "domain-layout",
        "duplicate-ssa",
        "cast-through",
        "arithmetic-through",
        "fifth-exp",
    ],
)
def test_selected_softmax_provenance_rejects_wrong_selected_paths(defect: str) -> None:
    graph, values = _prepared_softmax_fixture("Identity")
    nodes = {node.name: node for node in graph.node}
    if defect == "selected-float-layout":
        values["exp_layout_0"] = onnx.TensorProto.FLOAT
    elif defect == "selected-float-division":
        values["division_0"] = onnx.TensorProto.FLOAT
    elif defect == "wrong-reduction":
        nodes["sum_0"].input[0] = "unrelated_layout"
    elif defect == "wrong-div-numerator":
        nodes["division_0"].input[0] = "exp_layout_1"
    elif defect == "wrong-div-denominator":
        nodes["division_0"].input[1] = "sum_layout_1"
    elif defect == "bypassed-cast-back":
        nodes["back_0"].op_type = "Identity"
        del nodes["back_0"].attribute[:]
    elif defect == "wrong-cast-type":
        nodes["back_0"].attribute[0].i = onnx.TensorProto.DOUBLE
        values["back_0"] = onnx.TensorProto.DOUBLE
    elif defect == "wrong-producer":
        nodes["sum_0"].input[0] = "x_0"
    elif defect in {
        "duplicate-reduction",
        "duplicate-div",
        "duplicate-cast",
        "duplicate-ssa",
    }:
        name = {
            "duplicate-reduction": "sum_0",
            "duplicate-div": "division_0",
            "duplicate-cast": "back_0",
            "duplicate-ssa": "exp_layout_0",
        }[defect]
        node = onnx.NodeProto()
        node.CopyFrom(nodes[name])
        if defect != "duplicate-ssa":
            node.output[0] = "duplicate"
            values["duplicate"] = values[name]
        graph.node.append(node)
    elif defect == "cycle":
        nodes["exp_layout_0"].input[0] = "exp_layout_0"
    elif defect == "wrong-layout-port":
        nodes["exp_layout_0"].input[0] = "unrelated_layout"
        nodes["exp_layout_0"].input.append("exp_0")
    elif defect == "domain-layout":
        nodes["exp_layout_0"].domain = "unrelated.domain"
    elif defect == "cast-through":
        nodes["exp_layout_0"].op_type = "Cast"
        nodes["exp_layout_0"].attribute.append(
            onnx.helper.make_attribute("to", onnx.TensorProto.DOUBLE)
        )
    elif defect == "arithmetic-through":
        nodes["exp_layout_0"].op_type = "Add"
        nodes["exp_layout_0"].input.append("exp_0")
    elif defect == "fifth-exp":
        graph.node.append(onnx.helper.make_node("Exp", ["x_0"], ["extra_exp"]))
        values["extra_exp"] = onnx.TensorProto.DOUBLE
    with pytest.raises(AssertionError):
        _prepared_softmax_chains(graph, values)


@pytest.mark.parametrize(
    "order", [("original", "prepared", "second"), ("prepared", "second", "original")]
)
@pytest.mark.parametrize(
    "opset", (23, 26), ids=("original-opset23", "explicit-opset26")
)
def test_full_original_transformer_named_artifacts_and_both_cpu_profiles(
    order: tuple[str, ...], opset: int, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        deployment_target = ort_cpu_deployment_target(
            ort.__version__, sys.implementation.cache_tag or ""
        )
        (tmp_path / "target-request.json").write_text(
            json.dumps(
                dict(
                    requested_opset=opset,
                    implicit_upgrade=False,
                    sample_contract_sha256="d2ecaed3d332dac06d99bdac8326261c097aaa65f9403df2bfb3e20c618a3eb7",
                ),
                indent=2,
            )
        )
        try:
            bitcast_schema = onnx.defs.get_schema("BitCast", 26, "")
        except onnx.defs.SchemaError:
            bitcast_available = False
        else:
            assert bitcast_schema.since_version == 26
            bitcast_available = True
        original, owner, tokens = _original(tmp_path)
        prepared, record = prepare_nnx_transformer_softmax64(original)
        second, second_record = prepare_nnx_transformer_softmax64(original)
        calls = {"original": original, "prepared": prepared, "second": second}
        owners = (owner, record.prepared_owner, second_record.prepared_owner)
        snapshots = tuple(_state(value) for value in owners)
        bind = FunctionPlugin._bind_function_call
        observed = []

        def observe(self: Any, *args: Any, **kwargs: Any) -> Any:
            before = (self._orig_fn, self._unbound_orig_fn)
            result = bind(self, *args, **kwargs)
            assert self._orig_fn is before[0] and self._unbound_orig_fn is before[1]
            if kwargs.get("owner") is not None:
                observed.append(kwargs["unbound_method"])
            return result

        for name in order:
            directory = tmp_path / name
            references = _native(calls[name], tokens, directory)

            def export_model() -> Any:
                return ui._to_onnx(
                    calls[name],
                    [jax.ShapeDtypeStruct((3,), np.int32)],
                    precision_policy=PrecisionPolicy.SOURCE,
                    opset=opset,
                    input_names=["tokens"],
                    output_names=["logits"],
                    deployment_target=deployment_target,
                )

            with patch.object(FunctionPlugin, "_bind_function_call", observe):
                if opset == 23:
                    with pytest.raises(SourcePrecisionError) as caught:
                        export_model()
                    error = caught.value
                    assert (
                        error.category == "representation-limit" and error.opset == 23
                    )
                    assert error.primitive == "top_k" and "opset=26" in error.action
                    assert "BitCast" in str(error) and "runtime" in error.action
                    (directory / "original-opset23-failure.json").write_text(
                        json.dumps(
                            dict(
                                status="REPRESENTATION_LIMIT",
                                requested_opset=23,
                                message=str(error),
                                action=error.action,
                                runtime="UNVERIFIED",
                                numerics="UNVERIFIED",
                            ),
                            indent=2,
                        )
                    )
                    assert tuple(_state(value) for value in owners) == snapshots
                    record.assert_current()
                    second_record.assert_current()
                    continue
                if not bitcast_available:
                    with pytest.raises(onnx.checker.ValidationError) as unavailable:
                        export_model()
                    assert "No Op registered for BitCast" in str(unavailable.value)
                    (directory / "explicit-opset26-unavailable.json").write_text(
                        json.dumps(
                            dict(
                                status="REPRESENTATION_UNAVAILABLE",
                                requested_opset=26,
                                message=str(unavailable.value),
                                runtime="UNVERIFIED",
                                numerics="UNVERIFIED",
                            ),
                            indent=2,
                        )
                    )
                    assert tuple(_state(value) for value in owners) == snapshots
                    record.assert_current()
                    second_record.assert_current()
                    continue
                model = export_model()
            assert next(i.version for i in model.opset_import if not i.domain) == 26
            path = directory / "full-original-model.onnx"
            path.write_bytes(model.SerializeToString())
            model = onnx.load_model(path)
            onnx.checker.check_model(model, full_check=True)
            _artifact(model, name != "original")
            report = get_precision_report(model)
            assert report is not None and report.opset == 26
            requirements = report.deployment_requirements
            assert requirements is not None and requirements.target == deployment_target
            assert requirements.matrix_precision
            assert all(
                site.source_precision == ("HIGHEST", "HIGHEST")
                for site in requirements.matrix_precision
            )
            assert {c.category: c.status for c in report.checks}[
                "runtime-execution"
            ] is CheckStatus.UNVERIFIED
            assert {c.category: c.status for c in report.checks}[
                "numerical-acceptance"
            ] is CheckStatus.UNVERIFIED
            (directory / "raw-report.json").write_text(precision_report_to_json(report))
            for label, profile in [
                ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
                ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
            ]:
                cpu_events: list[dict[str, Any]] = []
                phase = "session"
                try:
                    options = ort.SessionOptions()
                    options.graph_optimization_level = profile
                    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    optimized_path = directory / f"optimized-{label}.onnx"
                    _configure_deployment_session(options, requirements, optimized_path)
                    session = ort.InferenceSession(
                        str(path), options, providers=["CPUExecutionProvider"]
                    )
                    session.disable_fallback()
                    phase = "deployment-observation"
                    observation = _observe_deployment_session(
                        model, session, ort, optimized_path
                    )
                    (directory / f"deployment-{label}.json").write_text(
                        json.dumps(asdict(observation), indent=2)
                    )
                    require_runtime_target(requirements, observation)
                    phase = "run"
                    returned = session.run(None, {"tokens": np.asarray(tokens)})
                    cpu_events.append(dict(status="returned", stage=phase))
                    (directory / f"cpu-{label}-events.json").write_text(
                        json.dumps(cpu_events, indent=2)
                    )
                    phase = "archive"
                    np.savez(
                        directory / f"actual-{label}-returned.npz",
                        **{f"leaf_{i}": value for i, value in enumerate(returned)},
                    )
                    cpu_events.append(dict(status="archived", stage=phase))
                    (directory / f"cpu-{label}-events.json").write_text(
                        json.dumps(cpu_events, indent=2)
                    )
                    phase = "arity-and-output-validation"
                    assert type(returned) is list and len(returned) == 1
                    actual = _array(returned[0])
                    np.savez(directory / f"actual-{label}.npz", actual=actual)
                    phase = "provider-validation"
                    assert session.get_providers() == ["CPUExecutionProvider"]
                    for index, reference in enumerate(references):
                        phase = f"comparison-{index}"
                        _compare(actual, reference)
                    cpu_events.append(dict(status="passed", stage="comparisons"))
                except BaseException as exc:
                    cpu_events.append(
                        dict(
                            status="raised",
                            stage=phase,
                            exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                            message=str(exc),
                        )
                    )
                    raise
                finally:
                    (directory / f"cpu-{label}-events.json").write_text(
                        json.dumps(cpu_events, indent=2)
                    )
            assert tuple(_state(value) for value in owners) == snapshots
            record.assert_current()
            second_record.assert_current()
        if opset == 23 or not bitcast_available:
            (tmp_path / "target-result.json").write_text(
                json.dumps(
                    dict(
                        status=(
                            "REPRESENTATION_LIMIT"
                            if opset == 23
                            else "REPRESENTATION_UNAVAILABLE"
                        ),
                        requested_opset=opset,
                        artifact_runtime_numerics="UNVERIFIED",
                    ),
                    indent=2,
                )
            )
            return
        assert record.root.method in observed and second_record.root.method in observed
        (tmp_path / "sample-contract.json").write_text(
            json.dumps(
                dict(
                    status="FINITE_SAMPLE_PASS",
                    rtol=_RTOL,
                    atol=_ATOL,
                    rule="abs(actual-reference) <= atol + rtol*abs(reference)",
                    original_and_prepared_equality_required=False,
                    scientific_FP64_and_general_fixed_source="UNVERIFIED",
                ),
                indent=2,
            )
        )


@pytest.mark.parametrize(
    "role_name,port",
    [
        ("Transformer", "capture_routing"),
        ("Transformer", "capture_hidden_states"),
        ("Transformer", "capture_block_debug"),
        ("TransformerBlock", "capture_routing"),
        ("TransformerBlock", "capture_debug"),
        ("AttentionCore", "capture_debug"),
        ("MLPBlockCore", "capture_routing"),
        ("MLPBlockCore", "capture_debug"),
    ],
)
def test_original_capture_branches_require_none_without_forcing_flags(
    role_name: str, port: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        original, _, _ = _original(tmp_path)
        _, record = prepare_nnx_transformer_softmax64(original)
        role = next(
            r for r in _roles(record.root) if r.owner_type.__name__ == role_name
        )
        receivers = {
            "Transformer": record.prepared_owner,
            "TransformerBlock": record.prepared_owner.blocks["block_0"],
            "AttentionCore": record.prepared_owner.blocks["block_0"].attention,
            "MLPBlockCore": record.prepared_owner.blocks["block_0"].mlp,
        }
        owner = receivers[role_name]
        effects: list[Any] = []
        snapshot = _state(record.prepared_owner)
        for call in (role.invoke, role.method):
            with pytest.raises(
                SourcePreparationError, match="original None capture ports"
            ):
                call(owner, object(), **{port: effects})
        assert effects == [] and _state(record.prepared_owner) == snapshot
        assert role.record.to_dict()["retained_branches"] is True
        record.assert_current()


def test_prepared_owner_merge_and_changed_state_controls(tmp_path: Path) -> None:
    with jax.enable_x64(True):
        original, _, tokens = _original(tmp_path)
        prepared, record = prepare_nnx_transformer_softmax64(original)
        graphdef, state = nnx.split(record.prepared_owner)
        merged = nnx.merge(graphdef, state, copy=True)
        assert merged is not record.prepared_owner
        _compare(_array(record.root.invoke(merged, tokens)), _array(prepared(tokens)))
        with pytest.raises(SourcePreparationError, match="receiver type"):
            record.root.invoke(object(), tokens)
        target = type(record.prepared_owner)
        original_method = vars(target)["__call__"]
        with apply_monkey_patches():
            active_method = vars(target)["__call__"]
            with pytest.raises(RuntimeError, match="nested restoration control"):
                with apply_monkey_patches():
                    assert vars(target)["__call__"] is active_method
                    with patch.object(target, "__call__", lambda *a, **k: None):
                        with pytest.raises(
                            SourcePreparationError,
                            match="pinned original GPT role binding changed",
                        ):
                            record.root.invoke(record.prepared_owner, tokens)
                    assert vars(target)["__call__"] is active_method
                    raise RuntimeError("nested restoration control")
            assert vars(target)["__call__"] is active_method
        assert vars(target)["__call__"] is original_method
        record.assert_current()
        pinned = next(
            role
            for role in record.root.original_roles
            if role.state.function.__qualname__ == "RMSNorm.__call__"
        )
        with patch.object(pinned.owner, pinned.attribute, lambda *a, **k: None):
            with pytest.raises(
                SourcePreparationError, match="pinned original GPT role binding"
            ):
                record.assert_current()
        record.assert_current()
        variable = record.prepared_owner.unembedding_kernel
        value = variable.get_raw_value()
        try:
            variable.set_value(value + np.float32(1))
            with pytest.raises(SourcePreparationError, match="stored state changed"):
                record.assert_current()
        finally:
            variable.set_value(value)
        record.assert_current()
        method = record.root.method
        defaults = method.__defaults__
        try:
            method.__defaults__ = (object(),)
            with pytest.raises(
                SourcePreparationError, match="metadata or code changed"
            ):
                record.assert_current()
        finally:
            method.__defaults__ = defaults
        record.assert_current()


def test_effective_x64_is_required_without_global_configuration_changes(
    tmp_path: Path,
) -> None:
    with jax.enable_x64(False):
        original, owner, _ = _original(tmp_path)
        snapshot = _state(owner)
        with pytest.raises(SourcePreparationError, match="effective JAX x64"):
            prepare_nnx_transformer_softmax64(original)
        assert not jax.config.x64_enabled and _state(owner) == snapshot


@pytest.mark.parametrize(
    "phase", ["warmed-call", "warmed-validation", "fresh-vs-eager"]
)
def test_native_return_evidence_survives_later_failure(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = np.arange(144, dtype=np.float32).reshape(3, 48)
    calls = 0
    validations = 0
    checked_array = _array

    def operation(tokens: Any) -> Any:
        nonlocal calls
        calls += 1
        if phase == "warmed-call" and calls == 2:
            raise RuntimeError("injected native evidence control")
        return actual.copy()

    def validate(value: Any) -> np.ndarray:
        nonlocal validations
        validations += 1
        if phase == "warmed-validation" and validations == 2:
            raise RuntimeError("injected native evidence control")
        return checked_array(value)

    def reject_comparison(*values: Any) -> None:
        raise RuntimeError("injected native evidence control")

    module = sys.modules[__name__]
    monkeypatch.setattr(jax, "jit", lambda function: function)
    monkeypatch.setattr(module, "_fresh", lambda function: function)
    monkeypatch.setattr(module, "_array", validate)
    if phase == "fresh-vs-eager":
        monkeypatch.setattr(module, "_compare", reject_comparison)
    with pytest.raises(RuntimeError, match="injected native evidence control"):
        _native(operation, object(), tmp_path)
    events = json.loads((tmp_path / "native-events.json").read_text())
    assert events[0]["status"] == "returned" and events[0]["stage"] == "fresh-call"
    assert events[-1]["status"] == "raised" and events[-1]["stage"] == phase
    assert events[-1]["exception_type"] == "builtins.RuntimeError"
    with np.load(tmp_path / "native-fresh-returned.npz") as saved:
        np.testing.assert_array_equal(saved["leaf_0"], actual)
        assert saved["leaf_0"].dtype == np.float32
    with np.load(tmp_path / "native-outputs.npz") as saved:
        np.testing.assert_array_equal(saved["fresh"], actual)
    if phase == "warmed-validation":
        with np.load(tmp_path / "native-warmed-returned.npz") as saved:
            np.testing.assert_array_equal(saved["leaf_0"], actual)


@pytest.mark.parametrize(
    "qualified",
    ["Transformer.__call__", "_softmax_torch_approx", "RMSNorm.__call__", "_linear"],
)
@pytest.mark.parametrize("fault", ["foreign-function", "same-module-code"])
def test_original_live_roles_are_bound_to_pinned_source_before_preparation(
    qualified: str, fault: str, tmp_path: Path
) -> None:
    from jax2onnx.plugins.examples.nnx import gpt_oss_flax as source

    with jax.enable_x64(True):
        original, owner, _ = _original(tmp_path)
        snapshot = _state(owner)
        parts = qualified.split(".")
        binding_owner = source if len(parts) == 1 else vars(source)[parts[0]]
        attribute = parts[-1]
        retained = vars(binding_owner)[attribute]

        def foreign(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("rebound role body must never run")

        replacement = foreign
        if fault == "same-module-code":
            changed = foreign.__code__.replace(
                co_filename=retained.__code__.co_filename,
                co_firstlineno=retained.__code__.co_firstlineno,
                co_name=retained.__code__.co_name,
                co_qualname=retained.__code__.co_qualname,
            )
            replacement = FunctionType(
                changed, vars(source), retained.__name__, retained.__defaults__
            )
            replacement.__module__ = retained.__module__
            replacement.__qualname__ = retained.__qualname__
            replacement.__kwdefaults__ = retained.__kwdefaults__
            replacement.__annotations__ = retained.__annotations__
        with patch.object(binding_owner, attribute, replacement):
            with pytest.raises(
                SourcePreparationError,
                match="pinned original GPT|source/live code changed",
            ):
                prepare_nnx_transformer_softmax64(original)
        assert vars(binding_owner)[attribute] is retained and _state(owner) == snapshot
        assert hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest() == (
            "850a02ae4a8ad64d06bfb30a0d18c2e0557d5fd0b2a17fb3abfc7b72f0611922"
        )


@pytest.mark.parametrize("fault", ["positional-default", "keyword-default"])
def test_original_literal_defaults_are_bound_before_preparation(
    fault: str, tmp_path: Path
) -> None:
    from jax2onnx.plugins.examples.nnx import gpt_oss_flax as source

    with jax.enable_x64(True):
        original, owner, _ = _original(tmp_path)
        snapshot = _state(owner)
        function = (
            source._softmax_torch_approx
            if fault == "positional-default"
            else source._swiglu
        )
        attribute = (
            "__defaults__" if fault == "positional-default" else "__kwdefaults__"
        )
        changed = (
            (0,) if fault == "positional-default" else {"alpha": 1.702, "limit": 8.0}
        )
        retained = getattr(function, attribute)
        try:
            setattr(function, attribute, changed)
            with pytest.raises(
                SourcePreparationError, match="role (kw)?defaults changed"
            ):
                prepare_nnx_transformer_softmax64(original)
        finally:
            setattr(function, attribute, retained)
        assert _state(owner) == snapshot
