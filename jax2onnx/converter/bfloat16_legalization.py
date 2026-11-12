# jax2onnx/converter/bfloat16_legalization.py

"""Keep bfloat16 exports inside the ONNX schemas of the target opset.

ONNX admitted bfloat16 operator by operator: ``Conv`` and the pooling operators
from opset 22, ``TopK`` from 24, ``Einsum`` from 28, and some operators not at
all. A node that reads or writes bfloat16 through a type parameter that still
excludes it is rejected by ``onnx.checker.check_model(..., full_check=True)``.

``legalize_bfloat16_operators`` runs after the optimizer. Where the excluded
type parameter is shared by the inputs and outputs of a node, it casts those
inputs to float32 and casts the results back, so the node computes in float32
and the surrounding graph keeps its bfloat16 values. Every bfloat16 value is a
float32 value, so the widening is exact; the result is rounded once at the node
boundary. Where casts cannot express the fix (generators typed by an attribute,
control-flow bodies, sequences) the export raises.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import onnx_ir as ir

from jax2onnx.ir_utils import iter_ir_functions

# Cast, which every rewrite here emits, accepts bfloat16 from opset 13.
_MIN_OPSET: Final[int] = 13
_STANDARD_DOMAINS: Final[frozenset[str]] = frozenset({"", "ai.onnx"})


class Bfloat16Rule(NamedTuple):
    """One type parameter that accepts float32 but not bfloat16.

    The rule holds for target opsets ``first_opset..last_opset``; ``last_opset``
    is None while the newest schema known to this table still excludes
    bfloat16. ``inputs`` and ``outputs`` are the positions bound to the type
    parameter; a variadic parameter covers every position from its index on.
    ``castable`` is False where casting the inputs cannot fix the node.
    """

    first_opset: int
    last_opset: int | None
    inputs: tuple[int, ...]
    variadic_inputs_from: int | None
    outputs: tuple[int, ...]
    variadic_outputs_from: int | None
    castable: bool


# Derived by scripts/emit_bfloat16_operator_rules.py from the ONNX schemas
# through this opset.
BFLOAT16_RULES_OPSET: Final[int] = 28

BFLOAT16_OPERATOR_RULES: Final[dict[str, tuple[Bfloat16Rule, ...]]] = {
    "Acos": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Acosh": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Asin": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Asinh": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Atan": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Atanh": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "AveragePool": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "BatchNormalization": (
        Bfloat16Rule(13, 13, (0, 1, 2, 3, 4), None, (0, 1, 2, 3, 4), None, True),
    ),
    "Bernoulli": (Bfloat16Rule(15, 21, (0,), None, (), None, False),),
    "Celu": (Bfloat16Rule(13, 27, (0,), None, (0,), None, True),),
    "Compress": (Bfloat16Rule(13, 27, (0,), None, (0,), None, True),),
    "ConcatFromSequence": (Bfloat16Rule(13, None, (), None, (0,), None, False),),
    "ConstantOfShape": (Bfloat16Rule(13, 19, (), None, (0,), None, False),),
    "Conv": (Bfloat16Rule(13, 21, (0, 1, 2), None, (0,), None, True),),
    "ConvTranspose": (Bfloat16Rule(13, 21, (0, 1, 2), None, (0,), None, True),),
    "Cos": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Cosh": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "CumSum": (Bfloat16Rule(13, 13, (0,), None, (0,), None, True),),
    "DeformConv": (Bfloat16Rule(19, 21, (0, 1, 2, 3, 4), None, (0,), None, True),),
    "Det": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Dropout": (Bfloat16Rule(13, 21, (1,), None, (), None, True),),
    "DynamicQuantizeLinear": (Bfloat16Rule(13, None, (0,), None, (), None, True),),
    "Einsum": (Bfloat16Rule(13, 27, (), 0, (0,), None, True),),
    "Elu": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "EyeLike": (
        Bfloat16Rule(13, 21, (), None, (0,), None, False),
        Bfloat16Rule(13, 21, (0,), None, (), None, False),
    ),
    "GRU": (Bfloat16Rule(13, 21, (0, 1, 2, 3, 5), None, (0, 1), None, True),),
    "GlobalAveragePool": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "GlobalLpPool": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "GlobalMaxPool": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "GreaterOrEqual": (Bfloat16Rule(13, 15, (0, 1), None, (), None, True),),
    "GridSample": (
        Bfloat16Rule(16, 21, (0,), None, (0,), None, True),
        Bfloat16Rule(16, 21, (1,), None, (), None, True),
    ),
    "HardSigmoid": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "HardSwish": (Bfloat16Rule(14, 21, (0,), None, (0,), None, True),),
    "If": (Bfloat16Rule(13, 15, (), None, (), 0, False),),
    "InstanceNormalization": (Bfloat16Rule(13, 21, (0, 1, 2), None, (0,), None, True),),
    "IsInf": (Bfloat16Rule(13, 19, (0,), None, (), None, True),),
    "LSTM": (Bfloat16Rule(13, 21, (0, 1, 2, 3, 5, 6, 7), None, (0, 1, 2), None, True),),
    "LeakyRelu": (Bfloat16Rule(13, 15, (0,), None, (0,), None, True),),
    "LessOrEqual": (Bfloat16Rule(13, 15, (0, 1), None, (), None, True),),
    "Loop": (Bfloat16Rule(13, 15, (), 2, (), 0, False),),
    "LpNormalization": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "LpPool": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "MaxPool": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "MaxRoiPool": (Bfloat16Rule(13, 21, (0, 1), None, (0,), None, True),),
    "MaxUnpool": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Mish": (Bfloat16Rule(18, 21, (0,), None, (0,), None, True),),
    "Multinomial": (Bfloat16Rule(13, 21, (0,), None, (), None, True),),
    "NegativeLogLikelihoodLoss": (
        Bfloat16Rule(13, 21, (0, 2), None, (0,), None, True),
    ),
    "OneHot": (
        Bfloat16Rule(13, 27, (2,), None, (0,), None, True),
        Bfloat16Rule(13, None, (0,), None, (), None, True),
        Bfloat16Rule(13, None, (1,), None, (), None, True),
    ),
    "Optional": (Bfloat16Rule(15, 27, (0,), None, (), None, False),),
    "OptionalGetElement": (
        Bfloat16Rule(15, 27, (), None, (0,), None, False),
        Bfloat16Rule(18, 27, (0,), None, (), None, False),
    ),
    "OptionalHasElement": (Bfloat16Rule(18, 27, (0,), None, (), None, True),),
    "PRelu": (Bfloat16Rule(13, 15, (0, 1), None, (0,), None, True),),
    "Pow": (Bfloat16Rule(13, 14, (1,), None, (), None, True),),
    "QuantizeLinear": (Bfloat16Rule(13, 18, (0,), None, (), None, True),),
    "RNN": (Bfloat16Rule(13, 21, (0, 1, 2, 3, 5), None, (0, 1), None, True),),
    "RandomNormal": (Bfloat16Rule(13, 21, (), None, (0,), None, False),),
    "RandomNormalLike": (
        Bfloat16Rule(13, 21, (), None, (0,), None, False),
        Bfloat16Rule(13, 21, (0,), None, (), None, False),
    ),
    "RandomUniform": (Bfloat16Rule(13, 21, (), None, (0,), None, False),),
    "RandomUniformLike": (
        Bfloat16Rule(13, 21, (), None, (0,), None, False),
        Bfloat16Rule(13, 21, (0,), None, (), None, False),
    ),
    "Range": (Bfloat16Rule(13, 26, (0, 1, 2), None, (0,), None, True),),
    "Resize": (Bfloat16Rule(13, None, (1,), None, (), None, True),),
    "ReverseSequence": (Bfloat16Rule(13, 27, (0,), None, (0,), None, True),),
    "RoiAlign": (Bfloat16Rule(13, 21, (0, 1), None, (0,), None, True),),
    "Round": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Scan": (Bfloat16Rule(13, 15, (), 0, (), 0, False),),
    "Selu": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "SequenceAt": (Bfloat16Rule(13, None, (), None, (0,), None, False),),
    "SequenceConstruct": (Bfloat16Rule(13, None, (), 0, (), None, False),),
    "SequenceInsert": (Bfloat16Rule(13, None, (1,), None, (), None, False),),
    "SequenceMap": (Bfloat16Rule(17, None, (), 1, (), None, False),),
    "Shrink": (Bfloat16Rule(13, None, (0,), None, (0,), None, True),),
    "Sin": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Sinh": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Softplus": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "Softsign": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "SplitToSequence": (Bfloat16Rule(13, 23, (0,), None, (), None, False),),
    "Tan": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "TfIdfVectorizer": (Bfloat16Rule(13, None, (), None, (0,), None, False),),
    "ThresholdedRelu": (Bfloat16Rule(13, 21, (0,), None, (0,), None, True),),
    "TopK": (Bfloat16Rule(13, 23, (0,), None, (0,), None, True),),
    "Unique": (Bfloat16Rule(13, 27, (0,), None, (0,), None, True),),
    "Where": (Bfloat16Rule(13, 15, (1, 2), None, (0,), None, True),),
}


def active_bfloat16_rules(op_type: str, opset: int) -> tuple[Bfloat16Rule, ...]:
    """Rules of ``op_type`` that hold at the target ``opset``."""
    return tuple(
        rule
        for rule in BFLOAT16_OPERATOR_RULES.get(op_type, ())
        if rule.first_opset <= opset
        and (rule.last_opset is None or opset <= rule.last_opset)
    )


def _is_bfloat16_tensor(value: ir.Value | None) -> bool:
    return (
        value is not None
        and isinstance(value.type, ir.TensorType)
        and value.dtype == ir.DataType.BFLOAT16
    )


def _positions(
    fixed: tuple[int, ...], variadic_from: int | None, count: int
) -> list[int]:
    positions: list[int] = [index for index in fixed if index < count]
    if variadic_from is not None:
        positions.extend(range(variadic_from, count))
    return positions


def _copy_shape(shape: ir.Shape | None) -> ir.Shape | None:
    return None if shape is None else ir.Shape(shape.dims)


class _Names:
    """Fresh value and node names that stay unique across the whole model."""

    def __init__(self, model: ir.Model) -> None:
        self._model: ir.Model = model
        # Collected on first use: most models need no new name at all.
        self._used: set[str] | None = None

    def _collect(self) -> set[str]:
        used: set[str] = set()
        graphs: list[ir.Graph] = [self._model.graph]
        graphs.extend(fn.graph for fn in iter_ir_functions(self._model.functions))
        for graph in graphs:
            scopes: list[ir.Graph] = [graph]
            for node in graph.all_nodes():
                scopes.extend(_attribute_graphs(node))
                used.update(value.name for value in node.outputs if value.name)
                if node.name:
                    used.add(node.name)
            for scope in scopes:
                used.update(
                    value.name
                    for value in (*scope.inputs, *scope.initializers.values())
                    if value.name
                )
        return used

    def fresh(self, base: str) -> str:
        if self._used is None:
            self._used = self._collect()
        candidate: str = base
        suffix: int = 0
        while candidate in self._used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        self._used.add(candidate)
        return candidate


def _attribute_graphs(node: ir.Node) -> list[ir.Graph]:
    graphs: list[ir.Graph] = []
    for attribute in node.attributes.values():
        if attribute.type is ir.AttributeType.GRAPH:
            graphs.append(attribute.as_graph())
        elif attribute.type is ir.AttributeType.GRAPHS:
            graphs.extend(attribute.as_graphs())
    return graphs


def _cast_node(
    source: ir.Value,
    target: ir.Value,
    dtype: ir.DataType,
    *,
    name: str,
    like: ir.Node,
) -> ir.Node:
    return ir.Node(
        "",
        "Cast",
        inputs=[source],
        attributes=[ir.AttrInt64("to", int(dtype))],
        outputs=[target],
        name=name,
        metadata_props=dict(like.metadata_props),
    )


def _unsupported(node: ir.Node, rule: Bfloat16Rule, opset: int) -> NotImplementedError:
    where: str = (
        f"ONNX {node.op_type} does not accept bfloat16 at opset {opset} "
        f"(node {node.name or '<unnamed>'})"
    )
    in_model: str = (
        "compute this step in float32 in the model and convert the result to "
        "bfloat16."
    )
    if rule.last_opset is None:
        return NotImplementedError(
            f"{where}, and no ONNX opset known to jax2onnx admits it. "
            f"{in_model.capitalize()}"
        )
    return NotImplementedError(
        f"{where}; it requires opset >= {rule.last_opset + 1}. Raise the target "
        f"opset, or {in_model}"
    )


def _legalize_node(
    node: ir.Node,
    opset: int,
    names: _Names,
    widened: dict[tuple[int, int], ir.Value],
) -> None:
    graph: ir.Graph | None = node.graph
    if graph is None:
        return
    for rule in active_bfloat16_rules(node.op_type, opset):
        input_positions: list[int] = [
            index
            for index in _positions(
                rule.inputs, rule.variadic_inputs_from, len(node.inputs)
            )
            if _is_bfloat16_tensor(node.inputs[index])
        ]
        output_positions: list[int] = _positions(
            rule.outputs, rule.variadic_outputs_from, len(node.outputs)
        )
        if not rule.castable:
            if input_positions or any(
                _is_bfloat16_tensor(node.outputs[index]) for index in output_positions
            ):
                raise _unsupported(node, rule, opset)
            continue
        if not input_positions:
            continue

        for index in input_positions:
            source: ir.Value | None = node.inputs[index]
            if source is None:
                continue
            key: tuple[int, int] = (id(graph), id(source))
            work: ir.Value | None = widened.get(key)
            if work is None:
                # One consumer-local Cast per source value and graph scope. It
                # precedes the first consumer, hence every later one as well.
                work = ir.Value(
                    name=names.fresh(f"{source.name or 'value'}_as_float32"),
                    shape=_copy_shape(source.shape),
                    type=ir.TensorType(ir.DataType.FLOAT),
                )
                graph.insert_before(
                    node,
                    _cast_node(
                        source,
                        work,
                        ir.DataType.FLOAT,
                        name=names.fresh(f"{work.name}_cast"),
                        like=node,
                    ),
                )
                widened[key] = work
            node.replace_input_with(index, work)

        for index in output_positions:
            raw: ir.Value = node.outputs[index]
            # The outputs share the type parameter of the widened inputs, so
            # the node now writes float32 here whatever the value declared.
            if raw.type is not None and not _is_bfloat16_tensor(raw):
                continue
            if not raw.uses() and not raw.is_graph_output():
                raw.type = ir.TensorType(ir.DataType.FLOAT)
                continue
            # The source value keeps its name, type, shape, metadata, consumers
            # and graph-output slot; the node itself writes a float32 work value.
            restored: ir.Value = ir.Value(
                name=raw.name,
                shape=_copy_shape(raw.shape),
                type=ir.TensorType(ir.DataType.BFLOAT16),
                doc_string=raw.doc_string,
                metadata_props=dict(raw.metadata_props),
            )
            ir.convenience.replace_all_uses_with(
                raw, restored, replace_graph_outputs=True
            )
            raw.name = names.fresh(f"{restored.name or 'value'}_in_float32")
            raw.type = ir.TensorType(ir.DataType.FLOAT)
            raw.metadata_props.clear()
            graph.insert_after(
                node,
                _cast_node(
                    raw,
                    restored,
                    ir.DataType.BFLOAT16,
                    name=names.fresh(f"{raw.name}_cast"),
                    like=node,
                ),
            )


def _legalize_graph(graph: ir.Graph, opset: int, names: _Names) -> None:
    widened: dict[tuple[int, int], ir.Value] = {}
    # ``all_nodes`` also yields the nodes of nested control-flow bodies.
    for node in list(graph.all_nodes()):
        if node.domain in _STANDARD_DOMAINS:
            _legalize_node(node, opset, names, widened)


def legalize_bfloat16_operators(model: ir.Model) -> None:
    """Rewrite or reject nodes whose schema excludes bfloat16 at the target opset.

    The top-level graph, nested control-flow bodies and ONNX Function bodies
    are covered. Graph inputs, graph outputs, Function signatures, initializers
    and value metadata keep their bfloat16 type and identity.
    """
    opset: int = model.graph.opset_imports.get("", 0)
    if opset < _MIN_OPSET:
        return
    names: _Names = _Names(model)
    _legalize_graph(model.graph, opset, names)
    for function in iter_ir_functions(model.functions):
        _legalize_graph(function.graph, function.opset_imports.get("", opset), names)
