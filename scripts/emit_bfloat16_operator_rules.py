#!/usr/bin/env python3
# scripts/emit_bfloat16_operator_rules.py

"""Derive the bfloat16 operator rules of the converter from the ONNX schemas.

The converter must not import ``onnx``, so
``jax2onnx/converter/bfloat16_legalization.py`` carries the rules as a literal
table. Run this script after an ONNX upgrade and paste its output over that
table; ``tests/extra_tests/converter/test_bfloat16_operator_rules.py`` compares
the table with the installed schemas.
"""

from __future__ import annotations

import collections
import sys
from typing import Final, TypeAlias

from onnx import defs

BFLOAT16: Final[str] = "tensor(bfloat16)"
FLOAT: Final[str] = "tensor(float)"

# Cast, the operator every rule relies on, accepts bfloat16 from opset 13.
MIN_OPSET: Final[int] = 13

# An output of these operators takes its type from an input without sharing a
# type parameter with it (for example ``EyeLike`` without ``dtype``, or the
# sequence built by ``SplitToSequence``). Casting that input would silently
# change the type of the output, so these rules cannot be legalized by casts.
TYPE_FORWARDING_OPS: Final[frozenset[str]] = frozenset(
    {
        "Bernoulli",
        "EyeLike",
        "Optional",
        "OptionalGetElement",
        "RandomNormalLike",
        "RandomUniformLike",
        "SequenceConstruct",
        "SequenceInsert",
        "SequenceMap",
        "SplitToSequence",
    }
)

# (first_opset, last_opset, inputs, variadic_inputs_from, outputs,
#  variadic_outputs_from, castable); ``last_opset`` is None while the newest
# installed schema still excludes bfloat16.
Rule: TypeAlias = tuple[
    int, int | None, tuple[int, ...], int | None, tuple[int, ...], int | None, bool
]
# The same rule without its opset range.
RuleShape: TypeAlias = tuple[
    tuple[int, ...], int | None, tuple[int, ...], int | None, bool
]


def max_opset() -> int:
    return int(defs.onnx_opset_version())


def schema_history() -> dict[str, list[defs.OpSchema]]:
    history: dict[str, list[defs.OpSchema]] = collections.defaultdict(list)
    for schema in defs.get_all_schemas_with_history():
        if schema.domain == "":
            history[schema.name].append(schema)
    for versions in history.values():
        versions.sort(key=lambda schema: schema.since_version)
    return dict(history)


def effective_schema(versions: list[defs.OpSchema], opset: int) -> defs.OpSchema | None:
    known = [schema for schema in versions if schema.since_version <= opset]
    if not known or known[-1].deprecated:
        return None
    return known[-1]


def rule_shapes(schema: defs.OpSchema) -> frozenset[RuleShape]:
    """Type parameters of ``schema`` that accept float32 but not bfloat16."""
    variadic = defs.OpSchema.FormalParameterOption.Variadic
    shapes: set[RuleShape] = set()
    for constraint in schema.type_constraints:
        allowed = set(constraint.allowed_type_strs)
        if FLOAT not in allowed or BFLOAT16 in allowed:
            continue
        inputs: list[int] = []
        outputs: list[int] = []
        variadic_inputs_from: int | None = None
        variadic_outputs_from: int | None = None
        homogeneous = True
        for index, parameter in enumerate(schema.inputs):
            if parameter.type_str != constraint.type_param_str:
                continue
            if parameter.option == variadic:
                variadic_inputs_from = index
                homogeneous = homogeneous and parameter.is_homogeneous
            else:
                inputs.append(index)
        for index, parameter in enumerate(schema.outputs):
            if parameter.type_str != constraint.type_param_str:
                continue
            if parameter.option == variadic:
                variadic_outputs_from = index
                homogeneous = homogeneous and parameter.is_homogeneous
            else:
                outputs.append(index)
        castable = (
            homogeneous
            and (bool(inputs) or variadic_inputs_from is not None)
            and schema.name not in TYPE_FORWARDING_OPS
        )
        shapes.add(
            (
                tuple(inputs),
                variadic_inputs_from,
                tuple(outputs),
                variadic_outputs_from,
                castable,
            )
        )
    return frozenset(shapes)


def derive_rules() -> dict[str, tuple[Rule, ...]]:
    newest = max_opset()
    table: dict[str, tuple[Rule, ...]] = {}
    for name, versions in sorted(schema_history().items()):
        open_ranges: dict[RuleShape, list[int]] = {}
        rules: list[Rule] = []

        def close(shape: RuleShape, first: int, last: int) -> None:
            rules.append((first, None if last == newest else last, *shape))

        for opset in range(MIN_OPSET, newest + 1):
            schema = effective_schema(versions, opset)
            shapes = rule_shapes(schema) if schema is not None else frozenset()
            for shape in sorted(set(open_ranges) - shapes):
                first, last = open_ranges.pop(shape)
                close(shape, first, last)
            for shape in shapes:
                if shape in open_ranges:
                    open_ranges[shape][1] = opset
                else:
                    open_ranges[shape] = [opset, opset]
        for shape, (first, last) in open_ranges.items():
            close(shape, first, last)
        if rules:
            table[name] = tuple(sorted(rules, key=repr))
    return table


def render(table: dict[str, tuple[Rule, ...]]) -> str:
    lines = [
        "# Derived by scripts/emit_bfloat16_operator_rules.py from the ONNX schemas",
        "# through this opset.",
        f"BFLOAT16_RULES_OPSET: Final[int] = {max_opset()}",
        "",
        "BFLOAT16_OPERATOR_RULES: Final[dict[str, tuple[Bfloat16Rule, ...]]] = {",
    ]
    for name, rules in table.items():
        calls = [
            f"Bfloat16Rule({', '.join(repr(field) for field in rule)})"
            for rule in rules
        ]
        one_line = f'    "{name}": ({calls[0]},),'
        if len(calls) == 1 and len(one_line) <= 88:
            lines.append(one_line)
            continue
        lines.append(f'    "{name}": (')
        lines.extend(f"        {call}," for call in calls)
        lines.append("    ),")
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    print(render(derive_rules()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
