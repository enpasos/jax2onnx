# tests/extra_tests/converter/test_bfloat16_operator_rules.py

"""The converter's bfloat16 rule table must match the installed ONNX schemas.

``converter/`` may not import ``onnx``, so the rules are a literal table. This
policy test derives them again from ``onnx.defs`` for every operator and target
opset the installed ONNX knows. After an ONNX upgrade that changes a rule,
regenerate the table with ``scripts/emit_bfloat16_operator_rules.py``.
"""

from __future__ import annotations

from jax2onnx.converter.bfloat16_legalization import (
    BFLOAT16_OPERATOR_RULES,
    BFLOAT16_RULES_OPSET,
    active_bfloat16_rules,
)
from scripts import emit_bfloat16_operator_rules as schemas


def _table_shapes(op_type: str, opset: int) -> frozenset[schemas.RuleShape]:
    return frozenset(
        (
            rule.inputs,
            rule.variadic_inputs_from,
            rule.outputs,
            rule.variadic_outputs_from,
            rule.castable,
        )
        for rule in active_bfloat16_rules(op_type, opset)
    )


def test_rule_table_matches_installed_onnx_schemas() -> None:
    history = schemas.schema_history()
    mismatches: list[str] = []
    for op_type in sorted(set(history) | set(BFLOAT16_OPERATOR_RULES)):
        versions = history.get(op_type, [])
        for opset in range(schemas.MIN_OPSET, schemas.max_opset() + 1):
            schema = schemas.effective_schema(versions, opset)
            expected = (
                schemas.rule_shapes(schema) if schema is not None else frozenset()
            )
            actual = _table_shapes(op_type, opset)
            if actual != expected:
                mismatches.append(
                    f"{op_type} at opset {opset}: table {sorted(actual)}, "
                    f"schema {sorted(expected)}"
                )
    assert not mismatches, (
        "BFLOAT16_OPERATOR_RULES differs from the installed ONNX schemas; "
        "regenerate it with scripts/emit_bfloat16_operator_rules.py:\n"
        + "\n".join(mismatches[:20])
    )


def test_rule_table_is_the_script_output() -> None:
    # The comparison above covers every installed opset. The literal encoding
    # (ranges, order) is only reproducible with the ONNX the table came from.
    if schemas.max_opset() == BFLOAT16_RULES_OPSET:
        table = {
            op_type: tuple(tuple(rule) for rule in rules)
            for op_type, rules in BFLOAT16_OPERATOR_RULES.items()
        }
        assert table == schemas.derive_rules()


def test_known_admission_opsets() -> None:
    # Spot checks that keep the table readable against the ONNX changelog.
    assert active_bfloat16_rules("Conv", 21) and not active_bfloat16_rules("Conv", 22)
    assert active_bfloat16_rules("MaxPool", 21)
    assert not active_bfloat16_rules("MaxPool", 22)
    assert active_bfloat16_rules("TopK", 23) and not active_bfloat16_rules("TopK", 24)
    assert active_bfloat16_rules("Einsum", 27)
    assert not active_bfloat16_rules("Einsum", 28)
    assert not active_bfloat16_rules("MatMul", 13)
    assert not active_bfloat16_rules("Add", 14)


def test_type_forwarding_and_generator_rules_are_not_castable() -> None:
    for op_type in ("EyeLike", "RandomNormalLike", "SplitToSequence"):
        assert all(not rule.castable for rule in BFLOAT16_OPERATOR_RULES[op_type])
    for op_type in ("RandomNormal", "RandomUniform", "ConstantOfShape", "Loop", "If"):
        assert all(not rule.castable for rule in BFLOAT16_OPERATOR_RULES[op_type])
