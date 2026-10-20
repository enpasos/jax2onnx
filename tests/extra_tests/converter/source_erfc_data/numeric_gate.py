# tests/extra_tests/converter/source_erfc_data/numeric_gate.py

"""Exact IEEE-byte comparisons; no framework or array-library imports."""

from fractions import Fraction
import math

FORMATS = {
    "float32": (4, 8, 23, Fraction(2, 1000000)),
    "float64": (8, 11, 52, Fraction(5, 100000000000000)),
}


def decode(word, dtype):
    width, exponent_bits, mantissa_bits, _ = FORMATS[dtype]
    sign = word >> (width * 8 - 1)
    exponent = (word >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = word & ((1 << mantissa_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        return ("nan" if mantissa else "-inf" if sign else "+inf"), sign, None
    bias = (1 << (exponent_bits - 1)) - 1
    significand = mantissa + ((1 << mantissa_bits) if exponent else 0)
    value = Fraction(significand) * Fraction(2) ** (
        max(exponent, 1) - bias - mantissa_bits
    )
    if sign:
        value = -value
    return (
        ("zero" if not significand else "normal" if exponent else "subnormal"),
        sign,
        value,
    )


def diagnostic_number(value):
    if value is None:
        return None
    try:
        number = float(value)
        if number == 0.0 and value != 0:
            return str(value)
        return number if math.isfinite(number) else str(value)
    except OverflowError:
        return str(value)


def compare_bytes(target, reference, dtype, *, exact=False):
    width, _, _, tolerance = FORMATS[dtype]
    assert len(target) == len(reference) and len(target) % width == 0
    elements = []
    for i in range(len(reference) // width):
        actual_word = int.from_bytes(target[i * width : (i + 1) * width], "little")
        reference_word = int.from_bytes(
            reference[i * width : (i + 1) * width], "little"
        )
        ac, asign, avalue = decode(actual_word, dtype)
        rc, rsign, rvalue = decode(reference_word, dtype)
        reasons = []
        absolute_error = None
        relative_error = None
        ulp_distance = None
        if rc == "nan":
            if ac != "nan":
                reasons.append("nan_mask")
        elif rc in {"+inf", "-inf"}:
            if ac != rc:
                reasons.append("infinity_mask_or_sign")
        elif ac in {"nan", "+inf", "-inf"}:
            reasons.append("unexpected_nonfinite")
        else:
            absolute_error = abs(avalue - rvalue)
            relative_error = absolute_error / abs(rvalue) if rvalue else None
            # Ordered IEEE words make ULP distance valid across both zero signs.
            sign_bit = 1 << (width * 8 - 1)
            mask = (1 << (width * 8)) - 1
            ordered_actual = (~actual_word & mask) if asign else actual_word | sign_bit
            ordered_reference = (
                (~reference_word & mask) if rsign else reference_word | sign_bit
            )
            ulp_distance = abs(ordered_actual - ordered_reference)
            if rc == "zero":
                if ac != "zero" or asign != rsign:
                    reasons.append("exact_signed_zero")
            elif ac == "zero":
                reasons.append("nonzero_reference_flushed_to_zero")
            elif exact:
                if actual_word != reference_word:
                    reasons.append("exact_source_rounding_boundary")
            elif rc == "subnormal":
                if ac != "subnormal" or asign != rsign:
                    reasons.append("subnormal_class_or_sign")
                if (
                    abs(
                        (actual_word & (sign_bit - 1))
                        - (reference_word & (sign_bit - 1))
                    )
                    > 8
                ):
                    reasons.append("subnormal_distance_over_8")
            elif relative_error > tolerance:
                reasons.append("relative_error_over_frozen_bound")
        elements.append(
            {
                "index": i,
                "target_word": f"{actual_word:0{width * 2}x}",
                "reference_word": f"{reference_word:0{width * 2}x}",
                "target_class": ac,
                "reference_class": rc,
                "target_signbit": asign,
                "reference_signbit": rsign,
                "absolute_error": diagnostic_number(absolute_error),
                "relative_error": diagnostic_number(relative_error),
                "ulp_distance": ulp_distance,
                "pass": not reasons,
                "reasons": reasons,
            }
        )
    failures = [item["index"] for item in elements if not item["pass"]]
    maximum_relative = max(
        (
            item["relative_error"]
            for item in elements
            if isinstance(item["relative_error"], (int, float))
        ),
        default=None,
    )
    maximum_ulp = max(
        (item["ulp_distance"] for item in elements if item["ulp_distance"] is not None),
        default=None,
    )
    return {
        "pass": not failures,
        "failing_indices": failures,
        "count": len(elements),
        "max_relative_error": maximum_relative,
        "worst_relative_indices": [
            item["index"]
            for item in elements
            if maximum_relative is not None
            and item["relative_error"] == maximum_relative
        ],
        "max_ulp_distance": maximum_ulp,
        "worst_ulp_indices": [
            item["index"]
            for item in elements
            if maximum_ulp is not None and item["ulp_distance"] == maximum_ulp
        ],
        "elements": elements,
    }
