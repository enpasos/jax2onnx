# jax2onnx/converter/normalization_options.py

"""Shared normalization representation requests, independent of precision policy."""

from typing import Final, Literal, cast

NormalizationMode = Literal["auto", "prefer_native", "force_decomposed"]
_VALID_NORMALIZATION_MODES: Final[frozenset[str]] = frozenset(
    {"auto", "prefer_native", "force_decomposed"}
)


def normalize_normalization_mode(value: str) -> NormalizationMode:
    mode = value.lower().strip()
    if mode not in _VALID_NORMALIZATION_MODES:
        raise ValueError(
            "Unsupported normalization_mode "
            f"'{value}'. Expected one of: {sorted(_VALID_NORMALIZATION_MODES)}"
        )
    return cast(NormalizationMode, mode)


def source_normalization_adaptations(
    mode: NormalizationMode, *, diagnostic: bool
) -> tuple[str, ...]:
    """Record a nondefault representation decision without claiming fusion parity.

    Audited SOURCE tracing retains the original typed native JAX DAG. A preferred
    native ONNX normalization needs a source-equivalence proof covering constants,
    casts, accumulation and rounding. Diagnostic exports bypass that audit and
    cannot establish the effective representation or source equivalence.
    """
    if mode == "auto":
        return ()
    if diagnostic:
        return (
            f"normalization: requested={mode}; effective=UNVERIFIED; "
            "source-equivalence=UNVERIFIED; reason=diagnostic export has no lowering audit",
        )
    reason = (
        "native ONNX normalization has no reviewed source-equivalence proof"
        if mode == "prefer_native"
        else "explicit representation requested"
    )
    return (
        f"normalization: requested={mode}; effective=typed-native-jax-dag; "
        f"reason={reason}; original constants, casts, accumulation and rounding retained",
    )
