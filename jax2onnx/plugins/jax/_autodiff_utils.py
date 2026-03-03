# jax2onnx/plugins/jax/_autodiff_utils.py

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Callable, Iterable, cast

import jax
from jax.extend.core import Primitive
from jax.interpreters import ad, batching


logger: logging.Logger = logging.getLogger("jax2onnx.plugins.autodiff")

_AD_BACKFILL_DISABLE_ENV: str = "JAX2ONNX_DISABLE_AD_BACKFILL"
_AD_DEBUG_ENV: str = "JAX2ONNX_AD_DEBUG"

# Generic transpose via jax.linear_transpose is valid only for allowlisted
# primitives where linearity assumptions hold for conversion-time traces.
_LINEAR_TRANSPOSE_FALLBACK_ALLOWLIST: set[str] = {
    "jax.numpy.add",
    "jax.numpy.concatenate",
    "jax.numpy.moveaxis",
    "jax.numpy.reshape",
    "jax.numpy.select",
    "jax.numpy.split",
    "jax.numpy.squeeze",
    "jax.numpy.stack",
    "jax.numpy.take",
    "jax.numpy.tile",
    "jax.numpy.transpose",
    "jax.numpy.where",
}

_ORIGINAL_RULE_FORWARDING_ALLOWLIST: set[tuple[str, str]] = {
    ("add", "jax.numpy.add"),
    ("concatenate", "jax.numpy.concatenate"),
    ("reshape", "jax.numpy.reshape"),
    ("split", "jax.numpy.split"),
    ("squeeze", "jax.numpy.squeeze"),
    ("tile", "jax.numpy.tile"),
    ("transpose", "jax.numpy.moveaxis"),
    ("transpose", "jax.numpy.transpose"),
}

# Known non-1:1 mappings that are intentionally blocked from original-rule
# forwarding until converter/runtime semantics are proven compatible.
_ORIGINAL_RULE_FORWARDING_BLOCKLIST: set[tuple[str, str]] = {
    ("concatenate", "jax.numpy.stack"),
    ("gather", "jax.numpy.take"),
    ("select_n", "jax.numpy.select"),
    ("select_n", "jax.numpy.where"),
}


def _validate_forwarding_policy_sets() -> None:
    overlap = _ORIGINAL_RULE_FORWARDING_ALLOWLIST & _ORIGINAL_RULE_FORWARDING_BLOCKLIST
    if overlap:
        pairs = ", ".join(f"{orig}->{new}" for orig, new in sorted(overlap))
        raise RuntimeError(
            "Original-rule forwarding policy conflict: allowlist/blocklist overlap: "
            f"{pairs}"
        )


_validate_forwarding_policy_sets()


def _env_enabled(var_name: str) -> bool:
    value = os.environ.get(var_name, "")
    return value.strip().lower() not in {"", "0", "false", "no"}


def _debug(msg: str, *args: object) -> None:
    if _env_enabled(_AD_DEBUG_ENV):
        logger.info(msg, *args)


@dataclass(frozen=True)
class ADBackfillStats:
    scanned: int = 0
    missing_transpose: int = 0
    installed: int = 0
    skipped_no_impl: int = 0
    skipped_not_allowlisted: int = 0
    disabled: bool = False


def get_linear_transpose_fallback_allowlist() -> frozenset[str]:
    return frozenset(_LINEAR_TRANSPOSE_FALLBACK_ALLOWLIST)


def get_original_rule_forwarding_allowlist() -> frozenset[tuple[str, str]]:
    return frozenset(_ORIGINAL_RULE_FORWARDING_ALLOWLIST)


def get_original_rule_forwarding_blocklist() -> frozenset[tuple[str, str]]:
    return frozenset(_ORIGINAL_RULE_FORWARDING_BLOCKLIST)


def _should_register_transpose(prim: Primitive, requested: bool | None) -> bool:
    if requested is None:
        return prim.name in _LINEAR_TRANSPOSE_FALLBACK_ALLOWLIST
    return requested


def register_original_rule_forwarding(
    *,
    orig_prim: Primitive,
    new_prim: Primitive,
    allowlist: set[tuple[str, str]],
    override: bool = False,
    forward_batching: bool = True,
) -> None:
    """Forward AD/batching rules from ``orig_prim`` to ``new_prim``.

    Forwarding is explicit-only: pair must appear in ``allowlist``.
    """

    pair = (orig_prim.name, new_prim.name)
    if pair not in allowlist:
        raise ValueError(
            "Original-rule forwarding denied for non-allowlisted mapping: "
            f"{orig_prim.name} -> {new_prim.name}"
        )

    if orig_prim in ad.primitive_jvps and (
        override or new_prim not in ad.primitive_jvps
    ):
        ad.primitive_jvps[new_prim] = ad.primitive_jvps[orig_prim]
    if orig_prim in ad.primitive_transposes and (
        override or new_prim not in ad.primitive_transposes
    ):
        ad.primitive_transposes[new_prim] = ad.primitive_transposes[orig_prim]
    if (
        forward_batching
        and orig_prim in batching.primitive_batchers
        and (override or new_prim not in batching.primitive_batchers)
    ):
        batching.primitive_batchers[new_prim] = batching.primitive_batchers[orig_prim]


def register_allowlisted_original_rule_forwarding(
    *,
    orig_prim: Primitive,
    new_prim: Primitive,
    override: bool = False,
    forward_batching: bool = True,
) -> None:
    register_original_rule_forwarding(
        orig_prim=orig_prim,
        new_prim=new_prim,
        allowlist=_ORIGINAL_RULE_FORWARDING_ALLOWLIST,
        override=override,
        forward_batching=forward_batching,
    )


def register_transpose_via_linear_transpose(
    prim: Primitive,
    impl: Callable[..., Any],
    *,
    override: bool = False,
) -> bool:
    """Register a generic transpose rule via ``jax.linear_transpose``."""

    if prim in ad.primitive_transposes and not override:
        _debug(
            "AD transpose registration skipped (existing rule kept): %s",
            prim.name,
        )
        return False

    def _transpose_rule(ct: Any, *args: Any, **params: Any) -> tuple[Any, ...]:
        if isinstance(ct, ad.Zero):
            return tuple(
                (
                    ad.Zero(arg.aval.to_tangent_aval())
                    if isinstance(arg, ad.UndefinedPrimal)
                    else None
                )
                for arg in args
            )

        undefined_positions = [
            idx for idx, arg in enumerate(args) if isinstance(arg, ad.UndefinedPrimal)
        ]
        if not undefined_positions:
            return tuple(None for _ in args)

        primal_args = [
            ad.zeros_like_aval(arg.aval) if isinstance(arg, ad.UndefinedPrimal) else arg
            for arg in args
        ]

        def _wrapped(*undefined_args: Any) -> Any:
            all_args = list(primal_args)
            for idx, value in zip(undefined_positions, undefined_args):
                all_args[idx] = value
            return impl(*all_args, **params)

        seed_primals = tuple(primal_args[idx] for idx in undefined_positions)
        cotangents = jax.linear_transpose(_wrapped, *seed_primals)(ct)
        cotangent_values = (
            tuple(cotangents)
            if isinstance(cotangents, tuple)
            else tuple(cotangents) if isinstance(cotangents, list) else (cotangents,)
        )

        if len(cotangent_values) != len(undefined_positions):
            raise RuntimeError(
                "Transpose arity mismatch for "
                f"{prim.name}: expected {len(undefined_positions)} cotangents, "
                f"got {len(cotangent_values)}"
            )

        out: list[Any] = []
        ct_idx = 0
        for arg in args:
            if isinstance(arg, ad.UndefinedPrimal):
                out.append(cotangent_values[ct_idx])
                ct_idx += 1
            else:
                out.append(None)
        return tuple(out)

    ad.primitive_transposes[prim] = _transpose_rule
    _debug("AD transpose registration installed: %s", prim.name)
    return True


def register_fallback_jvp_rule(
    prim: Primitive,
    impl: Callable[..., Any],
    *,
    register_transpose: bool | None = None,
    transpose_override: bool = False,
) -> None:
    """Register a generic fallback JVP: evaluate primal/tangent via ``impl``."""

    def _jvp_rule(
        primals: tuple[Any, ...], tangents: tuple[Any, ...], **params: Any
    ) -> tuple[Any, Any]:
        tangent_args = tuple(ad.instantiate_zeros(t) for t in tangents)
        primal_out = impl(*primals, **params)
        tangent_out = impl(*tangent_args, **params)
        return primal_out, tangent_out

    ad.primitive_jvps[prim] = _jvp_rule
    if _should_register_transpose(prim, register_transpose):
        register_transpose_via_linear_transpose(prim, impl, override=transpose_override)


def register_jvp_via_jax_jvp(
    prim: Primitive,
    impl: Callable[..., Any],
    *,
    register_transpose: bool | None = None,
    transpose_override: bool = False,
) -> None:
    """Register a general JVP by delegating to ``jax.jvp`` on ``impl``."""

    def _jvp_rule(
        primals: tuple[Any, ...], tangents: tuple[Any, ...], **params: Any
    ) -> tuple[Any, Any]:
        tangent_args = tuple(ad.instantiate_zeros(t) for t in tangents)

        def _wrapped(*xs: Any) -> Any:
            return impl(*xs, **params)

        return cast(tuple[Any, Any], jax.jvp(_wrapped, primals, tangent_args))

    ad.primitive_jvps[prim] = _jvp_rule
    if _should_register_transpose(prim, register_transpose):
        register_transpose_via_linear_transpose(prim, impl, override=transpose_override)


def register_jvp_rule(
    prim: Primitive,
    jvp_rule: Callable[..., tuple[Any, Any]],
    *,
    impl_for_transpose: Callable[..., Any] | None = None,
    register_transpose: bool | None = None,
    transpose_override: bool = False,
) -> None:
    """Register an explicit JVP rule, optionally adding transpose fallback."""

    ad.primitive_jvps[prim] = jvp_rule
    if not _should_register_transpose(prim, register_transpose):
        return

    transpose_impl = impl_for_transpose
    if transpose_impl is None:
        transpose_impl = getattr(prim, "impl", None)
    if not callable(transpose_impl):
        raise ValueError(
            "Cannot register transpose fallback for "
            f"{prim.name}: impl_for_transpose missing and prim.impl is not callable"
        )
    register_transpose_via_linear_transpose(
        prim,
        transpose_impl,
        override=transpose_override,
    )


def backfill_missing_transpose_rules(
    primitives: Iterable[Primitive],
    *,
    allowlist: set[str] | None = None,
) -> ADBackfillStats:
    """Install transpose fallbacks for primitives that already have JVP rules."""

    if _env_enabled(_AD_BACKFILL_DISABLE_ENV):
        _debug(
            "AD backfill disabled by %s=1",
            _AD_BACKFILL_DISABLE_ENV,
        )
        return ADBackfillStats(disabled=True)

    scanned = 0
    missing = 0
    installed = 0
    skipped_no_impl = 0
    skipped_not_allowlisted = 0
    seen: set[int] = set()
    active_allowlist = allowlist or _LINEAR_TRANSPOSE_FALLBACK_ALLOWLIST

    for prim in primitives:
        key = id(prim)
        if key in seen:
            continue
        seen.add(key)
        scanned += 1
        if prim not in ad.primitive_jvps or prim in ad.primitive_transposes:
            continue
        missing += 1
        if prim.name not in active_allowlist:
            skipped_not_allowlisted += 1
            _debug(
                "AD backfill skipped (not allowlisted): %s",
                prim.name,
            )
            continue
        impl = getattr(prim, "impl", None)
        if not callable(impl):
            skipped_no_impl += 1
            _debug(
                "AD backfill skipped (no callable impl): %s",
                prim.name,
            )
            continue
        if register_transpose_via_linear_transpose(prim, impl):
            installed += 1

    stats = ADBackfillStats(
        scanned=scanned,
        missing_transpose=missing,
        installed=installed,
        skipped_no_impl=skipped_no_impl,
        skipped_not_allowlisted=skipped_not_allowlisted,
    )
    _debug(
        "AD backfill summary scanned=%d missing=%d installed=%d skipped_no_impl=%d skipped_not_allowlisted=%d",
        stats.scanned,
        stats.missing_transpose,
        stats.installed,
        stats.skipped_no_impl,
        stats.skipped_not_allowlisted,
    )
    return stats
