# tests/extra_tests/framework/test_ad_rule_completeness.py

from __future__ import annotations

import jax
from jax import dtypes
import jax.numpy as jnp
from jax.interpreters import ad
import numpy as np

from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.plugins.jax._autodiff_utils import (
    get_linear_transpose_fallback_allowlist,
)
from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    import_all_plugins,
)


_MIGRATED_JNP_LINEAR: frozenset[str] = frozenset(
    {
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
)

_MIGRATED_NN: frozenset[str] = frozenset(
    {
        "jax.nn.celu",
        "jax.nn.gelu",
        "jax.nn.leaky_relu",
        "jax.nn.mish",
        "jax.nn.relu",
        "jax.nn.selu",
        "jax.nn.sigmoid",
        "jax.nn.silu",
        "jax.nn.soft_sign",
        "jax.nn.softplus",
    }
)


def _iter_leaf_primitives():
    seen: set[int] = set()
    for plugin in PLUGIN_REGISTRY.values():
        if not isinstance(plugin, PrimitiveLeafPlugin):
            continue
        prim = getattr(plugin.__class__, "_PRIM", None)
        if prim is None:
            continue
        key = id(prim)
        if key in seen:
            continue
        seen.add(key)
        yield prim


def _leaf_primitives_by_name() -> dict[str, object]:
    return {prim.name: prim for prim in _iter_leaf_primitives()}


def test_allowlisted_leaf_primitives_with_jvp_have_transpose_rules():
    allowlist = get_linear_transpose_fallback_allowlist()
    import_all_plugins()
    with _activate_plugin_worlds():
        missing = sorted(
            prim.name
            for prim in _iter_leaf_primitives()
            if prim.name in allowlist
            and prim in ad.primitive_jvps
            and prim not in ad.primitive_transposes
        )
    assert not missing, (
        "Allowlisted leaf primitives with JVP but no transpose rule found:\n"
        + "\n".join(f"- {name}" for name in missing)
    )


def test_non_jax_families_do_not_expose_unpaired_jvp_rules():
    non_jax_prefixes = ("linen.", "nnx.", "eqx.", "dm_pix.")
    import_all_plugins()
    with _activate_plugin_worlds():
        missing = sorted(
            prim.name
            for prim in _iter_leaf_primitives()
            if prim.name.startswith(non_jax_prefixes)
            and prim in ad.primitive_jvps
            and prim not in ad.primitive_transposes
        )
    assert (
        not missing
    ), "Non-jax leaf primitives with JVP but no transpose rule found:\n" + "\n".join(
        f"- {name}" for name in missing
    )


def test_migrated_jnp_linear_primitives_have_jvp_and_transpose() -> None:
    import_all_plugins()
    with _activate_plugin_worlds():
        prims = _leaf_primitives_by_name()
        missing = [
            name
            for name in sorted(_MIGRATED_JNP_LINEAR)
            if name not in prims
            or prims[name] not in ad.primitive_jvps
            or prims[name] not in ad.primitive_transposes
        ]
    assert not missing, (
        "Expected migrated jax.numpy linear primitives to have JVP+transpose:\n"
        + "\n".join(f"- {name}" for name in missing)
    )


def test_migrated_nn_primitives_have_jvp() -> None:
    import_all_plugins()
    with _activate_plugin_worlds():
        prims = _leaf_primitives_by_name()
        missing = [
            name
            for name in sorted(_MIGRATED_NN)
            if name not in prims or prims[name] not in ad.primitive_jvps
        ]
    assert (
        not missing
    ), "Expected migrated jax.nn primitives to have JVP rules:\n" + "\n".join(
        f"- {name}" for name in missing
    )


def test_migrated_nn_primitives_are_not_allowlisted_for_linear_transpose() -> None:
    allowlist = get_linear_transpose_fallback_allowlist()
    overlap = sorted(_MIGRATED_NN.intersection(allowlist))
    assert not overlap, (
        "Nonlinear jax.nn primitives unexpectedly allowlisted for linear-transpose "
        "fallback:\n" + "\n".join(f"- {name}" for name in overlap)
    )


def _float0_const_dtypes(closed_jaxpr: jax.core.ClosedJaxpr) -> list[np.dtype]:
    found: list[np.dtype] = []
    for const in closed_jaxpr.consts:
        try:
            dtype = np.asarray(const).dtype
        except Exception:
            continue
        if dtype == dtypes.float0:
            found.append(dtype)
    return found


def test_where_select_take_grad_jaxprs_have_no_float0_consts() -> None:
    import_all_plugins()
    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    idx = jnp.asarray([0, 2], dtype=jnp.int32)
    cond_a = jnp.asarray(
        [[True, False, False], [False, True, False]],
        dtype=jnp.bool_,
    )
    cond_b = jnp.asarray(
        [[False, True, False], [False, False, True]],
        dtype=jnp.bool_,
    )

    with _activate_plugin_worlds():
        where_closed = jax.make_jaxpr(
            lambda z: jax.grad(lambda y: jnp.sum(jnp.where(y > 0, y, 0.0)))(z)
        )(x)
        select_closed = jax.make_jaxpr(
            lambda z: jax.grad(
                lambda y: jnp.sum(
                    jnp.select([cond_a, cond_b], [y, 2.0 * y], default=0.0)
                )
            )(z)
        )(x)
        take_closed = jax.make_jaxpr(
            lambda z: jax.grad(lambda y: jnp.sum(jnp.take(y, idx, axis=1)))(z)
        )(x)

    where_float0 = _float0_const_dtypes(where_closed)
    select_float0 = _float0_const_dtypes(select_closed)
    take_float0 = _float0_const_dtypes(take_closed)

    assert not where_float0, f"where grad jaxpr contained float0 consts: {where_float0}"
    assert (
        not select_float0
    ), f"select grad jaxpr contained float0 consts: {select_float0}"
    assert not take_float0, f"take grad jaxpr contained float0 consts: {take_float0}"
