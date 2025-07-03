# file: jax2onnx/sandbox/poc_symbolic_primitive2.py
"""
Minimal proof-of-concept: a custom concat primitive that understands
symbolic dimensions created with jax.export.symbolic_shape.
"""

from typing import Sequence
import logging

import jax
import jax.numpy as jnp
from jax import core, export
from jax.extend.core import Primitive

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("POC_SymbolicPrimitive")

# ----------------------------------------------------------------------
# 1.  Define the primitive ------------------------------------------------
# ----------------------------------------------------------------------
poc_concat_p = Primitive("poc_concat")
poc_concat_p.multiple_results = False


# ----------------------------------------------------------------------
# 2.  Fallback Python implementation (only needed for eager mode) --------
# ----------------------------------------------------------------------
def _poc_concat_impl(*arrays, axis: int):
    from jax import lax

    # use the correct keyword *dimension* (or just pass axis positionally)
    return lax.concatenate(arrays, dimension=axis)


poc_concat_p.def_impl(_poc_concat_impl)


# ----------------------------------------------------------------------
# 3.  abstract_eval – delegates shape logic to jax.export ---------------
# ----------------------------------------------------------------------
def _poc_concat_abstract_eval(*avals: core.ShapedArray, axis: int):
    """
    Compute output shape/dtype using `jax.export`.  This works as long as
    the *original* (un-patched) jnp.concatenate is used inside the helper
    `f`, so we stash that function in a global.
    """

    if "_POC_ORIG_FN" not in globals():
        raise RuntimeError("Global _POC_ORIG_FN not initialised")

    orig_concat = globals()["_POC_ORIG_FN"]

    # Build ShapeDtypeStructs from the incoming avals
    shapespecs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]

    # helper that calls the original concatenate
    def f(*args):
        return orig_concat(args, axis=axis)

    # Ask JAX to export → gives us out_avals that include symbolic dims
    exported = export.export(jax.jit(f))(*shapespecs)
    out_aval = exported.out_avals[0]  # single result

    logger.debug(f"abstract_eval result   : {out_aval.shape} {out_aval.dtype}")

    return core.ShapedArray(out_aval.shape, out_aval.dtype)


poc_concat_p.def_abstract_eval(_poc_concat_abstract_eval)


# ----------------------------------------------------------------------
# 4.  User-facing wrapper -------------------------------------------------
# ----------------------------------------------------------------------
def poc_concat_wrapper(arrays: Sequence[jax.Array], *, axis: int = 0):
    # IMPORTANT: bind *individual* tensors, not the tuple itself
    return poc_concat_p.bind(*arrays, axis=axis)


# ----------------------------------------------------------------------
# 5.  PoC driver ----------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("---- PoC with symbolic batch dimension 'B' ----")

    # a) Create the symbolic dim
    B = export.symbolic_shape("B")[0]  # _DimExpr
    logger.info(f"symbolic dim object: {B!r}")

    # b) Remember the real jnp.concatenate (needed inside abstract_eval)
    _POC_ORIG_FN = jnp.concatenate  # noqa: N806  (upper-case global)

    # c) Patch jnp.concatenate so tracing hits our primitive
    def patched_concat(arrays, *, axis: int = 0, **kw):
        # ignore other kwargs for this PoC
        return poc_concat_wrapper(arrays, axis=axis)

    import types  # for type checking in IDEs

    assert isinstance(jnp.concatenate, types.FunctionType)
    jnp.concatenate = patched_concat  # monkey-patch

    # d) Function we will trace – uses jnp.concatenate *after* patch
    def fn(a, b):
        return jnp.concatenate((a, b), axis=1)

    # e) Trace with ShapeDtypeStruct inputs that contain the symbolic dim
    a_spec = jax.ShapeDtypeStruct((B, 1, 8), jnp.float32)
    b_spec = jax.ShapeDtypeStruct((B, 10, 8), jnp.float32)

    jaxpr = jax.make_jaxpr(fn)(a_spec, b_spec)
    logger.info("Traced JAXPR with symbolic shape:")
    print(jaxpr)

    # f) Run with two concrete batch sizes – proves no retracing
    for batch in (3, 5):
        a = jnp.ones((batch, 1, 8), jnp.float32)
        b = jnp.ones((batch, 10, 8), jnp.float32) * 2
        out = fn(a, b)
        logger.info(f"batch={batch}, out.shape={out.shape}")
