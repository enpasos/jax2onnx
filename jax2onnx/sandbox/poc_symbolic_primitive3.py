# file: jax2onnx/sandbox/poc_symbolic_primitive3.py
"""
PoC: custom concatenate primitive whose *abstract_eval* delegates
shape-inference to **jax.eval_shape** while the outer jax.make_jaxpr
trace is still running – including a symbolic batch dimension "B".
"""

from typing import Sequence
import logging
import types

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
# 1.  Primitive definition ---------------------------------------------
# ----------------------------------------------------------------------
poc_concat_p = Primitive("poc_concat")
poc_concat_p.multiple_results = False


# ----------------------------------------------------------------------
# 2.  Python fallback (eager) ------------------------------------------
# ----------------------------------------------------------------------
def _poc_concat_impl(*arrays, axis: int):
    from jax import lax

    return lax.concatenate(arrays, dimension=axis)


poc_concat_p.def_impl(_poc_concat_impl)


# ----------------------------------------------------------------------
# 3.  abstract_eval – **via jax.eval_shape** ---------------------------
# ----------------------------------------------------------------------
def _poc_concat_abstract_eval(*avals: core.ShapedArray, axis: int):
    """
    • Runs *inside* the outer trace started by jax.make_jaxpr.
    • Converts incoming avals → ShapeDtypeStruct.
    • Uses jax.eval_shape on the *original* jnp.concatenate.
    """

    if "_POC_ORIG_FN" not in globals():
        raise RuntimeError("Global _POC_ORIG_FN not initialised")

    orig_concat = globals()["_POC_ORIG_FN"]

    # 1) Convert the incoming abstract values
    specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]

    # 2) Helper that calls the UN-patched concatenate
    def f(*xs):
        return orig_concat(xs, axis=axis)

    # 3) Let JAX work out the result shape/dtype
    out_aval = jax.eval_shape(f, *specs)  # <- key change
    out_aval = jax.tree_util.tree_leaves(out_aval)[0]  # single result

    logger.debug(f"abstract_eval result: {out_aval.shape}  {out_aval.dtype}")
    return core.ShapedArray(out_aval.shape, out_aval.dtype)


poc_concat_p.def_abstract_eval(_poc_concat_abstract_eval)


# ----------------------------------------------------------------------
# 4.  User-facing wrapper ----------------------------------------------
# ----------------------------------------------------------------------
def poc_concat_wrapper(arrays: Sequence[jax.Array], *, axis: int = 0):
    # Bind *individual* tensors, not the tuple itself
    return poc_concat_p.bind(*arrays, axis=axis)


# ----------------------------------------------------------------------
# 5.  PoC driver --------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("---- PoC with symbolic batch dimension 'B' ----")

    # a) Create symbolic dim
    B = export.symbolic_shape("B")[0]
    logger.info(f"symbolic dim object: {B!r}")

    # b) Remember real jnp.concatenate
    _POC_ORIG_FN = jnp.concatenate  # noqa: N806

    # c) Monkey-patch jnp.concatenate so tracing hits our primitive
    def patched_concat(arrays, *, axis: int = 0, **kw):
        return poc_concat_wrapper(arrays, axis=axis)

    assert isinstance(jnp.concatenate, types.FunctionType)
    jnp.concatenate = patched_concat

    # d) Function that uses the patched op
    def fn(a, b):
        return jnp.concatenate((a, b), axis=1)

    # e) Trace with symbolic shapes      (== “inside trace” part)
    a_spec = jax.ShapeDtypeStruct((B, 1, 8), jnp.float32)
    b_spec = jax.ShapeDtypeStruct((B, 10, 8), jnp.float32)

    jaxpr = jax.make_jaxpr(fn)(a_spec, b_spec)
    logger.info("Traced JAXPR with symbolic shape:")
    print(jaxpr)

    # f) Run with different batch sizes – no retracing
    for batch in (3, 5):
        a = jnp.ones((batch, 1, 8), jnp.float32)
        b = jnp.ones((batch, 10, 8), jnp.float32) * 2
        out = fn(a, b)
        logger.info(f"batch={batch}, out.shape={out.shape}")
