# jax2onnx/converter/linen_handler.py
from __future__ import annotations

import inspect
import re
from contextlib import contextmanager

import flax.nnx as nnx_var
from flax.nnx import graph
from flax.nnx.bridge import variables as bv
import jax
from flax import errors as flax_errors
from flax.core import scope as flax_scope

# -------------------------------------------------------------------- #
# 🩹  Register RNG-related variable types so nnx_attrs_to_linen_vars
#     can convert them into standard Linen collections.
# -------------------------------------------------------------------- #
try:
    from flax.nnx.rnglib import RngKey, RngCount
    from flax.nnx import variablelib

    _REGISTRY = (("rngs", RngKey), ("counters", RngCount))
    for name, typ in _REGISTRY:
        if name not in variablelib.VariableTypeCache:
            variablelib.register_variable_name(name, typ)
except Exception:
    # Older Flax versions might miss rnglib or use another API – fine.
    pass


# -------------------------------------------------------------------- #
# Helper: temporarily turn Flax’ shape check lenient for symbolic dims
# -------------------------------------------------------------------- #
_SYMBOLIC_RE = re.compile(r"(Var\(|Traced|DynamicJaxprTrace)")


@contextmanager
def _suspend_flax_shape_check():
    """Ignore ScopeParamShapeError caused by symbolic dims such as “B”."""
    orig_param = flax_scope.Scope.param

    def _lenient_param(self, name, init_fn, *a, **kw):
        try:
            return orig_param(self, name, init_fn, *a, **kw)
        except flax_errors.ScopeParamShapeError as err:
            if _SYMBOLIC_RE.search(str(err)):
                existing = self.get_variable("params", name)
                return getattr(existing, "value", existing)
            raise

    flax_scope.Scope.param = _lenient_param
    try:
        yield
    finally:
        flax_scope.Scope.param = orig_param


# -------------------------------------------------------------------- #
# Helper: keep **only** nnx.Variables & GraphDefs – strip everything else
# -------------------------------------------------------------------- #
def _only_variable_leaves(tree):
    """
    Produce a copy of *tree* that retains
      • `flax.nnx.Variable` leaves (Params, States, Counters, …)
      • `flax.nnx.graph.GraphDef` objects (to preserve sub-graphs)
    and drops everything else.
    """
    if isinstance(tree, (nnx_var.Variable, graph.GraphDef)):
        return tree

    if isinstance(tree, dict):
        kept = {k: v_ for k, v_ in ((k, _only_variable_leaves(v)) for k, v in tree.items()) if v_ is not None}
        return kept or None

    if isinstance(tree, (list, tuple)):
        seq = [_only_variable_leaves(x) for x in tree]
        seq = [x for x in seq if x is not None]
        if not seq:
            return None
        return seq if isinstance(tree, list) else tuple(seq)

    if hasattr(tree, "__dict__"):
        return _only_variable_leaves(vars(tree))

    return None


# -------------------------------------------------------------------- #
#  Public hook – called by jax2onnx for every nnx.bridge.ToNNX wrapper
# -------------------------------------------------------------------- # 
def to_onnx_for_linen_bridge(wrapper, inputs, **_):
    """
    Turn a `flax.nnx.bridge.ToNNX` wrapper into a *pure* JAX function so
    that jax2onnx can trace it.
    """
    print("✅ Detected `nnx.bridge.ToNNX`. Applying Flax-0.11 handler.")

    linen_mod = wrapper.module

    # 1️⃣  Extract the nnx.Variable leaves from the wrapper -------------------
    raw_attrs = {k: v for k, v in vars(wrapper).items() if not k.startswith("_")}
    nnx_attrs = _only_variable_leaves(raw_attrs) or {}

    # 2️⃣  Convert them into standard Linen variable collections --------------
    linen_vars = bv.nnx_attrs_to_linen_vars(nnx_attrs)

    # 3️⃣  Bind those variables to the *original* module ----------------------
    bound_mod = linen_mod.bind(linen_vars)

    # 4️⃣  Wrap it into a side-effect-free function for jax2onnx --------------
    def pure_apply(*args):
        # This is where the tracing for Linen modules happens. The patch MUST be here.
        with _suspend_flax_shape_check():
            return bound_mod(*args)

    # jax2onnx expects (callable, list_of_positional_inputs)
    return pure_apply, list(inputs)