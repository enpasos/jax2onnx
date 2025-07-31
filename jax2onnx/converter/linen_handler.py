import inspect
import flax.nnx as nnx_var
from flax.nnx.bridge import variables as bv
from flax.nnx import graph 
import re
from contextlib import contextmanager

import jax
from flax.core import scope as flax_scope
from flax import errors as flax_errors 

_SYMBOLIC_RE = re.compile(r"(Var\(|Traced|DynamicJaxprTrace)")

@contextmanager
def _suspend_flax_shape_check():
    orig_param = flax_scope.Scope.param

    def _lenient_param(self, name, init_fn, *a, **kw):
        try:
            return orig_param(self, name, init_fn, *a, **kw)
        except flax_errors.ScopeParamShapeError as err:
            # If the mismatch mentions a traced / symbolic dim, ignore it.
            if _SYMBOLIC_RE.search(str(err)):
                existing = self.get_variable("params", name)
                # unwrap if it has .value, otherwise return the raw array
                return getattr(existing, "value", existing)
            raise

    flax_scope.Scope.param = _lenient_param
    try:
        yield
    finally:
        flax_scope.Scope.param = orig_param

def _only_variable_leaves(tree):
    """Return `tree` with all non-variable sub-trees removed."""
    if isinstance(tree, nnx_var.Variable) or isinstance(tree, graph.GraphDef):
        return tree
    if isinstance(tree, dict):
        kept = {
            key: val
            for key, val in ((k, _only_variable_leaves(v)) for k, v in tree.items())
            if val is not None
        }
        return kept or None
    # any other object → drop it
    return None


def to_onnx_for_linen_bridge(wrapper, inputs, **_):
    print("✅ Detected `nnx.bridge.ToNNX`. Applying Flax-0.11 handler.")

    linen_mod  = wrapper.module
    linen_cls  = linen_mod.__class__

    # 1️⃣  collect ONLY attrs that end up as nnx.Variable leaves
    nnx_attrs_raw = {
        k: v for k, v in vars(wrapper).items() if not k.startswith("_")
    }
    nnx_attrs = _only_variable_leaves(nnx_attrs_raw) or {}

    # 2️⃣  convert nnx → linen
    linen_vars = bv.nnx_attrs_to_linen_vars(nnx_attrs)
    params     = linen_vars.get("params", {})

    # 3️⃣  capture constructor kwargs the usual way …
    ctor_kwargs = {
        p.name: getattr(linen_mod, p.name)
        for p in inspect.signature(linen_cls.__init__).parameters.values()
        if p.name != "self" and hasattr(linen_mod, p.name)
    }

    def pure_apply(*args):
        fresh = linen_cls(**ctor_kwargs)
        with _suspend_flax_shape_check():
            return fresh.apply({"params": params}, *args)

    return pure_apply, list(inputs)
