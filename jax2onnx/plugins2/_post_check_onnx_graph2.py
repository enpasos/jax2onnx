# jax2onnx/plugins2/_post_check_onnx_graph2.py
from __future__ import annotations
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ONNX optional import (we also work with onnx_ir-only models)
try:
    import onnx
    from onnx import ModelProto
    _HAS_ONNX = True
except Exception:
    ModelProto = object  # type: ignore
    _HAS_ONNX = False

# ---------------- Public API ----------------

SpecItem = Union[
    str,                                         # "A[:shape] -> B[:shape] -> C[:shape]"
    Tuple[str, Dict[str, Any]],                  # ("path", { extra predicates })
]

def expect_graph2(
    specs: Sequence[SpecItem],
    *,
    symbols: Optional[Dict[str, Optional[int]]] = None,
    mode: str = "all",               # "all" (default) or "any"
    must_absent: Optional[Iterable[str]] = None,
    no_unused_inputs: bool = False,
    search_functions: bool = True,
):
    """
    Return a callable(model_or_ir) -> bool for pytest.

    Parameters
    ----------
    specs : list[str | (str, dict)]
        Each string is a path like "Op[:shape] -> Op[:shape] -> ...".
        Shapes are written with 'x' or '×' between dims, e.g. "Bx20" or "?x10".
        Optional extra predicates in the tuple form: ("path", {"attrs": {...}, "counts": {...}}).
    symbols : dict
        Symbol table for shapes, e.g. {"B": None}. Symbols unify across all shapes.
    mode : "all" | "any"
        If "all": every spec must match at least one path (default).
        If "any": at least one spec must match.
    must_absent : list[str]
        Operators that must not appear anywhere (top graph or functions).
    no_unused_inputs : bool
        If True, fail when the top graph contains dangling inputs.
    search_functions : bool
        If True (default), search all function bodies in addition to the top graph.

    Returns
    -------
    callable
        A predicate you can use in tests:  assert expect_graph2([...]) (model)
    """
    def _run(model) -> bool:
        gv = _GraphView(model, search_functions=search_functions)
        ok = True
        # must_absent
        if must_absent:
            for op in must_absent:
                if gv.count_op(op) > 0:
                    gv._fail(f"Operator '{op}' present but must be absent")
                    ok = False
        # no unused inputs (top graph only)
        if no_unused_inputs:
            unused = gv.unused_graph_inputs()
            if unused:
                gv._fail(f"Unused graph inputs: {sorted(unused)}")
                ok = False

        # path specs
        if specs:
            matches = []
            for item in specs:
                if isinstance(item, tuple):
                    path, preds = item
                else:
                    path, preds = item, {}
                m = gv.match_path_with_shapes(path, symbols=symbols or {}, **preds)
                matches.append(m)
                ok = ok and m
            if mode == "any":
                ok = any(matches)
            elif mode != "all":
                gv._fail(f"Unknown mode={mode!r}")
                ok = False
        return ok
    return _run

# ---------------- Implementation ----------------

_SHAPE_SEP = re.compile(r"\s*[x×]\s*")

class _GraphView:
    def __init__(self, model, *, search_functions: bool):
        self.model = model
        self.search_functions = search_functions
        self.errors: List[str] = []

        # Graph inventory: top + functions
        self.graphs: List[Tuple[str, Any]] = []
        self._add_graph("top", _top_graph(model))
        if search_functions:
            for name, g in _function_graphs(model):
                self._add_graph(f"fn:{name}", g)

    def _add_graph(self, name: str, g: Any):
        if g is not None:
            self.graphs.append((name, g))

    def _fail(self, msg: str):
        self.errors.append(msg)

    # -- basic queries --

    def count_op(self, op_type: str) -> int:
        c = 0
        for _, g in self.graphs:
            for n in _nodes(g):
                if n.op_type == op_type:
                    c += 1
        return c

    def unused_graph_inputs(self) -> List[str]:
        name, g = self.graphs[0]  # top only
        used = set()
        for n in _nodes(g):
            for v in _inputs_of(n):
                nm = _value_name(v)
                if nm:
                    used.add(nm)
        for v in _graph_outputs(g):
            nm = _value_name(v)
            if nm:
                used.add(nm)
        res = []
        for v in _graph_inputs(g):
            nm = _value_name(v)
            if nm and nm not in used:
                res.append(nm)
        return res

    # -- Path matcher with inline shapes --

    def match_path_with_shapes(self, path: str, *, symbols: Dict[str, Optional[int]], attrs: Dict[str, Any] = None, counts: Dict[str, int] = None) -> bool:
        attrs = attrs or {}
        counts = counts or {}
        ok_any_graph = False

        # Parse path: tokens like "Op[:shape]" split by ->
        tokens = [t.strip() for t in path.strip("^$ ").split("->")]
        steps: List[Tuple[str, Optional[Tuple]]] = []
        for tok in tokens:
            if ":" in tok:
                op, sh = tok.split(":", 1)
                steps.append((op.strip(), _parse_shape(sh)))
            else:
                steps.append((tok, None))

        for gname, g in self.graphs:
            if _match_path_on_graph(g, steps, dict(symbols)):
                # extra: counts
                ok_counts = True
                for op, want in counts.items():
                    if sum(1 for n in _nodes(g) if n.op_type == op) != want:
                        ok_counts = False
                        break
                # extra: attrs (shallow)
                ok_attrs = True
                for op, reqs in attrs.items():
                    for n in _nodes(g):
                        if n.op_type != op:
                            continue
                        if not _node_has_attrs(n, reqs):
                            ok_attrs = False
                            break
                if ok_counts and ok_attrs:
                    ok_any_graph = True
                    break
        if not ok_any_graph:
            self._fail(f"path not found: {path!r}")
        return ok_any_graph


# ---------- helpers for both ONNX and IR -----------

def _top_graph(model):
    return getattr(model, "graph", None)

def _function_graphs(model):
    # ONNX ModelProto
    if _HAS_ONNX and isinstance(model, ModelProto):
        for f in model.functions:
            # f has .node/.attribute; treat as a graph-like by returning f
            yield (f"{f.domain}:{f.name}", f)
        return
    # IR model (onnx_ir)
    funcs = getattr(model, "functions", None) or getattr(model, "_functions", None) or []
    if isinstance(funcs, dict):
        vals = funcs.values()
        for f in vals:
            yield (f"{getattr(f,'domain','')}:{getattr(f,'name','')}", getattr(f, "graph", None))
    else:
        for f in funcs:
            yield (f"{getattr(f,'domain','')}:{getattr(f,'name','')}", getattr(f, "graph", None))

def _nodes(g):
    return list(getattr(g, "nodes", getattr(g, "_nodes", getattr(g, "node", []))))

def _graph_inputs(g):
    arr = getattr(g, "inputs", getattr(g, "input", []))
    try: return list(arr)
    except Exception: return []

def _graph_outputs(g):
    arr = getattr(g, "outputs", getattr(g, "output", []))
    try: return list(arr)
    except Exception: return []

def _inputs_of(n):
    return getattr(n, "inputs", getattr(n, "input", []))

def _outputs_of(n):
    return getattr(n, "outputs", getattr(n, "output", []))

def _value_name(v) -> str:
    return getattr(v, "name", "")

def _shape_of_value(v) -> Optional[Tuple]:
    shp = getattr(v, "shape", None)
    if shp is None:
        return None
    # onnx_ir shape → tuple of ints/strings/None
    dims = []
    for d in getattr(shp, "dims", getattr(shp, "dim", shp)):
        if isinstance(d, int):
            dims.append(d)
        else:
            try:
                dims.append(int(d))
            except Exception:
                s = str(d)
                if s in ("None", "", "?", "unk", "unknown"):
                    dims.append(None)
                else:
                    dims.append(s)  # symbol token
    return tuple(dims)

_SHAPE_SEP = re.compile(r"\s*[x×]\s*")

def _parse_shape(s: str) -> Tuple:
    s = s.strip()
    s = _SHAPE_SEP.sub("x", s.replace(" ", ""))
    parts = s.split("x") if s else []
    dims: List[Optional[Union[int,str]]] = []
    for p in parts:
        if p in ("?", "None", ""):
            dims.append(None)
            continue
        try:
            dims.append(int(p))
        except Exception:
            dims.append(p)  # symbol like 'B'
    return tuple(dims)

def _unify_shape(expected: Tuple, actual: Optional[Tuple], env: Dict[str, Optional[int]]) -> bool:
    if actual is None:
        # allow unknown if expected dims are all None or symbols (we record nothing)
        return True
    if len(expected) != len(actual):
        return False
    for e, a in zip(expected, actual):
        if e is None:
            continue
        if isinstance(e, int):
            if a is not None and a != e:
                return False
        else:  # symbol
            val = env.get(e)
            if val is None:
                if isinstance(a, int):
                    env[e] = a  # bind
            else:
                if isinstance(a, int) and a != val:
                    return False
    return True

def _node_has_attrs(n, reqs: Dict[str, Any]) -> bool:
    # shallow attribute read; supports dict-like "attributes" or ONNX list
    attrs = getattr(n, "attributes", getattr(n, "attribute", None))
    # dict-like
    if isinstance(attrs, dict):
        src = attrs
    else:
        src = {}
        if isinstance(attrs, (list, tuple)):
            for a in attrs:
                nm = getattr(a, "name", getattr(a, "key", None))
                # Try common scalar carriers
                if hasattr(a, "value"):
                    val = getattr(a, "value")
                else:
                    val = getattr(a, "i", getattr(a, "f", getattr(a, "s", None)))
                if nm:
                    src[nm] = val
    for k, want in reqs.items():
        if k not in src:
            return False
        v = src[k]
        if callable(want):
            if not want(v): return False
        elif v != want:
            return False
    return True

def _match_path_on_graph(g, steps: List[Tuple[str, Optional[Tuple]]], env: Dict[str, Optional[int]]) -> bool:
    nodes = _nodes(g)
    # op index
    index: Dict[str, List[int]] = {}
    for i, n in enumerate(nodes):
        index.setdefault(n.op_type, []).append(i)

    # Try all starting candidates
    start_op = steps[0][0]
    for i0 in index.get(start_op, []):
        env_copy = dict(env)
        if _path_from(nodes, i0, steps, env_copy):
            env.update(env_copy)
            return True
    return False

def _path_from(nodes, i0: int, steps: List[Tuple[str, Optional[Tuple]]], env: Dict[str, Optional[int]]) -> bool:
    i = i0
    if nodes[i].op_type != steps[0][0]:
        return False
    # shape after first node
    sh = steps[0][1]
    if sh is not None:
        outs = _outputs_of(nodes[i])
        if not outs:
            return False
        if not any(_unify_shape(sh, _shape_of_value(v), env) for v in outs):
            return False
    # walk adjacency by “shares an output → input” (single successor acceptable)
    for s in range(1, len(steps)):
        want_op, want_shape = steps[s]
        next_idx = _unique_successor(nodes, i)
        if next_idx is None:
            return False
        if nodes[next_idx].op_type != want_op:
            return False
        if want_shape is not None:
            outs = _outputs_of(nodes[next_idx])
            if not outs or not any(_unify_shape(want_shape, _shape_of_value(v), env) for v in outs):
                return False
        i = next_idx
    return True

def _unique_successor(nodes, i: int) -> Optional[int]:
    outs = _outputs_of(nodes[i])
    succ: Optional[int] = None
    for j, n in enumerate(nodes):
        if j == i: continue
        ins = _inputs_of(n)
        if any(ov in ins for ov in outs):
            succ = j
            break
    return succ
