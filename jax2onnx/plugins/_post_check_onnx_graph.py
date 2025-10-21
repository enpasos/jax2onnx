# jax2onnx/plugins/_post_check_onnx_graph.py

from __future__ import annotations
import json
import re
from typing import (
    Any,
    Dict,
    Final,
    Iterable,
    List,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Set,
    Tuple,
    Union,
)
from collections import deque
import numpy as np

# Optional ONNX helper (disabled; kept for API compatibility without importing onnx)
_onnx_numpy_helper: Final[Any | None] = None

ShapeDim = Optional[Union[int, str]]

# No onnx imports here by policy — we work via duck-typing only.

SpecItem = Union[
    str,  # "A[:shape] -> B[:shape] -> C[:shape]"
    Tuple[str, Dict[str, Any]],  # ("path", { extra predicates })
    Mapping[str, Any],  # {"graph": ..., "path": ..., ...}
]


DEFAULT_PASSTHROUGH_OPS: Final[Set[str]] = {
    "Reshape",
    "Identity",
    "Cast",
    "CastLike",
    "Squeeze",
    "Unsqueeze",
    "Flatten",
    "Shape",
    "Gather",
    "Concat",
    "Where",
    "Add",
}


def expect_graph(
    specs: Sequence[SpecItem],
    *,
    symbols: Optional[Mapping[str, ShapeDim]] = None,
    mode: str = "all",  # "all" (default) or "any"
    must_absent: Optional[Iterable[str]] = None,
    no_unused_inputs: bool = False,
    no_unused_function_inputs: bool = False,
    search_functions: bool = False,  # default: check TOP graph only
    explain_on_fail: bool = True,
):
    """Return a callable(model_or_ir) -> bool for pytest.

    **Basic paths**
        ``specs`` is a sequence describing graph fragments.  Each entry can be:

        * a plain string – e.g. ``"Gemm -> Relu -> Gemm"``
        * a mapping with a ``"path"`` key and optional predicates
        * a ``(path, options)`` tuple

        Nodes are written in evaluation order using ``->``.  Append
        ``:shape`` to assert the first output tensor's shape, e.g.
        ``"Dropout:Bx64"``.  Shapes use ``x`` separators and can mix
        integers with symbolic labels bound via ``symbols``.

    **Entry options**
        When using the mapping/tuple forms a number of predicates are
        available:

        ``inputs``
            Dict mapping 0-based input indices (integer or numeric string)
            to constraints.  Supported predicates:

            - ``{"const": value}`` – require a scalar initializer equal to
              ``value``.  Shape-only wrappers (``Expand``, ``Reshape``, etc.)
              are folded automatically.
            - ``{"const_bool": bool}`` – convenience wrapper for boolean
              scalars.
            - ``{"initializer_name": "ratio"}`` – expect a specific
              initializer name.
            - ``{"absent": True}`` – assert the input slot is missing/empty.

        ``must_absent``
            List of operator names that must not appear anywhere in the
            searched graph(s).

        ``counts``
            Dict of ``op_type -> exact count`` to validate the total
            occurrences of particular ops.

        ``symbols``
            Extra symbolic dimension bindings local to the entry.

        ``graph``
            Limit the search to specific subgraph/function names when
            ``search_functions`` is enabled.

    **Global options**
        ``mode``
            ``"all"`` (default) requires every spec to match at least once;
            ``"any"`` succeeds if one spec matches.

        ``must_absent``
            A global blacklist of operator names.

        ``no_unused_inputs`` / ``no_unused_function_inputs``
            Assert that the top-level graph (and optionally function bodies)
            have no dangling inputs.

        ``search_functions``
            If true, run the matcher on every imported function body in
            addition to the main graph.

    Passthrough operators (Reshape, Cast, Squeeze, etc.) are skipped when
    traversing forward edges so that small shape helper chains do not need to
    be spelled out explicitly.  Constant extraction for ``inputs`` predicates
    uses the same passthrough set, allowing patterns to reference the scalar
    even if the graph materialises it via ``Expand``.

    Parameters
    ----------
    specs : list[str | (str, dict)]
        Each string is a path like "Op[:shape] -> Op[:shape] -> ...".
        Shapes use 'x' or '×' between dims (e.g., "Bx20", "?x10").
        Optional tuple form adds predicates: ("path", {"attrs": {...}, "counts": {...}}).
    symbols : dict
        Declare symbols (e.g., {"B": None}); they unify across all shapes in the path.
    mode : "all" | "any"
        "all": every spec must match at least once (default). "any": at least one spec matches.
    must_absent : list[str]
        Operators that must not appear anywhere.
    no_unused_inputs : bool
        If True, fail when the top graph contains dangling inputs.
    search_functions : bool
        If True, search all function bodies in addition to the top graph.
    explain_on_fail : bool
        If True, print a short diagnostic report on failure.

    Returns
    -------
    callable
        A predicate you can use in tests:  assert expect_graph([...])(model)
    """

    def _parse_spec(item: SpecItem) -> Dict[str, Any]:
        if isinstance(item, Mapping):
            if "path" not in item:
                raise ValueError("expect_graph spec dict requires a 'path' entry")
            return dict(item)
        if isinstance(item, tuple):
            path, preds = item
            spec = {"path": path}
            if isinstance(preds, Mapping):
                spec.update(preds)
            else:
                raise TypeError(
                    "expect_graph tuple spec must provide a mapping as the second item"
                )
            return spec
        # plain string path
        return {"path": item}

    def _run(model) -> bool:
        gv = _GraphView(
            model,
            search_functions=search_functions,
            passthrough_ops=DEFAULT_PASSTHROUGH_OPS,
        )
        ok = True
        # must_absent
        if must_absent:
            for op in must_absent:
                if gv.count_op(op) > 0:
                    offs = gv.list_ops(op)
                    where = ", ".join(f"{name}#{idx}" for name, idx in offs[:5])
                    gv._fail(
                        f"Operator '{op}' present but must be absent "
                        f"(first occurrences: {where}{'…' if len(offs)>5 else ''})"
                    )
                    ok = False
        # no unused inputs (top graph only)
        if no_unused_inputs:
            unused = gv.unused_graph_inputs()
            if unused:
                gv._fail(f"Unused graph inputs: {sorted(unused)}")
                ok = False
        if no_unused_function_inputs:
            if not search_functions:
                gv._fail(
                    "no_unused_function_inputs=True requires search_functions=True"
                )
                ok = False
            else:
                unused_by_fn = gv.unused_function_inputs()
                if unused_by_fn:
                    details = ", ".join(
                        f"{name}: {sorted(vals)}"
                        for name, vals in sorted(unused_by_fn.items())
                    )
                    gv._fail(f"Unused function inputs: {details}")
                    ok = False

        # path specs
        if specs:
            matches = []
            for item in specs:
                spec = _parse_spec(item)
                path = spec.get("path")
                if not isinstance(path, str):
                    raise TypeError("expect_graph path spec must be a string")
                symbol_env: Dict[str, ShapeDim] = dict(symbols or {})
                spec_symbols = spec.get("symbols")
                if spec_symbols:
                    symbol_env.update(spec_symbols)
                ok_match, reason, matched_nodes = gv.match_path_with_shapes(
                    path,
                    symbols=symbol_env,
                    attrs=spec.get("attrs"),
                    counts=spec.get("counts"),
                    graph_filter=spec.get("graph"),
                    must_absent=spec.get("must_absent"),
                    input_constraints=spec.get("inputs"),
                )
                matches.append(ok_match)
                if not ok_match:
                    if reason:
                        gv._fail(reason)
                    ok = False
        if mode == "any":
            ok = any(matches)
        elif mode != "all":
            gv._fail(f"Unknown mode={mode!r}")
            ok = False

        if not ok and explain_on_fail:
            print("\n[expect_graph] FAILED")
            for line in gv.errors:
                print("  -", line)
        return ok

    return _run


# ---------------- Implementation ----------------

_SHAPE_SEP: Final[Pattern[str]] = re.compile(r"\s*[x×]\s*")
_NUMERIC_SUFFIX: Final[Pattern[str]] = re.compile(r"_[0-9]+$")


def _strip_numeric_suffix(op: str) -> str:
    return _NUMERIC_SUFFIX.sub("", op)


def _op_matches(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    return _strip_numeric_suffix(expected) == _strip_numeric_suffix(actual)


def _sanitize_graph_selector(name: str) -> str:
    if not name:
        return name
    parts = name.split(":")
    if not parts:
        return name
    parts[-1] = _strip_numeric_suffix(parts[-1])
    return ":".join(parts)


def _parse_shape(s: str) -> Tuple:
    """
    Parse a compact shape string into a tuple:
      - integers -> int
      - '?' or empty -> None
      - other tokens -> symbol strings (e.g., 'B')
    Examples:
      "Bx20"      -> ('B', 20)
      "?x10"      -> (None, 10)
      "20"        -> (20,)
      "" or None  -> ()
    """
    if s is None:
        return tuple()
    s = s.strip()
    if not s:
        return tuple()
    # normalize separators and remove spaces: "B x 20" -> "Bx20"
    s = _SHAPE_SEP.sub("x", s.replace(" ", ""))
    parts = s.split("x") if s else []
    dims: List[ShapeDim] = []
    for p in parts:
        if p in ("?", "None", ""):
            dims.append(None)
            continue
        try:
            dims.append(int(p))
        except Exception:
            dims.append(p)  # treat as symbol
    return tuple(dims)


class _GraphView:
    def __init__(self, model, *, search_functions: bool, passthrough_ops: set[str]):
        self.model = model
        self.search_functions = search_functions
        self.passthrough_ops = set(passthrough_ops)
        self.errors: List[str] = []

        # Graph inventory: top + functions (duck-typed)
        self.graphs: List[Tuple[str, Any]] = []
        self._add_graph("top", _top_graph(model))
        if search_functions:
            for name, g in _function_graphs(model):
                self._add_graph(f"fn:{name}", g)

        # Build shape indices (name -> tuple dims) for each graph we see (ONNX only).
        self._shape_index: Dict[str, Dict[str, Tuple[ShapeDim, ...]]] = {}
        # Live-node indices (reachable from graph outputs) for each graph.
        self._live_index: Dict[str, Set[int]] = {}
        for gname, g in self.graphs:
            self._shape_index[gname] = _build_shape_index(g)
            self._live_index[gname] = _compute_live_node_indices(g)

    def _add_graph(self, name: str, g: Any):
        if g is not None:
            self.graphs.append((name, g))

    def _fail(self, msg: str):
        self.errors.append(msg)

    # -- basic queries --

    def count_op(
        self,
        op_type: str,
        *,
        live_only: bool = True,
        graphs: Optional[Iterable[str]] = None,
    ) -> int:
        c = 0
        allowed = set(graphs) if graphs is not None else None
        for gname, g in self.graphs:
            if allowed is not None and gname not in allowed:
                continue
            nodes = _nodes(g)
            live = self._live_index.get(gname, set()) if live_only else None
            for idx, n in enumerate(nodes):
                if live is not None and idx not in live:
                    continue
                current = getattr(n, "op_type", "")
                if _op_matches(op_type, current):
                    c += 1
        return c

    def list_ops(
        self,
        op_type: str,
        *,
        live_only: bool = True,
        graphs: Optional[Iterable[str]] = None,
    ) -> List[Tuple[str, int]]:
        out = []
        allowed = set(graphs) if graphs is not None else None
        for gname, g in self.graphs:
            if allowed is not None and gname not in allowed:
                continue
            nodes = _nodes(g)
            live = self._live_index.get(gname, set()) if live_only else None
            for idx, n in enumerate(nodes):
                if live is not None and idx not in live:
                    continue
                current = getattr(n, "op_type", "")
                if _op_matches(op_type, current):
                    out.append((gname, idx))
        return out

    def _unused_inputs_for_graph(self, g) -> List[str]:
        used = set()
        for n in _nodes(g):
            for v in _inputs_of(n):
                nm = _value_name(v)
                if nm:
                    used.add(nm)
        outs = _graph_outputs(g)
        for v in outs:
            nm = _value_name(v)
            if nm:
                used.add(nm)
        res = []
        for v in _graph_inputs(g):
            nm = _value_name(v)
            if nm and nm not in used:
                res.append(nm)
        return res

    def unused_graph_inputs(self) -> List[str]:
        _name, g = self.graphs[0]  # top only
        return self._unused_inputs_for_graph(g)

    def unused_function_inputs(self) -> Dict[str, List[str]]:
        res: Dict[str, List[str]] = {}
        for name, g in self.graphs[1:]:
            unused = self._unused_inputs_for_graph(g)
            if unused:
                res[name] = unused
        return res

    # -- Path matcher with inline shapes --

    def match_path_with_shapes(
        self,
        path: str,
        *,
        symbols: Mapping[str, ShapeDim],
        attrs: Optional[Dict[str, Any]] = None,
        counts: Optional[Dict[str, int]] = None,
        graph_filter: Optional[Any] = None,
        must_absent: Optional[Iterable[str]] = None,
        input_constraints: Optional[Mapping[Any, Any]] = None,
    ) -> Tuple[bool, str, List[int]]:
        attrs = attrs or {}
        counts = counts or {}
        ok_any_graph = False
        reasons: List[str] = []
        allowed = _normalize_graph_filter(graph_filter)
        considered_any = False
        matched_indices: List[int] = []

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
            if allowed is not None and not _graph_filter_allows(allowed, gname):
                continue
            considered_any = True
            ok, why, matched = _match_path_on_graph(
                g,
                steps,
                dict(symbols),
                self.passthrough_ops,
                gname,
                self._shape_index[gname],
            )
            if not ok:
                reasons.append(why)
                continue

            if input_constraints:
                nodes_seq = _nodes(g)
                if not matched:
                    reasons.append(
                        f"[{gname}] internal error: path matched but no node indices captured"
                    )
                    continue
                target_idx = matched[-1]
                if target_idx >= len(nodes_seq):
                    reasons.append(
                        f"[{gname}] matched node index out of range for path {path!r}"
                    )
                    continue
                node_obj = nodes_seq[target_idx]
                ok_inputs, err = _check_input_constraints(
                    node_obj,
                    input_constraints,
                    nodes_seq,
                    g,
                )
                if not ok_inputs:
                    reasons.append(f"[{gname}] {err}")
                    continue

            if must_absent:
                violation = None
                for op in must_absent:
                    if self.count_op(op, graphs=[gname]) > 0:
                        violation = (
                            f"[{gname}] operator '{op}' present but must be absent"
                        )
                        break
                if violation is not None:
                    reasons.append(violation)
                    continue

            if counts:
                mismatch = None
                for op, want in counts.items():
                    got = self.count_op(op, live_only=False, graphs=[gname])
                    if got != want:
                        mismatch = (
                            f"[{gname}] expected {want} '{op}' nodes but found {got}"
                        )
                        break
                if mismatch is not None:
                    reasons.append(mismatch)
                    continue

            if attrs:
                attr_issue = None
                for op, reqs in attrs.items():
                    nodes_for_op = [n for n in _nodes(g) if n.op_type == op]
                    if not nodes_for_op:
                        attr_issue = (
                            f"[{gname}] operator '{op}' not found for attribute check"
                        )
                        break
                    if any(not _node_has_attrs(n, reqs) for n in nodes_for_op):
                        attr_issue = (
                            f"[{gname}] operator '{op}' missing required attributes"
                        )
                        break
                if attr_issue is not None:
                    reasons.append(attr_issue)
                    continue

            ok_any_graph = True
            matched_indices = matched
            break
        if not considered_any and allowed is not None:
            self._fail(f"graph not found: {graph_filter!r}")
            return False, f"graph not found: {graph_filter!r}", []
        if not ok_any_graph:
            reason = (
                f"path not found: {path!r} :: " + " | ".join(reasons)
                if reasons
                else f"path not found: {path!r}"
            )
            return False, reason, []
        return True, "", matched_indices


# ---------- helpers for IR/ONNX without importing onnx ----------


def _node_has_attrs(node, reqs: Dict[str, Any]) -> bool:
    attrs = getattr(node, "attributes", None)
    if attrs is None:
        attrs = getattr(node, "attribute", None)
    attr_map = {}
    if isinstance(attrs, dict):
        attr_map = attrs
    elif isinstance(attrs, (list, tuple)):
        for a in attrs:
            name = getattr(a, "name", None)
            if name:
                val = getattr(a, "value", None)
                if val is None and hasattr(a, "ints"):
                    val = tuple(getattr(a, "ints"))
                attr_map[name] = val
    else:
        attr_map = getattr(node, "_attributes", {}) or {}
    for key, expected in (reqs or {}).items():
        if attr_map.get(key) != expected:
            return False
    return True


def _top_graph(model):
    return getattr(model, "graph", None)


def _function_graphs(model):
    """
    Return iterable of (name, graph_or_graphlike).
    - onnx_ir: model.functions (dict/list) with .graph
    - ONNX: model.functions (list of FunctionProto) — we return the function object
            directly; _nodes() can read 'node' from it; shapes may be unavailable.
    """
    funcs = (
        getattr(model, "functions", None) or getattr(model, "_functions", None) or []
    )
    if isinstance(funcs, dict):
        vals = funcs.values()
        for f in vals:
            yield (
                f"{getattr(f,'domain','')}:{getattr(f,'name','')}",
                getattr(f, "graph", None),
            )
    else:
        try:
            for f in funcs:
                # Try to expose a graph-like: prefer .graph, else the function itself (has .node)
                g = getattr(f, "graph", f)
                yield (f"{getattr(f,'domain','')}:{getattr(f,'name','')}", g)
        except Exception:
            return


def _nodes(g):
    # onnx_ir graphs may use .nodes/_nodes; ONNX uses .node
    return list(getattr(g, "nodes", getattr(g, "_nodes", getattr(g, "node", []))))


def _graph_filter_allows(
    normalized: Tuple[Set[str], Set[str], Set[str]], graph_name: str
) -> bool:
    exact, prefixes, _ = normalized
    if graph_name in exact:
        return True
    return any(graph_name.startswith(prefix) for prefix in prefixes)


def _normalize_graph_filter(
    graph_filter: Any,
) -> Optional[Tuple[Set[str], Set[str], Set[str]]]:
    if graph_filter is None:
        return None
    items: List[str]
    if isinstance(graph_filter, str):
        items = [graph_filter]
    else:
        try:
            items = list(graph_filter)
        except TypeError:
            items = [str(graph_filter)]

    exact_matches: Set[str] = set()
    prefix_matches: Set[str] = set()
    recorded_entries: Set[str] = set()

    def _add_entry(entry: str):
        if not entry:
            return
        exact_matches.add(entry)
        recorded_entries.add(entry)
        sanitized = _sanitize_graph_selector(entry)
        exact_matches.add(sanitized)
        recorded_entries.add(sanitized)
        if ":" not in entry:
            prefix_matches.add(f"{entry}:")
            prefix_matches.add(f"{sanitized}:")

    for item in items:
        if not isinstance(item, str):
            continue
        if item == "top":
            _add_entry("top")
            continue
        if item.startswith("fn:"):
            _add_entry(item)
            trimmed = item[3:]
            if trimmed:
                _add_entry(trimmed)
        else:
            _add_entry(item)
            _add_entry(f"fn:{item}")
    if not exact_matches and not prefix_matches:
        return None
    return exact_matches, prefix_matches, recorded_entries


def _graph_inputs(g):
    arr = getattr(g, "inputs", getattr(g, "input", []))
    try:
        return list(arr)
    except Exception:
        return []


def _graph_outputs(g):
    arr = getattr(g, "outputs", getattr(g, "output", []))
    try:
        return list(arr)
    except Exception:
        return []


def _inputs_of(n):
    return getattr(n, "inputs", getattr(n, "input", []))


def _outputs_of(n):
    return getattr(n, "outputs", getattr(n, "output", []))


def _value_name(v) -> str:
    return getattr(v, "name", v if isinstance(v, str) else "")


def _shape_of_value(v) -> Optional[Tuple[ShapeDim, ...]]:
    """onnx_ir Value -> shape tuple; ONNX will use _shape_of_output via index."""
    shp = getattr(v, "shape", None)
    if shp is None:
        return None
    # onnx_ir shape → tuple of ints/strings/None
    dims: List[ShapeDim] = []
    for d in getattr(shp, "dims", getattr(shp, "dim", shp)):
        if isinstance(d, int):
            dims.append(d)
        else:
            try:
                dims.append(int(d))
            except Exception:
                s = str(d)
                dims.append(None if s in ("None", "", "?", "unk", "unknown") else s)
    return tuple(dims)


# ---------- ONNX shape index (duck-typed) ----------


def _build_shape_index(g) -> Dict[str, Tuple]:
    """
    For ONNX graphs: read shapes from value_info / inputs / outputs (duck-typed).
    For onnx_ir graphs: return {} and rely on Value.shape at use sites.
    """
    idx: Dict[str, Tuple] = {}
    # If there is no 'value_info', we assume onnx_ir and skip.
    has_vi = hasattr(g, "value_info")
    if not has_vi and not hasattr(g, "input") and not hasattr(g, "output"):
        return idx
    for coll_name in ("value_info", "input", "output"):
        coll = getattr(g, coll_name, None)
        if coll is None:
            continue
        try:
            for vi in coll:
                name = getattr(vi, "name", "")
                shp = _shape_from_value_info(vi)
                if name and shp is not None:
                    idx[name] = shp
        except Exception:
            continue
    return idx


def _shape_from_value_info(vi) -> Optional[Tuple]:
    """
    Duck-typed extraction from ONNX ValueInfoProto:
      vi.type.tensor_type.shape.dim -> list of dims with fields dim_value / dim_param
    """
    try:
        ttype = getattr(getattr(vi, "type", None), "tensor_type", None)
        shape = getattr(ttype, "shape", None)
        dims_msg = getattr(shape, "dim", None) or getattr(shape, "dims", None)
        if dims_msg is None:
            return None
        dims: List[ShapeDim] = []
        for d in dims_msg:
            # Try HasField if present (protobuf API); otherwise, fall back
            has_field = getattr(d, "HasField", None)
            if callable(has_field):
                if has_field("dim_value"):
                    dims.append(int(getattr(d, "dim_value")))
                    continue
                if has_field("dim_param"):
                    dp = getattr(d, "dim_param")
                    dims.append(str(dp) if dp else None)
                    continue
                dims.append(None)
            else:
                # Fallback: read attributes directly (0 dim_value == unknown in practice)
                dv = getattr(d, "dim_value", 0)
                dp = getattr(d, "dim_param", "") or ""
                if isinstance(dv, int) and dv != 0:
                    dims.append(int(dv))
                elif dp:
                    dims.append(str(dp))
                else:
                    dims.append(None)
        return tuple(dims)
    except Exception:
        return None


def _shape_of_output(v, shape_index: Dict[str, Tuple]) -> Optional[Tuple]:
    """
    v can be a name (ONNX) or a Value (onnx_ir).
    """
    # onnx_ir Value
    if hasattr(v, "shape"):
        return _shape_of_value(v)
    # ONNX name lookup
    name = v if isinstance(v, str) else getattr(v, "name", "")
    return shape_index.get(name)


# ---------- Strict unification ----------


def _unify_shape(
    expected: Tuple[ShapeDim, ...],
    actual: Optional[Tuple[ShapeDim, ...]],
    env: Dict[str, ShapeDim],
) -> bool:
    """
    Strict unification:
      - If expected is an int: actual must be the same int (not None).
      - If expected is a symbol (str): actual must be a concrete int; bind or check.
      - If expected is None ('?'): accept any (including None).
      - Ranks must match.
    """
    if actual is None:
        # Accept only if *all* expected dims are '?' (None)
        return all(e is None for e in expected)
    if len(expected) != len(actual):
        return False
    for e, a in zip(expected, actual):
        if e is None:
            continue
        if isinstance(e, int):
            if not isinstance(a, int):
                return False
            if a != e:
                return False
        elif isinstance(e, str):
            aval: ShapeDim
            if isinstance(a, int):
                aval = a
            elif isinstance(a, str):
                sval = a.strip()
                if sval in ("", "None", "?", "unk", "unknown"):
                    aval = None
                else:
                    aval = sval
            else:
                if a is None:
                    aval = None
                else:
                    try:
                        aval = int(a)  # type: ignore[arg-type]
                    except Exception:
                        sval = str(a).strip()
                        aval = (
                            None
                            if sval in ("", "None", "?", "unk", "unknown")
                            else sval
                        )
            if aval is None:
                return False
            bound = env.get(e)
            if bound is None:
                env[e] = aval
            else:
                if bound != aval:
                    return False
        else:
            return False
    return True


# ---------- Core matching ----------


def _match_path_on_graph(
    g,
    steps: List[Tuple[str, Optional[Tuple]]],
    env: Dict[str, ShapeDim],
    passthrough_ops: set[str],
    gname: str,
    shape_index: Dict[str, Tuple],
) -> Tuple[bool, str, List[int]]:
    nodes = _nodes(g)
    consumer_map = _build_consumer_map(nodes)
    # Try all starting candidates
    start_op = steps[0][0]
    starts = [i for i, n in enumerate(nodes) if _op_matches(start_op, n.op_type)]
    if not starts:
        present = sorted({_strip_numeric_suffix(n.op_type) for n in nodes})
        show = ", ".join(present[:10])
        return (
            False,
            f"[{gname}] missing start op '{start_op}' (ops present: {show}{'…' if len(present)>10 else ''})",
            [],
        )
    reason = "start mismatch"
    for i0 in starts:
        env_copy = dict(env)
        matched_indices: List[int] = []
        ok, r = _path_from(
            nodes,
            i0,
            steps,
            env_copy,
            passthrough_ops,
            shape_index,
            consumer_map,
            matched_indices,
        )
        if ok:
            env.update(env_copy)
            return True, "", matched_indices
        reason = r
    return False, reason, []


def _path_from(
    nodes,
    i0: int,
    steps: List[Tuple[str, Optional[Tuple]]],
    env: Dict[str, ShapeDim],
    passthrough_ops: set[str],
    shape_index: Dict[str, Tuple],
    consumer_map: Dict[Tuple, List[int]],
    matched: Optional[List[int]] = None,
) -> Tuple[bool, str]:
    matched_local: List[int] = []
    i = i0
    if not _op_matches(steps[0][0], nodes[i].op_type):
        return False, "start mismatch"
    matched_local.append(i)

    # shape after first node (check the MAIN data edge = first output only)
    sh = steps[0][1]
    if sh is not None:
        outs = _outputs_of(nodes[i])
        if not outs:
            return False, f"no outputs on first node {nodes[i].op_type}"
        # check ONLY first output (main data tensor)
        a0 = _shape_of_output(outs[0], shape_index)
        if not _unify_shape(sh, a0, env):
            return (
                False,
                f"shape mismatch after {nodes[i].op_type}: expected {sh}, actual_first={a0}",
            )

    # walk adjacency by “shares an output → input”; allow passthrough ops
    for s in range(1, len(steps)):
        want_op, want_shape = steps[s]
        next_idx, trace = _walk_to_op(nodes, i, want_op, passthrough_ops, consumer_map)
        if next_idx is None:
            return (
                False,
                f"could not reach '{want_op}' from '{nodes[i].op_type}'"
                + (f" (chain: {' -> '.join(trace)})" if trace else ""),
            )
        if want_shape is not None:
            outs = _outputs_of(nodes[next_idx])
            if not outs:
                return False, f"no outputs on node {nodes[next_idx].op_type}"
            # check ONLY first output (main data tensor)
            a0 = _shape_of_output(outs[0], shape_index)
            if not _unify_shape(want_shape, a0, env):
                return (
                    False,
                    f"shape mismatch after {nodes[next_idx].op_type}: expected {want_shape}, actual_first={a0}",
                )
        i = next_idx
        matched_local.append(i)
    if matched is not None:
        matched.extend(matched_local)
    return True, ""


def _value_keys(v) -> List[Tuple[str, Any]]:
    keys: List[Tuple[str, Any]] = []
    name = _value_name(v)
    if name:
        keys.append(("name", name))
    if not isinstance(v, str):
        try:
            keys.append(("id", id(v)))
        except Exception:
            pass
    return keys if keys else [("anon", id(v))]


def _normalize_input_index(key: Any) -> int:
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        try:
            return int(key)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"input key must be an int or numeric string, got {key!r}"
            ) from exc
    raise TypeError(f"input key must be int or str, got {type(key)!r}")


def _check_input_constraints(
    node, spec: Mapping[Any, Any], nodes, graph
) -> Tuple[bool, str]:
    ins = _inputs_of(node)
    for raw_idx, constraint in spec.items():
        idx = _normalize_input_index(raw_idx)
        rule = constraint or {}
        if rule.get("absent"):
            if idx < len(ins) and not _is_missing_input(ins[idx]):
                return False, f"input[{idx}] expected to be absent"
            continue
        if idx >= len(ins):
            return False, f"input[{idx}] missing (only {len(ins)} inputs present)"
        value = ins[idx]

        init_name = rule.get("initializer_name")
        if init_name is not None:
            if _value_name(value) != init_name:
                return (
                    False,
                    f"input[{idx}] expected initializer '{init_name}', found '{_value_name(value)}'",
                )

        if "const" in rule or "const_bool" in rule:
            arr = _extract_constant_array(value, nodes, graph)
            if arr is None:
                return False, f"input[{idx}] is not constant"
            if "const_bool" in rule:
                actual_bool = bool(np.asarray(arr).reshape(()).astype(np.bool_))
                expected_bool = bool(rule["const_bool"])
                if actual_bool != expected_bool:
                    return (
                        False,
                        f"input[{idx}] expected bool {expected_bool}, found {actual_bool}",
                    )
            if "const" in rule:
                expected = np.asarray(rule["const"])
                actual = np.asarray(arr)
                if actual.shape != expected.shape:
                    if actual.size == 1 and expected.size == 1:
                        actual = actual.reshape(())
                        expected = expected.reshape(())
                    else:
                        return (
                            False,
                            f"input[{idx}] constant shape mismatch: expected {expected.shape}, found {actual.shape}",
                        )
                if actual.dtype.kind == "b" or expected.dtype.kind == "b":
                    if not np.array_equal(
                        actual.astype(np.bool_), expected.astype(np.bool_)
                    ):
                        return False, f"input[{idx}] constant mismatch"
                else:
                    if not np.allclose(
                        actual.astype(np.float64), expected.astype(np.float64)
                    ):
                        return False, f"input[{idx}] constant mismatch"
    return True, ""


def _is_missing_input(val) -> bool:
    if val is None:
        return True
    if isinstance(val, str):
        return val == ""
    name = _value_name(val)
    return name in (None, "")


def _extract_constant_array(
    value, nodes, graph, depth: int = 0
) -> Optional[np.ndarray]:
    if depth > 6:
        return None
    arr = _value_constant_payload(value)
    if arr is not None:
        return arr
    name = _value_name(value)
    if name:
        init_arr = _initializer_to_numpy(graph, name)
        if init_arr is not None:
            return init_arr
    producer = _find_producer_node(nodes, value)
    if producer is None:
        return None
    op_type = getattr(producer, "op_type", "")
    if op_type == "Constant":
        tensor = _constant_attr_tensor(producer)
        return _tensor_to_numpy(tensor)
    if op_type in {"Expand", "Reshape", "Squeeze", "Unsqueeze"}:
        inputs = _inputs_of(producer)
        if inputs:
            return _extract_constant_array(inputs[0], nodes, graph, depth + 1)
    return None


def _value_constant_payload(val) -> Optional[np.ndarray]:
    if isinstance(val, str) or val is None:
        return None
    for attr in ("const_value", "_const_value", "value", "data", "numpy"):
        payload = getattr(val, attr, None)
        if payload is None:
            continue
        arr = _tensor_to_numpy(payload)
        if arr is not None:
            return arr
        try:
            return np.asarray(payload)
        except Exception:
            continue
    return None


def _initializer_to_numpy(graph, name: Optional[str]) -> Optional[np.ndarray]:
    if not name:
        return None
    for attr_name in ("initializer", "initializers", "_initializers"):
        inits = getattr(graph, attr_name, None)
        if inits is None:
            continue
        try:
            iterator = list(inits)
        except Exception:
            continue
        for init in iterator:
            if getattr(init, "name", "") == name:
                arr = _tensor_to_numpy(init)
                if arr is not None:
                    return arr
    return None


def _tensor_to_numpy(tensor) -> Optional[np.ndarray]:
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    if _onnx_numpy_helper is not None:
        try:
            return _onnx_numpy_helper.to_array(tensor)
        except Exception:
            pass
    try:
        arr = np.asarray(tensor)
        if arr.dtype != object:
            return arr
    except Exception:
        pass
    raw = getattr(tensor, "raw_data", None)
    if raw:
        dtype = _onnx_dtype_to_np(getattr(tensor, "data_type", 0))
        arr = np.frombuffer(raw, dtype=dtype)
        dims = getattr(tensor, "dims", None)
        dims = tuple(dims) if dims else ()
        return arr.reshape(dims) if dims else arr
    for field, dtype in (
        ("float_data", np.float32),
        ("double_data", np.float64),
        ("int64_data", np.int64),
        ("int32_data", np.int32),
        ("uint64_data", np.uint64),
        ("float_data", np.float32),
        ("int64s", np.int64),
        ("ints", np.int64),
        ("floats", np.float32),
    ):
        data = getattr(tensor, field, None)
        if data:
            arr = np.array(list(data), dtype=dtype)
            dims = getattr(tensor, "dims", None)
            dims = tuple(dims) if dims else ()
            return arr.reshape(dims) if dims else arr
    return None


def _onnx_dtype_to_np(code: int) -> np.dtype:
    mapping = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        4: np.uint16,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        9: np.bool_,
        10: np.float16,
        11: np.float64,
    }
    return np.dtype(mapping.get(code, np.float32))


def _constant_attr_tensor(node) -> Any:
    attrs = getattr(node, "attributes", None)
    if isinstance(attrs, dict):
        return attrs.get("value")
    attr_list = attrs or getattr(node, "attribute", None)
    if isinstance(attr_list, (list, tuple)):
        for a in attr_list:
            if getattr(a, "name", None) == "value":
                val = getattr(a, "value", None)
                if val is None and hasattr(a, "t"):
                    val = getattr(a, "t")
                return val
    # Some IRs expose _attributes mapping
    extra = getattr(node, "_attributes", None)
    if isinstance(extra, dict):
        return extra.get("value")
    return None


def _find_producer_node(nodes, value) -> Optional[Any]:
    target_name = _value_name(value)
    for node in nodes:
        for out in _outputs_of(node):
            if value is out:
                return node
            if target_name and _value_name(out) == target_name:
                return node
    return None


class _SymbolTable:
    _PREFERRED = tuple("BCTNMRQXYZSH")

    def __init__(self) -> None:
        self._raw_to_symbol: Dict[str, str] = {}
        self._used: set[str] = set()

    def symbol_for(self, raw: Any) -> str:
        key = str(raw)
        cached = self._raw_to_symbol.get(key)
        if cached is not None:
            return cached
        symbol = self._choose_symbol(key)
        self._raw_to_symbol[key] = symbol
        self._used.add(symbol)
        return symbol

    def expect_symbols(self) -> Dict[str, None]:
        if not self._used:
            return {}
        return {sym: None for sym in sorted(self._used)}

    def _choose_symbol(self, candidate: str) -> str:
        token = candidate.strip()
        if _looks_like_symbol(token) and token not in self._used:
            return token
        for sym in self._PREFERRED:
            if sym not in self._used:
                return sym
        idx = 0
        while True:
            sym = f"S{idx}"
            if sym not in self._used:
                return sym
            idx += 1


_SYMBOL_TOKEN: Final[Pattern[str]] = re.compile(r"^[A-Z](?:[A-Z0-9_]{0,3})$")


def _looks_like_symbol(token: str) -> bool:
    if not token:
        return False
    return bool(_SYMBOL_TOKEN.match(token))


def _format_dim(dim: Any, symtab: _SymbolTable) -> str:
    if dim is None:
        return "?"
    if isinstance(dim, int):
        return str(dim)
    if isinstance(dim, str):
        return symtab.symbol_for(dim)
    try:
        intval = int(dim)  # type: ignore[arg-type]
    except Exception:
        return symtab.symbol_for(dim)
    return str(intval)


def _shape_to_spec_str(
    shape: Optional[Tuple[ShapeDim, ...]], symtab: _SymbolTable
) -> str:
    if shape is None:
        return ""
    if not shape:
        return ""
    dims = [_format_dim(dim, symtab) for dim in shape]
    return "x".join(dims)


def _format_step(
    op_type: str, shape: Optional[Tuple[ShapeDim, ...]], symtab: _SymbolTable
) -> str:
    spec_shape = _shape_to_spec_str(shape, symtab)
    return op_type if not spec_shape else f"{op_type}:{spec_shape}"


def _producer_index(nodes) -> Dict[Tuple[str, Any], Tuple[int, Any]]:
    mapping: Dict[Tuple[str, Any], Tuple[int, Any]] = {}
    for idx, node in enumerate(nodes):
        for out_val in _outputs_of(node):
            for key in _value_keys(out_val):
                mapping[key] = (idx, out_val)
    return mapping


def _is_constant_value(nodes, producer_map, graph, value) -> bool:
    arr = _extract_constant_array(value, nodes, graph)
    if arr is not None:
        return True
    for key in _value_keys(value):
        prod = producer_map.get(key)
        if prod is None:
            continue
        node_idx, _ = prod
        node = nodes[node_idx]
        if getattr(node, "op_type", "") == "Constant":
            return True
    return False


def _pick_chain_input(
    nodes, producer_map, graph, inputs: Sequence[Any]
) -> Optional[Any]:
    if not inputs:
        return None
    scored: list[tuple[int, int, Any]] = []
    for idx, val in enumerate(inputs):
        score = 0
        if _is_constant_value(nodes, producer_map, graph, val):
            score = 2
        scored.append((score, idx, val))
    scored.sort()
    return scored[0][2]


def _trace_main_chain(
    output_val,
    nodes,
    producer_map: Dict[Tuple[str, Any], Tuple[int, Any]],
    graph,
    *,
    hop_limit: int,
) -> List[Tuple[Any, Any]]:
    chain: List[Tuple[Any, Any]] = []
    current = output_val
    visited: set[int] = set()
    hops = 0
    while hops < hop_limit:
        producer = None
        for key in _value_keys(current):
            producer = producer_map.get(key)
            if producer is not None:
                break
        if producer is None:
            break
        node_idx, produced = producer
        if node_idx in visited:
            break
        visited.add(node_idx)
        chain.append((nodes[node_idx], produced))
        inputs = _inputs_of(nodes[node_idx])
        if not inputs:
            break
        next_input = _pick_chain_input(nodes, producer_map, graph, inputs)
        if next_input is None:
            break
        current = next_input
        hops += 1
    chain.reverse()
    return chain


def _summarize_graph_primary_paths(
    g,
    *,
    shape_index: Dict[str, Tuple],
    symtab: _SymbolTable,
    hop_limit: int,
) -> List[str]:
    nodes = _nodes(g)
    if not nodes:
        return []
    producer_map = _producer_index(nodes)
    specs: List[str] = []
    seen: set[str] = set()
    for out_val in _graph_outputs(g):
        chain = _trace_main_chain(out_val, nodes, producer_map, g, hop_limit=hop_limit)
        if not chain:
            continue
        parts = []
        for node, produced in chain:
            shape = _shape_of_output(produced, shape_index)
            parts.append(_format_step(getattr(node, "op_type", ""), shape, symtab))
        spec = " -> ".join(parts)
        if spec and spec not in seen:
            seen.add(spec)
            specs.append(spec)
    return specs


def auto_expect_graph_spec(
    model,
    *,
    graph: Optional[Union[str, Sequence[str]]] = "top",
    search_functions: Optional[bool] = None,
    passthrough_ops: Optional[Iterable[str]] = None,
    hop_limit: int = 128,
    include_no_unused_inputs: bool = True,
) -> Dict[str, Any]:
    """Generate expect_graph specs from the current ONNX/IR graph structure."""

    pas = set(passthrough_ops or DEFAULT_PASSTHROUGH_OPS)
    normalized = _normalize_graph_filter(graph)
    if search_functions is None:
        search_functions = normalized is not None and any(
            entry != "top" and not entry.startswith("fn:")
            for entry in (normalized[2] if normalized else set())
        )
    gv = _GraphView(model, search_functions=bool(search_functions), passthrough_ops=pas)
    target_graphs = normalized
    symtab = _SymbolTable()
    specs: List[Union[str, Dict[str, Any]]] = []
    for gname, g in gv.graphs:
        if target_graphs is not None and not _graph_filter_allows(target_graphs, gname):
            continue
        shape_index = gv._shape_index.get(gname, {})
        paths = _summarize_graph_primary_paths(
            g, shape_index=shape_index, symtab=symtab, hop_limit=hop_limit
        )
        for path in paths:
            if gname != "top":
                specs.append({"graph": gname, "path": path})
            else:
                specs.append(path)
    if specs:
        specs = sorted(specs, key=_spec_sort_key)
    result: Dict[str, Any] = {
        "specs": specs,
        "mode": "all",
        "search_functions": bool(search_functions),
    }
    symbols = symtab.expect_symbols()
    if symbols:
        result["symbols"] = symbols
    if include_no_unused_inputs and not gv.unused_graph_inputs():
        result["no_unused_inputs"] = True
    return result


def expect_graph_from_spec(spec: Dict[str, Any]):
    """Instantiate expect_graph from a spec generated by auto_expect_graph_spec."""

    specs = spec.get("specs", [])
    if not specs:
        raise ValueError("spec does not contain any path specifications")
    kwargs = {
        "symbols": spec.get("symbols"),
        "mode": spec.get("mode", "all"),
        "must_absent": spec.get("must_absent"),
        "no_unused_inputs": spec.get("no_unused_inputs", False),
        "no_unused_function_inputs": spec.get("no_unused_function_inputs", False),
        "search_functions": spec.get("search_functions", False),
    }
    return expect_graph(specs, **kwargs)


def expect_graph_from_file(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"expect_graph spec file '{path}' must contain an object")
    return expect_graph_from_spec(data)


def _spec_sort_key(item: Any) -> Tuple[str, str]:
    if isinstance(item, str):
        return ("top", item)
    if isinstance(item, Mapping):
        graph = str(item.get("graph", "")) or "top"
        path = str(item.get("path", ""))
        return (graph, path)
    return ("top", str(item))


def _build_consumer_map(nodes) -> Dict[Tuple, List[int]]:
    mapping: Dict[Tuple, List[int]] = {}
    for idx, node in enumerate(nodes):
        for inp in _inputs_of(node):
            for key in _value_keys(inp):
                mapping.setdefault(key, []).append(idx)
    return mapping


def _consumer_indices(
    nodes, idx: int, consumer_map: Dict[Tuple, List[int]]
) -> List[int]:
    outs = _outputs_of(nodes[idx])
    seen: Set[int] = set()
    result: List[int] = []
    for ov in outs:
        for key in _value_keys(ov):
            for cand in consumer_map.get(key, []):
                if cand == idx or cand in seen:
                    continue
                seen.add(cand)
                result.append(cand)
    return result


def _walk_to_op(
    nodes,
    i: int,
    target_op: str,
    passthrough_ops: set[str],
    consumer_map: Dict[Tuple, List[int]],
) -> Tuple[Optional[int], List[str]]:
    candidates = _consumer_indices(nodes, i, consumer_map)
    if not candidates:
        return None, []

    # Prioritise direct matches before broader search.
    for cand in candidates:
        if _op_matches(target_op, nodes[cand].op_type):
            return cand, [nodes[cand].op_type]

    queue: deque[Tuple[int, List[str]]] = deque()
    visited: Set[int] = set()
    best_trace: List[str] = []

    for cand in candidates:
        op = nodes[cand].op_type
        if op in passthrough_ops:
            queue.append((cand, [op]))
            visited.add(cand)

    while queue:
        idx, trace = queue.popleft()
        if len(trace) > len(best_trace):
            best_trace = trace
        if len(trace) >= 64:
            continue
        for nxt in _consumer_indices(nodes, idx, consumer_map):
            op = nodes[nxt].op_type
            new_trace = trace + [op]
            if _op_matches(target_op, op):
                return nxt, new_trace
            if op in passthrough_ops and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, new_trace))

    return None, best_trace


# ---------- Liveness (reachable-from-outputs) ----------


def _compute_live_node_indices(g) -> Set[int]:
    """
    Compute the set of node indices reachable from graph outputs by walking
    backwards along producer edges (name-based). Works for ONNX and onnx_ir.
    """
    nodes = _nodes(g)
    if not nodes:
        return set()
    # Build name -> producer index map
    prod_by_name: Dict[str, int] = {}
    for idx, n in enumerate(nodes):
        for out_name in _outputs_of(n):
            nm = _value_name(out_name)
            if nm:
                prod_by_name[nm] = idx
    # Seed with graph outputs (names)
    frontier: List[str] = [_value_name(v) for v in _graph_outputs(g) if _value_name(v)]
    live_nodes: Set[int] = set()
    visited_tensors: Set[str] = set()
    steps = 0
    while frontier and steps < 100000:
        steps += 1
        name = frontier.pop()
        if not name or name in visited_tensors:
            continue
        visited_tensors.add(name)
        pidx = prod_by_name.get(name)
        if pidx is None:
            continue
        if pidx in live_nodes:
            continue
        live_nodes.add(pidx)
        # enqueue this producer's *data-flow* input tensor names
        # Special-case known ops with control-like inputs (e.g., Dropout.training_mode at idx 2)
        n = nodes[pidx]
        ins = list(_inputs_of(n))
        if getattr(n, "op_type", "") == "Dropout":
            if len(ins) > 2 and _value_name(ins[2]):
                ins = ins[:3]
            else:
                ins = ins[:2]
        for iv in ins:
            ivn = _value_name(iv)
            if ivn:
                frontier.append(ivn)
    return live_nodes
