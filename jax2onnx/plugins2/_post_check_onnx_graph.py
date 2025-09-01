# file: jax2onnx/plugins2/_post_check_onnx_graph.py

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Iterable, Set, Any

_TOKEN_RE = re.compile(r"^\s*([A-Za-z_][\w\.]*)\s*(?:\(\s*(\d+)\s*\))?\s*$")
_ANCHOR_START = "^"
_ANCHOR_END = "$"


def _node_in_names(n) -> List[str]:
    xs = getattr(n, "inputs", None)
    if xs is None:
        xs = getattr(n, "input", [])
    out = []
    for v in xs:
        name = getattr(v, "name", None)
        out.append(name if name is not None else str(v))
    return out


def _node_out_names(n) -> List[str]:
    xs = getattr(n, "outputs", None)
    if xs is None:
        xs = getattr(n, "output", [])
    out = []
    for v in xs:
        name = getattr(v, "name", None)
        out.append(name if name is not None else str(v))
    return out


def _build_graph_index(model) -> Tuple[List[Any], Dict[str, List[int]], Dict[int, Set[int]], Dict[int, int]]:
    """
    Returns:
      nodes: list of nodes in model.graph.node
      by_type: op_type -> [node indices] (in graph order)
      adj: index -> set(next indices) if any output of index is consumed by next
      indeg: index -> number of producer nodes (helps with '^' anchor)
    """
    nodes: List[Any] = list(model.graph.node)
    by_type: Dict[str, List[int]] = defaultdict(list)
    for i, n in enumerate(nodes):
        by_type[n.op_type].append(i)

    out_to_idx: Dict[str, int] = {}
    for i, n in enumerate(nodes):
        for o in _node_out_names(n):
            out_to_idx[o] = i

    adj: Dict[int, Set[int]] = {i: set() for i in range(len(nodes))}
    indeg: Dict[int, int] = {i: 0 for i in range(len(nodes))}
    for j, n in enumerate(nodes):
        for in_name in _node_in_names(n):
            i = out_to_idx.get(in_name)
            if i is not None:
                adj[i].add(j)
                indeg[j] += 1
    return nodes, by_type, adj, indeg


def _parse_path_core(s: str) -> List[Tuple[str, Optional[int]]]:
    """
    Parse 'Transpose(0)->Conv(1)->Transpose(2)' into
    [('Transpose', 0), ('Conv', 1), ('Transpose', 2)].
    If '(k)' is omitted, index is None and we match any instance of that op.
    Indices are absolute graph indices (0-based) in model.graph.node.
    """
    parts = [p for p in s.split("->") if p.strip()]
    result: List[Tuple[str, Optional[int]]] = []
    for p in parts:
        m = _TOKEN_RE.match(p)
        if not m:
            raise ValueError(f"Bad segment '{p}' in pattern '{s}'")
        op = m.group(1)
        idx = int(m.group(2)) if m.group(2) is not None else None
        result.append((op, idx))
    return result


def _parse_path(pattern: str) -> Tuple[bool, List[Tuple[str, Optional[int]]], bool]:
    """
    Supports optional anchors:
      '^' at the beginning -> require start node to be a graph source (no producer)
      '$' at the end       -> require final node to be a graph sink (no consumer)
    """
    p = pattern.strip()
    anchored_start = p.startswith(_ANCHOR_START)
    anchored_end = p.endswith(_ANCHOR_END)
    if anchored_start:
        p = p[1:].lstrip()
    if anchored_end:
        p = p[:-1].rstrip()
    tokens = _parse_path_core(p)
    return anchored_start, tokens, anchored_end


def _match_one_path(model, pattern: str, *, require_prefix: bool = False, require_suffix: bool = False) -> bool:
    nodes, by_type, adj, indeg = _build_graph_index(model)
    a_start, tokens, a_end = _parse_path(pattern)
    # Combine explicit anchors in the pattern with the global match mode.
    require_prefix = require_prefix or a_start
    require_suffix = require_suffix or a_end

    # Build candidate node-index sets for each token.
    cand: List[List[int]] = []
    for op, maybe_idx in tokens:
        if maybe_idx is not None:
            # absolute node index; must exist and match the op
            if 0 <= maybe_idx < len(nodes) and nodes[maybe_idx].op_type == op:
                cand.append([maybe_idx])
            else:
                return False
        else:
            lst = by_type.get(op, [])
            if not lst:
                return False
            cand.append(lst)

    # DFS/backtrack along adjacency to find a chain that respects the candidates.
    def dfs(pos: int, node_idx: int) -> bool:
        if pos == len(cand) - 1:
            # If we require a suffix anchor, the final node must be a sink.
            if require_suffix and len(adj.get(node_idx, ())) > 0:
                return False
            return True
        for nxt in cand[pos + 1]:
            if nxt in adj.get(node_idx, ()):
                if dfs(pos + 1, nxt):
                    return True
        return False

    for start in cand[0]:
        # If we require a prefix anchor, the first node must be a source.
        if require_prefix and indeg.get(start, 0) > 0:
            continue
        if dfs(0, start):
            return True
    return False


def expect_graph(patterns: Iterable[str], *, mode: str = "all", match: str = "contains"):
    """
    Factory returning a `post_check_onnx_graph`-compatible callable.

    patterns: e.g. ["Transpose(0)->Conv(1)->Transpose(2)"]
    mode:
      - "all": every pattern must be found in the graph (default)
      - "any": at least one pattern must be found
    match:
      - "contains" (default): pattern may appear as a subpath in a larger chain
      - "prefix":   like '^pattern' (start must be a graph source)
      - "suffix":   like 'pattern$' (end must be a graph sink)
      - "exact":    like '^pattern$' (both ends anchored)
    """
    pats = list(patterns)

    def _checker(model) -> bool:
        req_prefix = match in ("prefix", "exact")
        req_suffix = match in ("suffix", "exact")
        results = [
            bool(_match_one_path(model, p, require_prefix=req_prefix, require_suffix=req_suffix))
            for p in pats
        ]
        return all(results) if mode == "all" else any(results)

    return _checker
