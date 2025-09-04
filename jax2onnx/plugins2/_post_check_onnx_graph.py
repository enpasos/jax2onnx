# file: jax2onnx/plugins2/_post_check_onnx_graph.py

from __future__ import annotations
from typing import List, Sequence
import re
import onnx


def _op_paths(model: onnx.ModelProto) -> List[str]:
    """
    Return all root→sink operator sequences as 'OpA->OpB->OpC' strings.
    Graph inputs/outputs are ignored; only operator nodes are included.
    Fan-in / fan-out is allowed; each DFS branch yields one sequence.
    """
    nodes = list(model.graph.node)
    # tensor -> producing node index
    producer: dict[str, int] = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            if o:
                producer[o] = i
    # adjacency among operator nodes
    succ: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    pred_count: dict[int, int] = {i: 0 for i in range(len(nodes))}
    for j, n in enumerate(nodes):
        for inp in n.input:
            p = producer.get(inp)
            if p is not None:
                succ[p].append(j)
                pred_count[j] += 1
    roots = [i for i in range(len(nodes)) if pred_count[i] == 0]
    sinks = {i for i in range(len(nodes)) if not succ[i]}

    paths: list[str] = []

    def dfs(i: int, acc: List[str]):
        acc.append(nodes[i].op_type)
        if i in sinks:
            paths.append("->".join(acc))
        else:
            for j in succ[i]:
                dfs(j, acc)
        acc.pop()

    for r in roots:
        dfs(r, [])
    # de-duplicate sequences preserving order
    return list(dict.fromkeys(paths))


def expect_graph(patterns: Sequence[str], mode: str = "any", match: str = "contains"):
    """
    Build a checker that validates the presence of operator sequences.
    Matching is over root→sink operator paths extracted from the graph.

    match semantics:
      - 'contains': pattern must occur as a substring of some root→sink op-path.
      - 'prefix'  : pattern must match the start of some root→sink op-path.
      - 'suffix'  : pattern must match the end   of some root→sink op-path.
      - 'exact'   : pattern must exactly equal some root→sink op-path
                    (equivalent to ^pattern$) **AND there must be no other
                    root→sink op-paths in the graph** (i.e., the patterns
                    account for the *entire* operator graph; “nothing else”).
    """
    regs = [re.compile(p) for p in patterns]
    require_all = mode == "all"

    def _check(model: onnx.ModelProto) -> bool:
        paths = _op_paths(model)
        # Fast path for the common 'contains' / 'prefix' / 'suffix' modes
        if match != "exact":
            if match == "prefix":
                def matcher(r, s):  # start-anchored
                    m = r.search(s)
                    return bool(m and m.start() == 0)
            elif match == "suffix":
                def matcher(r, s):  # end-anchored
                    m = r.search(s)
                    return bool(m and m.end() == len(s))
            else:  # 'contains'
                def matcher(r, s):
                    return bool(r.search(s))

            hits = []
            for r in regs:
                ok = any(matcher(r, p) for p in paths)
                hits.append(ok)
                if not ok and not require_all:
                    return False
            return all(hits) if require_all else any(hits)

        # --- strict 'exact' mode: full-match AND nothing else in the graph ---
        # 1) Each pattern must full-match at least one root→sink path (per mode).
        pattern_hit = [False] * len(regs)
        # 2) Every root→sink path in the graph must be accounted for by some pattern.
        path_covered = [False] * len(paths)

        for pi, p in enumerate(paths):
            for ri, r in enumerate(regs):
                if r.fullmatch(p):
                    path_covered[pi] = True
                    pattern_hit[ri] = True

        # Patterns requirement
        if require_all:
            if not all(pattern_hit):
                return False
        else:  # 'any'
            if not any(pattern_hit):
                return False

        # “Nothing else”: every root→sink path must be matched by at least one pattern.
        return all(path_covered)

    return _check
