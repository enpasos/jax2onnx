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
    - match='contains': pattern must occur as a substring of some op-path.
    - match='exact':    pattern must exactly equal some op-path (full match).
    Matching is over root→sink op-paths; extra fan-in/out is allowed.
    """
    regs = [re.compile(p) for p in patterns]
    require_all = mode == "all"

    def _check(model: onnx.ModelProto) -> bool:
        paths = _op_paths(model)
        if match == "exact":

            def matcher(r, s):
                return bool(r.fullmatch(s))

        else:

            def matcher(r, s):
                return bool(r.search(s))

        results = []
        for r in regs:
            ok = any(matcher(r, p) for p in paths)
            results.append(ok)
            if not ok and not require_all:
                return False
        return all(results) if require_all else any(results)

    return _check
