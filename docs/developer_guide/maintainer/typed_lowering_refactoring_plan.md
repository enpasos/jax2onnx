# Typed Lowering Core Refactoring Plan

## Objective and Boundaries

Make the path from a JAX equation to graph-connected `ir.Value` outputs easy
to read and statically check. The normal path should have one lowering
contract, explicit value operations, and public ONNX IR graph APIs. Keep
existing numerical behavior, graph structure, ONNX Function signatures, and
control-flow wiring unless a reproduced bug justifies a separately tested fix.

The concrete benefit is fewer decisions at each equation: dispatch does not
inspect plugin signatures, guess return shapes, or probe context capabilities;
graph helpers do not rediscover ownership through names. Necessary semantic
branches and `try/finally` restoration remain explicit. Moving every existing
branch into a universal adapter would not meet this objective.

Initial scope: `typing_support.py`, `lowering_dispatch.py`, `output_binding.py`,
value management in `ir_context.py`, `optimizer_graph_utils.py`, plugin
registration, and a small reference plugin/builder pilot. Excluded: wholesale
dynamic builder migration, optimizer rearchitecture, changed dtype policy or
pass ordering, and public API redesign. Do not merge the PR as part of this work.

## Stage 0: Preserve and Stabilize the Existing Changes

1. Inventory the working diff, preserve the authorized code/docs/tests on a
   dedicated branch, and keep generated ONNX artifacts out of Git.
2. Reproduce and fix current regressions before extending the refactoring.
   Cover removed producers, stale node snapshots, failed-import retries,
   precision-context restoration, and dtype comparisons where affected.
3. Run focused regressions, the full generated/local suite, formatting, lint,
   typing, builder-policy checks, and the strict documentation build.
4. Push the baseline and dispatch `.github/workflows/ci.yml` for that branch.
   Record the checked-out SHA and wait for every job's terminal result.
5. Open the PR after the baseline CI succeeds. Continue the stages below in
   separately reviewable commits on that PR; update its description to match
   the final implementation.

Acceptance: the preserved baseline has passing local checks and green remote
CI, with a PR URL and baseline commit/run recorded. A dispatched or running
workflow is not passing evidence.

## Stage 1: One Canonical Lowering Contract

First-party lowerings implement:

```python
from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.typing_support import LoweringContextProtocol


def lower(ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
    ...  # Emit IR nodes and bind every non-drop equation output.
```

Tighten the existing protocol and normalize registered lowerings once. The
equation loop invokes the canonical interface directly; remove its signature
cache and repeated signature/result-form discovery. Keep original plugin
identity for stacktrace metadata and primitive-call diagnostics.

Preserve supported legacy interfaces through an explicit registration adapter:
`get_handler(converter)`, `lower(ctx, eqn, params)`, and return-only lowerings.
The adapter binds returned values and preserves existing non-drop/partial
binding rules. Internal registries and their documented extension points must
remain usable. Invalid lowerings report a concrete registration/binding error.

Acceptance: canonical lowerings have the typed `None` contract, the canonical
dispatch path does no signature inspection or result coercion, and regressions
prove legacy handlers, params, returns, metadata, and unsupported-plugin errors.

## Stage 2: Explicit Value Operations

Give `IRContext` and `LoweringContextProtocol` three deliberate operations:

| Operation | Contract |
| --- | --- |
| `lookup_value_for_var` | Return an existing `ir.Value` or `None`; no allocation or literal materialization. |
| `require_value_for_var` | Return an input binding, materializing a JAX literal when needed; report a missing binding. |
| `allocate_value_for_var` | Reuse an existing value or explicitly create an output placeholder. |

Replace the expected `KeyError`/catch allocation path with lookup. Use these
operations in binding assertions and representative lowerings. Keep
`get_value_for_var` as a documented compatibility convenience while existing
plugins are migrated deliberately, with input/output intent visible at each
new or touched call site.

Preserve literal dtype handling, symbolic-dimension origins, output identity,
ONNX Function constants as `Constant` nodes, function precision rules, drop
variables, and control-flow captures. Use JAX types through `_compat/jax.py`;
keep duck-typed support confined to documented compatibility boundaries.

Acceptance: lookup has no side effects, allocation uses no exception as its
normal branch, missing inputs fail at their equation, and literal/function/
control-flow regressions retain their expected graph structure and numerics.

## Stage 3: Graph-Native Ownership and Access

Use `ir.Graph`, `ir.Node`, and `ir.Value` for canonical graph queries. Read
`value.producer()`, `value.consumers()`, node/value ownership, and graph
input/initializer membership directly. A removed producer does not connect an
output merely because the value retains graph metadata.

Provide explicitly named compatibility helpers for detached node sequences or
name-based lookup, preserving supported tests and callers. Typed graph helpers
must not silently switch to names when an ownership check fails. A snapshot's
matching length or endpoints never establishes membership.

Reuse public ONNX IR mutation helpers such as `replace_all_uses_with` and
`graph.remove` where they preserve the pass contract; reuse constant access
helpers supported by the minimum dependency version. Keep graph outputs,
initializers, use-def links, and subgraph/function ownership intact.

Acceptance: graph-native helpers use public typed APIs without broad fallback
catch blocks; tests distinguish identity from names, detached producers, stale
snapshots, shared-name values, and graph inputs/initializers. Existing optimizer
structure tests still pass without changing pass order.

## Stage 4: Reference Patterns and Typed Builder Pilot

Document one complete canonical plugin and one graph rewrite: require inputs,
allocate or emit typed outputs, attach dtype/shape metadata, bind outputs, and
preserve ownership. Link the pattern from the architecture, plugin, builder,
and typing guides so contributors have a single coherent contract.

Add typed `op`/`op_multi_out` signatures to the existing builder protocol using
public `onnx_ir.tape` types where available. Migrate a small representative
single-/multi-output pilot, including optional inputs and explicit outputs.
Keep project naming, metadata, initializer bookkeeping, and existing dynamic
shorthand compatible. Do not invent a second operator/type hierarchy or expand
this stage into a repository-wide builder rewrite.

Acceptance: the reference patterns match executable code; mypy checks the pilot
against real ONNX IR types; policy, numerical, and structural tests pass on
supported JAX/Python stacks and minimum dependencies. Final PR-head CI is green.

## Validation and Completion Evidence

Use focused behavior/structure tests after each stage. Before baseline shipping
and final completion, run the generated/full test suite and:

```bash
poetry run ruff check .
poetry run pre-commit run --all-files
./scripts/check_typing.sh
poetry run python scripts/check_ir_builder_usage.py
poetry run mkdocs build --strict --site-dir /tmp/jax2onnx-typed-lowering-docs
git diff --check
```

Remote CI is the final stack gate: Python 3.11's older resolver stack, the newer
JAX stacks, minimum dependency versions, normalization accuracy, and web smoke
checks. Do not infer one stack's success from another. Each completion entry
needs test results plus implementation evidence for its contract; green tests
alone do not prove that runtime guessing was removed.

| Stage | Status | Completion evidence |
| --- | --- | --- |
| 0: Baseline stabilization and PR | In progress | Record baseline SHA, local checks, successful CI run, and PR URL. |
| 1: Canonical lowerings | Pending | Record canonical dispatch/protocol changes and compatibility regressions. |
| 2: Value operations | Pending | Record explicit call sites and literal/function/control-flow tests. |
| 3: Graph-native helpers | Pending | Record ownership/compatibility boundaries and structural regressions. |
| 4: Reference patterns and builder pilot | Pending | Record pilot, documentation, full checks, and final-head CI run. |
