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
| 0: Baseline stabilization and PR | Complete | Baseline `4eb96a1f`: 4,307 local tests, all hooks/typing/docs checks, and 35 focused tests with ONNX-IR 0.2.1 passed. All eight jobs in [CI run 38636341586](https://github.com/enpasos/jax2onnx/actions/runs/38636341586) passed before [PR #274](https://github.com/enpasos/jax2onnx/pull/274) was opened. |
| 1: Canonical lowerings | Implemented | `lowering_registry.py` prepares calls at registration; all 605 first-party entries use direct native methods. The equation loop uses typed, initialized context state. Registry mutation/identity, legacy signatures/results, explicit converters, nested logs, and metadata restoration are covered in `test_lowering_registry.py` and `test_lowering_dispatch.py`. |
| 2: Value operations | Implemented | Pure lookup, required inputs, and output allocation are explicit in `IRContext`. Native output guards check graph ownership; legacy aliases are normalized at their boundary. Context/output tests cover missing bindings, literal scope/dtype, partial and dropped results, removed producers, and foreign same-name values. |
| 3: Graph-native helpers | Implemented | Optimizer queries receive the live graph and use native producer/consumer ownership. Explicit detached-list/name helpers preserve compatibility. Graph regression tests cover captures, shadowing, stale snapshots, removal, and cross-graph rewrites. 126 focused optimizer tests passed on current and minimum ONNX-IR. |
| 4: Reference patterns and builder pilot | Implemented; final gates below | Typed `op` / `op_multi_out` forward public Tape APIs. Neg and TopK are executable references. The optimizer guide includes an executable graph rewrite preserving names, metadata, outputs, and ownership; the plugin, builder, typing, and architecture guides link the patterns. 30 builder/pilot tests and the graph reference passed on current and minimum ONNX-IR, with generated numerical/structural coverage. |

### First Checkpoint Acceptance (`2f73468b`)

The first implementation checkpoint was checked as one PR head. Local verification
passed 4,345 generated/full tests on Python 3.12.9, JAX 0.11.2, and ONNX-IR 1.0.0,
all repository hooks, typing/RNG and builder-policy checks, and a strict
documentation build. Isolated ONNX-IR 0.2.1 verification
passed 244 converter/control-flow tests with strict optimizer failure handling;
the intentional nonfatal-policy test passed separately with its normal setting.

All eight jobs passed in [CI run 38644853117](https://github.com/enpasos/jax2onnx/actions/runs/38644853117)
for `2f73468b` of [PR #274](https://github.com/enpasos/jax2onnx/pull/274).
The baseline run above is a separate checkpoint and cannot substitute for this
final gate. The PR remains unmerged.

## Follow-up: Coherent Native Conversion Path

The first checkpoint establishes a useful core, but does not complete the
original readability goal. At `2f73468b`, registration still discovers native
contracts from annotations, Function lowering immediately falls back to an
`Any` context and mutable capture dictionaries, and child-context construction
still probes capabilities. The two builder pilots do not demonstrate adoption
through the surrounding conversion path. These are implementation gaps, even
with passing CI.

The next stages finish the same PR. Preserve numerical behavior, precision,
Function signatures, capture ordering, names, graph ownership, and existing
legacy extension conventions. Changes in behavior require a reproduced problem,
a concrete benefit, and an appropriate regression test.

| Stage | Work | Inspectable acceptance |
| --- | --- | --- |
| 5: Explicit native registration | Register first-party lowerings through a typed native entry point. Keep legacy registration and raw mapping assignment at the compatibility boundary. | Native registration does not inspect signatures or annotations. Original objects, aliases, removal/restoration, and legacy decorators/handlers/results retain their contracts. |
| 6: Typed Function phases | Separate callee/capture preparation, key and reuse selection, body construction, and call emission. Use concrete ordered capture records. | The full native route and its helpers accept a typed context. No mutable dictionary entries for dynamic captures, context-capability probes, or broad casts replace the previous `Any` path. Arbitrary Python value inspection and JAX-version details have named boundaries. |
| 7: Consistent child scopes | Share explicit child-context construction for ONNX Functions and control-flow bodies. | No constructor retries, `Any` casts, or method monkeypatches are needed to create a native child. Settings, symbolic origins, recording ownership, naming prefixes, and function bookkeeping preserve their specific sharing/copying policies. |
| 8: Pattern adoption at shared boundaries | Use a closed typed builder interface in native emission. Migrate shared unary/control-flow helpers and surrounding read-existing input/output bindings where they provide a concrete consistency gain. | Native emission is checked without unrestricted dynamic attribute lookup. Inputs and completed outputs require bindings; output allocation is deliberate. Optional inputs, metadata, dtype/shape, initializer representation, and output identity remain correct. |
| 9: Coherence and final acceptance | Update the linked guides and executable examples around the resulting production path, remove redundant adapters/helpers, and critically review the architecture. | A reader can follow the native chain without understanding legacy adapters. Remaining branches reflect semantic decisions or documented compatibility. Full local checks and all final-head CI jobs pass; source review separately demonstrates the readability and typing improvements. |

Adoption is based on its effect on the conversion path. Registration-only
declarations may change across first-party plugins to make their contract
explicit; this does not require rewriting all operator lowerings. Dynamic
builder shorthand can remain available for existing leaf plugins while native
emission uses a closed interface.

### Follow-up validation

Use the existing Function suites for reuse/uniqueness, captures, runtime flags,
namespaces, nested functions, constant representation, symbolic shapes, and
precision. Add outcome regressions for ordered tensor keyword captures,
unused dynamic captures, and child-origin/settings inheritance where coverage
is missing. Run converter and control-flow tests after the shared scope change,
and numerical/structural unary tests after the shared helper migration.

Prove the native builder interface rejects invalid inputs and unknown methods
with a negative typing example. Inspect the named production paths as well as
running tests: lowerings, capture records, child creation, native emission,
value binding, and graph discovery. A lower global `if`/`try` count alone is
not an acceptance criterion; dtype choices and `try/finally` restoration remain
necessary.

Before final shipping, regenerate tests, run the full bounded local suite and
the checks above, verify the minimum ONNX-IR surface, and wait for terminal
success of every CI job on the new PR head. Preserve the first checkpoint's
evidence separately from these new results. Do not merge or post review replies.

| Follow-up stage | Status | Evidence |
| --- | --- | --- |
| 5: Explicit registration | Implemented | All 605 discovered first-party entries register direct methods without signature inspection. Native aliases/restoration, weak-reference lifetime, slots-only instances, and legacy decorator/result conventions have regressions. Of 399 registration declarations migrated, 393 files contain only declaration changes; six also adopt the shared native patterns below. |
| 6: Function phases | Implemented | The former 373-line `_lower_in_context` is a 17-line coordinator over typed capture/key/trace/build/emit phases. Ordered tensor kwargs, unused formals, cached reuse, static-state differences, hoisted constants, and scalar/vector constant kwargs have structural and ORT checks. |
| 7: Child scopes | Implemented | One factory owns settings, symbolic origins, shared bookkeeping, and naming. Child tests cover Function/If scopes, graph-local values, recording ownership, exact names, nested Function definitions, and independently varied runtime flags. Loop/scan remain explicit. |
| 8: Shared adoption | Implemented | Neg/TopK, shared unary/cast helpers, shared control-flow helpers, Function calls, and NCHW output emission use the closed `IRBuilderAPI`. Final outputs and JIT/PJIT/custom-VJP inputs require bindings. Positive/negative mypy examples prove invalid inputs and unknown builder methods are rejected. |
| 9: Final acceptance | Architecture accepted; final validation gate | Read-only architectural review found substantive simplification across the named native chain. Final local suite, static checks, executable documentation, minimum-IR results, and final-head CI must all satisfy the acceptance below. |

### Architectural assessment and behavior benefits

The Function split removes partially populated capture dictionaries and context
capability discovery. All native phase helpers take concrete contexts, values,
specs, and records; the coordinator makes the cache/build/call decision visible.
Child setup now lives once instead of in competing Function/control-flow
constructors. Shared control-flow helpers also use public `Value` fields and
typed Tape methods, removing their remaining reflection/cast/exception shims.
The helper cleanup was required by the architectural review, despite passing
tests before it.

Remaining branches represent dtype choices, optional metadata, reuse/uniqueness,
scope capture rules, or restoration with `try/finally`. Arbitrary Python
configuration inspection and JAX-private tracer access have named boundaries.
Legacy extensions retain their adapter, while existing leaf operators retain
dynamic shorthand. New native helpers and both reference lowerings use the
closed builder surface. The weak identity catalog preserves mutable-registry
compatibility without retaining discarded plugins.

The behavior changes have reproduced benefits:

- JAX 0.11 tensor keyword captures resolve their direct JAX atom instead of
  failing conversion with `TracerArrayConversionError`.
- Resolved scalar constants are passed into body tracing as constants; the
  earlier code retained an outer tracer and failed the same conversion.
- Functions built inside control-flow bodies retain their model-level
  definitions and registry, rather than failing or dropping definitions.
- If branches lazily capture declared external BOOL parameters into local
  values. Both If → Function and Function → If → Function exports execute
  correctly while predicate and flag vary independently. Function formals and
  Loop/scan carried-input arities retain their scope contracts.

The legacy producer helper still accepts live Graph sequences and retains its
historical foreign/disconnected-name fallback. Regression tests document that
contract; the stricter native graph helpers are unchanged.

### Follow-up verification and remote acceptance

The final local implementation passed **4,377 generated/full tests**, with zero
skips, errors, or failures, on Python 3.12.9 / JAX 0.11.2 / ONNX-IR 1.0.0.
All repository hooks, Ruff, typing/RNG, builder policies, and the strict
documentation build passed. The newly staged child test also passed all nine
cases after its initializer call was adjusted to the keyword-naming policy.

An isolated, verified ONNX-IR **0.2.1** installation passed **281** converter,
control-flow, graph, and builder tests with strict optimizer failures. The
intentional nonfatal-policy test passed separately under its normal setting.
Shared control-flow helper checks passed 65 focused and 92 generated cases on
each IR version. Exact Markdown examples for native Neg registration, typed
Transpose, Split, and optional-middle-input Clip passed graph/type assertions,
the ONNX checker, and ORT execution on both versions. Positive/negative mypy
examples accepted valid emission and rejected incorrect inputs and unknown
builder methods. A fresh plugin import with signature inspection forbidden
confirmed all 605 registrations use direct native methods.

The architectural review accepts the named native path against the original
readability, typing, and shared-pattern goals. Final remote acceptance additionally
requires terminal success of **all eight CI jobs at the new PR head**, including
both JAX stacks, minimum dependencies, normalization, web smoke, and optional
MaxText/MaxDiffusion coverage. The older checkpoint's run does not satisfy this
gate. [PR #274 checks](https://github.com/enpasos/jax2onnx/pull/274/checks) and the
PR description record the final head and run once that gate passes. The PR stays
unmerged and no review replies are posted.
