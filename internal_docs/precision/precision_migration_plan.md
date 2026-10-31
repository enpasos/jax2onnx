# Computation precision migration plan

This is the accepted plan for the complete migration, updated 2026-10-31.
The result must be one ordinary source-preserving export contract across
existing functionality, including unchanged supplied and third-party JAX
sources. At the documented checkpoint, SOURCE is private and ordinary export
still uses LEGACY. The unmerged PR may adopt the new contract incrementally;
retaining the old public behavior until full qualification is not a separate
implementation requirement. Keep legacy paths only for concrete CI or meaningful
reference comparisons, with removal criteria. Complete capability, compatibility,
scientific validation and independent Astra acceptance remain merge requirements.

The [target design](precision_target_state.md) owns the semantic contract.
The [implementation work plan](precision_implementation_work_plan.md) owns the
complete B–F coverage and executable acceptance obligations. The
[current state](precision_current_state.md) distinguishes implemented capabilities
from open gates; the [handoff](implementation_handoff.md) retains detailed evidence.
The [user migration guide](precision_migration_plan_user_view.md) explains the
expected caller changes without implementation history.

## Objective and settled boundaries

Export must preserve the typed computation selected by JAX: input promotion
properties, model-local settings, constants, numeric conversions, bitcasts,
accumulation requirements and precision-relevant primitive parameters. ONNX
representation, semantic lowering, runtime execution and numerical acceptance
remain separate checks. A known failure is failed; missing evidence is
unverified; an unfinished implementation is migration work.

Precision belongs to the model and its preparation. The export's typed inputs
and active JAX configuration instantiate those choices during tracing. Export
must not silently widen FP32 arithmetic, lower required FP64 work to FP32, or
reinterpret an exact cast as a minimum-precision request. Entirely low-precision
models, selected wider regions and scientific FP64 computation are all required.

Supported workflows using fixed sources must remain available through external
configuration, preparation and caller adapters. This includes prior FP64
workflows and existing runtime controls. Where model configuration cannot
express an intentional change to a fixed numerical choice, explicit JAX-side
preparation must define the new computation and an independent reference before
export. No generic transformation API is assumed. A source rewrite or static
binding cannot substitute for a missing adapter or a previously dynamic control.

Use official supported dependencies with the existing lockfile cohorts and
markers. Custom JAX/jaxlib/XLA/LLVM builds, patches, overlays or substitutions
require separate explicit user approval and are currently prohibited. The
unapproved compiler experiment has been removed; its retained counterexample
belongs in the [dependency record](../../docs/about/dependencies.md).

## Next implementation work

Current work prioritizes correctness and affordable, complete validation:

1. Finish the active CI scheduling repair and qualify all STANDARD whole-job
   budgets without losing main-equivalent coverage. Preserve complete EXTRA,
   support, optional and browser ownership before activating replacement gates.
2. Retain the reviewed completed-export ownership repair and NNX early-rejection
   containment. Complete the broader state, RNG and transformed-scope contracts;
   rejection alone does not complete migration.
3. Resolve strict extrema and other numerical failures, keeping original feeds,
   zero signs, NaN behavior and downstream witnesses. The existing user decision
   requires official native roles to agree. Contradictions stay FAILED until
   they agree; changing that rule would require a separate explicit decision.
4. Complete independent references, criteria and actual artifact-bound execution
   for the current inventory. Proof-producer and accounting controls do not
   establish production numerical acceptance.
5. Continue the remaining operators, scopes, fixed models and artifact work in
   the phase matrix, then complete integrated acceptance and the public cutover.

The [CI gate](precision_ci_gate.md) permits reviewed routine focused checks and
separate exhaustive lanes. Every accepted obligation remains assigned, and all
must pass on the exact final commit. The named BF16 publication exception does
not waive its mandatory failed scientific gate or any new CI failure.

## Phase order and exit criteria

| Phase | Purpose | Exit criterion |
| --- | --- | --- |
| A: private integration | Exercise a bounded source contract through the real converter, facade and sample tools. | The stated subset has scoped evidence. This is an intermediate checkpoint, not the migration's completion scope. |
| B: inventory and references | Reconcile every existing registration, metadata case, manual regression and optional model; prepare original JAX references without legacy coercion. | Every obligation retains identity, contract, owner, executable selector and honest outcome; criteria and native references are reviewed before numerical acceptance. |
| C: values, scopes and interface | Preserve canonical values, constants, complex representations, contexts, functions, control flow, dynamic shapes, named inputs, layouts and artifacts. | Existing boundary and scope capabilities pass through the shared pipeline and serialized artifacts; temporary exclusions are removed or remain explicit blockers. |
| D: numerical and framework coverage | Complete all existing operator/parameter families and framework interception paths, including state/RNG/AD, FFT and linear algebra. | Source semantics and required arithmetic are independently established; genuine limits have evidence, and unimplemented cases are not relabeled as limits. |
| E: full models and deployment | Run original model/corpus inputs on the required official cohorts, minimum runtimes and browser targets. | Complete artifacts, independent references, state sequences, numerical criteria and resource gates pass at their declared scope. Passing operators alone cannot qualify a model. |
| F: integrated transition and acceptance | Complete the coherent ordinary export/validation contract, migrate callers/docs and reconcile the full inventory. Branch transitions may be implemented earlier. | No unresolved migration gaps or unexplained skips; all accepted gates pass on the exact final head and independent review accepts completion before merge. |

B starts alongside private integration. C and D can proceed in independently
reviewable groups after shared representation and helper contracts are agreed.
E begins as real model dependencies become available; it finishes only when the
complete required models and targets are accounted for. F depends on the whole
contract, not a promising subset. Implementation uses Sol Ultra; architecture
and independent acceptance use Astra Ultra.

## Preserve the shared pipeline

SOURCE must use the ordinary tracing, dispatch, IR-only builder, optimizer,
finalization and serialization path. Do not maintain a second product exporter.
Retain the [native builder rules](../../docs/developer_guide/advanced_topics/onnx_ir_builder.md)
and [subgraph contracts](../../docs/developer_guide/advanced_topics/subgraph_input_handling.md).

A JAX variable has one canonical typed value. Consumer adaptations create local
values and preserve other consumers and direct outputs. Operand roles differ:
comparison results are Boolean, indices remain integer, bitcasts reinterpret
storage, and accumulation has its own contract. Output dtype alone cannot prove
the arithmetic or hardware algorithm.

Function specialization must preserve precision-relevant inputs, captures and
call-site settings without retaining completed exports' models through cache
keys. Preserve legitimate live traces and ordinary/LEGACY/SOURCE cache isolation.
Do not fix ownership with unconditional global cache clearing. State preparation
must reject unsupported mutable static payloads before touching the original
model, or implement reviewed isolation with complete state handling.

Apply semantic checks through nested functions, branches, loops and transformed
bodies, including unexecuted branches that remain in the artifact. Required
casts, constants, shapes and precision attributes must survive optimization and
serialization. A checker-valid graph with a lost explicit attribute still fails
the source contract.

## Build independent numerical evidence

Reuse original metadata, factories, inputs and structural anchors, but do not
use legacy dtype rewriting as the source oracle. Keep both caller-x64 scenarios
and original model choices. Prepared FP64 or other intentionally transformed
scenarios have their own explicit references and criteria.

Check dtype, shape and output structure before numerical values. Integer and
Boolean outputs remain exact in mixed results. Floating acceptance uses reviewed
criteria appropriate to the operation and includes difficult finite values,
NaNs, signed zeros/infinities, subnormals and rounding boundaries where relevant.
Ordinary equality does not prove a zero sign; include explicit sign checks and
downstream compositions such as reciprocals. Valid all-zero outputs are allowed.
Bitwise checks are used where required, not as an indiscriminate NaN-payload rule.

Retain eager, fresh-JIT and warmed-JIT reference roles and both ORT optimization
modes where required. A configured denormal profile is distinct from default
execution. References are captured before runtime side effects can change their
floating-point environment. Failures in official JAX, converter lowering, ONNX
representation and ORT must retain separate diagnoses.

Missing references, absent criteria and skipped numerical execution cannot pass.
Green rejection/diagnostic tests do not establish numerical equivalence. Original
structural callback failures remain visible beside reviewed SOURCE counterparts;
a legacy fusion or node-count expectation must not become a permanent substitute
for source semantics.

## The final public transition

The normal call remains `to_onnx(fn, inputs=..., ...)`. Implement its coherent
transition on the branch when the actual change is ready for normal review and
relevant tests; unfinished unrelated migration work is not by itself a reason
to defer it. The [temporary legacy bridge assessment](precision_implementation_work_plan.md#temporary-legacy-bridges)
identifies current CI/reference dependencies and their removal criteria.
Preserve omission of the
old precision argument with an internal sentinel long enough to give an
actionable migration error:

| Call after transition | Required behavior |
| --- | --- |
| Legacy precision argument omitted | Preserve caller configuration, typed/weak input properties and source arithmetic; use FP32 for shape-only inputs. |
| Explicit `enable_double_precision=False` | Reject before tracing; explain removal of the flag and explicit JAX/model dtype configuration. |
| Explicit `enable_double_precision=True` | Reject before tracing; provide the FP64 configuration/preparation recipe and typed inputs. |

There is no permanent public mode selector or per-operation LEGACY fallback.
This is an intentional argument/default change, so repository callers, examples,
API docs, limitations, coverage and changelog must move together. Existing
capabilities remain required even when their correct new arithmetic differs
from an old artifact.

`input_params` becomes one whole-model runtime-input contract. Trace declared
values as actual typed root arguments and forward them through Function/If/Loop
boundaries. Bind static configuration through construction, closures or partial
application. Preserve defaults, literal overrides, unused-input behavior, both
Boolean values and every original supported control. Never attach synthetic
flags to an already specialized root.

Retain existing proto/IR/file return types and web packaging. The versioned
precision report records supplied, normalized, traced and exported properties,
settings, warnings and scoped checks. Export leaves runtime and numerical
acceptance unverified. The separate sample helper binds evidence to actual tensor
payloads, samples, references, criteria and runtime settings, and its guard
rejects failed, unverified or mismatched evidence.

Migrate `allclose`, browser validation and generated source tests under the same
contract. They must not cast away dtype mismatches, force legacy JAX settings,
or use CPU ONNX output as the browser's source oracle. Keep complex packing and
layout adapters explicit.

## Completion and rollback

At significant milestones, and when their assumptions change, run the complete
affected acceptance lanes. Before finishing, rerun the entire accepted suite on
the exact final commit across supported cohorts, including original full models,
minimum dependencies, browser execution, strict docs, typing, lint and pinned
hooks. Meet the [CI budgets](precision_ci_gate.md) and retain failed predecessors.
Starting a run or committing an intermediate result is not acceptance.

During development, legacy paths have only their documented temporary
CI/reference purpose. An earlier release or retained artifact provides a
comparison/rollback reference after transition; the new exporter does not mix
old and new precision rules. A reviewed branch API change can precede complete
migration, but no release, merge or completion claim follows from a partial
profile, plan update or green focused lane.
