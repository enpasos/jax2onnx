# Initializer Metadata Validation and Release Gate

This document describes enforcement of the
[public specification](../../user_guide/initializer_metadata_contract.md), the
maintained contract tests, and the checks that must continue to pass before
release. The guarantee rests on controlled transformations and independent
source records, not on finding a few surviving metadata strings.

## Original silent-loss regressions

The original prototype accepted annotations and silently lost them in the
following exports. Both cases now preserve their sources. Their original
outputs were valid ONNX: `onnx.checker` alone cannot detect metadata loss.

```python
import jax.numpy as jnp
from jax2onnx import to_onnx

weight = jnp.asarray(2.0, dtype=jnp.float32)

def metadata(source):
    return {"source": "weight"} if source is weight else None

model = to_onnx(
    lambda x: x + jnp.full((2,), weight),
    [(2,)],
    initializer_metadata=metadata,
)
```

`plugins/jax/numpy/full.py` previously created an unannotated tensor attribute
for `ConstantOfShape`, then the original initializer was removed. Annotated
fills now use a graph computation that consumes the original stored scalar.

The Function-only constant folding path in `plugins/jax/numpy/reshape.py`
previously created an unannotated constant for a source reshaped inside an
`@onnx_function` body. It now uses the registered
`inline_reshape_initializer` path.

```python
import jax.numpy as jnp
from jax2onnx import onnx_function, to_onnx

weight = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

def metadata(source):
    return {"source": "weight"} if source is weight else None

@onnx_function
class Inner:
    def __call__(self, x):
        return x + jnp.reshape(weight, (3, 2))

inner = Inner()
model = to_onnx(lambda x: inner(x), [(3, 2)], initializer_metadata=metadata)
```

`test_initializer_metadata_materialization.py` covers both cases, including
direct and nested Functions, exact signed-zero/NaN payloads, and artifact edits
that verify the source's actual consumers. Their expectation is successful
preservation; neither rejection nor a warning passes these tests.

## Enforcement design requirements

### Independent source records

`initializer_metadata_session` establishes one export-local registry, independent
of metadata being verified. `converter/initializer_metadata.py` records source
identities, capture occurrences, normalized snapshots, typed reference storage,
and source-to-representation/use associations. Strong references keep object
identities valid for the export's duration. Context-local state is restored on
success, failure, and a nested export initiated by a callback.

The same source captured in multiple scopes may have several representations.
Function-definition reuse must not erase distinct captures or invent a new
obligation merely for each runtime call. Callback consistency is checked for
every observed invocation of the same source, including skipped results.

Do not reconstruct provenance from values, shapes, generated names, user keys,
metadata equality, or matching hashes alone. Two independently captured arrays
can have identical bytes and properties and must still remain independent.

The registry must not be serialized into user properties, retained in exported
IR objects, or reused by a later conversion. Keep this internal; the contract
does not require a public provenance schema or parameter-discovery API.

### Controlled transformations

All paths that materialize captured data must participate in provenance
tracking. Copying a tensor, folding a constant, lifting a `Constant` into an
initializer, cloning a graph, and constructing a Function body are relevant
operations. A private identity marker on the first tensor is insufficient if
lowerings can silently manufacture replacements without registering them.

Register pure C-order reshapes with their source, dtype, element count and
payload relation. Retain other computations as graph operations consuming the
protected source, or reject them. Never manufacture lineage by finding an
equal-valued tensor after the fact. Coalescing its stored representation with
an independent unannotated or generated tensor, or materializing their combined
value as a constant, is prohibited. Runtime graph operations may consume both.

The `broadcast_in_dim` lowering uses the existing
`inline_reshape_initializer` helper for an annotated constant when static,
nonnegative shapes and a valid increasing axis mapping prove that the operation
only inserts singleton axes. The source tensor shape and source/result dtypes
must agree with that proof. A vector reshaped to `(1, N)` or `(N, 1, 1)` retains
its C-order payload and uses the existing reshape registration and final
validators. A broadcast that repeats elements retains graph operations
consuming the protected source. This lowering optimization does not add a
general constant-folding permission.

Every lowering and optimizer that can read/materialize constant payloads must
be audited or routed through a protected materialization interface. Add policy
checks where needed to prevent future paths from bypassing that interface.
The final validator cannot infer arbitrary lost dataflow from output bytes;
local transformation rules and recorded associations are required inputs to it.

The shared IR tensor readers and context constant evaluator suppress reads of
annotated sources and their derived values. The reshape helper explicitly reads
and registers an approved representation. Lowering guards check source
dependencies, including separately proved outputs of nested scopes. Syntactic
policy tests block common raw-read and clone-registration bypasses; they
supplement runtime checks and code review, rather than prove arbitrary Python
dataflow safe.

Unreviewed external lowering implementations are recorded during conversion.
If an annotation has been accepted, they fail before execution; if an annotation
is accepted later, that capture fails. Exports with no accepted annotations
retain normal extension behavior. A new verified lowering must use the guarded
interfaces and undergo the same review and tests as the shipped plugins.
User model callables lowered by the built-in Function plugin remain supported.

Graph assembly records output and node-use evidence before construction and
cloning. Clone registration verifies structural correspondence before copying
provenance. A final live-graph identity record rejects later unregistered
replacements. This closes the case where one correct tagged branch masks a
lost second branch, including a replacement before final model assembly.

Both optimizer profiles use `verified_rewrites.simplify_graph` for narrowly
proved substitutions: same-dtype casts, identity operations, static
identity/inverse reshapes, and allowlisted pure common subexpressions with the
exact same ordered input value objects and attributes. Ordinary exports run this
shared cleanup after the existing root and Function rewrite sweep, then remove
unused nodes. Omitted and all-skip callbacks use the same ordinary pipeline.
Annotated exports use the shared cleanup and unused-node removal, with the
additional registry checks below; other rewrites remain disabled until audited.

The shared rules never merge stored tensors by content. Unknown operations and
tensor-bearing nodes do not qualify. Identity and inverse-reshape rules cannot
remove dtype changes or bypass annotation-dependent reshape controls; exact
same-input CSE can share such computations while retaining their inputs. Public
graph outputs retain their names and Function signatures.

`replace_uses_with_verified_value` first verifies the existing live-topology
seal and independently repeats the local semantic proof. Before mutating the
graph, it projects the exact permitted substitution onto the existing topology,
including nested uses and graph outputs. After rewiring, the actual topology
must match that prediction. Only this proved delta advances the seal; accepting
whatever graph a general pass happens to produce is prohibited. A prior
corruption or an unrelated edge change during an approved rewrite must fail.
The final validators still verify storage, annotations, capture liveness, and
recorded source dependencies.

Further optimizations require explicit preservation evidence and tests before
joining the annotated profile. Maintained parity fixtures compare computation
structure with omitted, all-skip, and accepted callbacks; they do not establish
equal graph size for every model.

Function reuse also participates in source tracking. Candidate captured sources
are inspected before content-based reuse can discard their identities. Functions
with independent annotated sources receive independent definitions, including
nested pytree modules. Functions without accepted annotations retain ordinary
reuse behavior. Same-source reuse remains permitted.
Payload-free identity keys precede candidate binding; content fingerprints are
deferred until callbacks have observed the actual captures. Discarded unannotated
candidates restore the ordinary Function registry and naming state.

### Final validation

After all exporter-owned rewrites, validate every accepted capture and every
stored representative across the root graph, recursive subgraphs, local
Functions, and all tensor-valued attributes. Resolve function references and
value ownership; matching node or tensor names is insufficient.

Validate the public specification's two quantifiers: each accepted capture has
a live carrier, and every derived stored carrier has the correct entries,
reference storage, provenance, and uses. A live tagged direct use must not hide
an untagged folded copy used elsewhere. A detached tagged tensor is not proof
of preservation. Annotations cannot simply be moved onto another equal-valued
source to make the check pass.

Reachability is structural, including reachable calls and all bodies of
reachable control-flow nodes regardless of constant predicates or trip counts;
this checker is not a symbolic proof of numerical influence or differentiation.
Keep payload comparisons exact at the logical element level, normalizing byte
order and serialization layout, without treating NaNs or signed zeros as
interchangeable. Unsupported or unverifiable representation changes fail.

The public return boundary compares serialized storage and positional graph
topology against verified IR, including nested graphs and Functions. Tensor
payload bits and properties must agree; ordinary protobuf float-attribute
rounding is normalized. Duplicate tensor property keys are rejected before an
IR decoder could collapse them. File mode writes to staging storage, reloads
external data, validates, then publishes verified content. Callback and contract
failures leave existing destination files unchanged. Multi-file I/O crash
recovery remains outside this feature's promise.

`InitializerMetadataError` must escape the converter's ordinary permissive
optimizer/postprocessing failure handling. A logged exception followed by a
successful export violates IM-08. Keep the preservation check mandatory when
the ordinary `strict_optimizer_failures` policy is disabled.

## Contract-to-test matrix

Tests target the public API as well as responsible internal checks. Files named
below live under `tests/extra_tests/framework/`.

| Contract | Required tests and oracle | Maintained coverage |
| --- | --- | --- |
| IM-01 | Actual capture identity and callback-before-conversion; pre-capture copies stay distinct; Function cache fingerprints cannot preempt callbacks. | `test_initializer_metadata.py`, `test_initializer_metadata_contract.py`, Function identity tests |
| IM-02 | Copied UTF-8/empty values; invalid results/text and callback exceptions. | `test_initializer_metadata_contract.py`, existing callback validation tests |
| IM-03 | `{}`/`None` equivalence; inconsistent results in both directions; failed-then-clean and nested exports. | `test_initializer_metadata_contract.py`, `test_initializer_metadata_function_identity.py` |
| IM-04 | Every promised dtype, scalar/empty/dense storage, exact signed zeros/NaNs, annotated complex rejection. | `test_initializer_metadata_contract.py` |
| IM-05 | Registered reshapes/clones; separate equal-valued/equal-tag sources, unannotated/generated neighbors, independent edited outputs. | `test_initializer_metadata_materialization.py`, `test_initializer_metadata_function_identity.py`, `test_initializer_metadata_broadcast.py`, existing reshape/optimization tests |
| IM-06 | Complete scope inventory; swapped, replaced, merged, pruned, or orphan storage; one valid branch cannot mask another lost use. | `test_initializer_metadata_scopes.py`, `test_initializer_metadata_faults.py`, assembly/clone regressions |
| IM-07 | IR/proto/file, external-data reload, serializer corruption, duplicate keys, untouched destinations on failure. | `test_initializer_metadata_serialization.py`, existing return-mode tests |
| IM-08 | Exception reasons/context; permissive optimizer policy cannot bypass errors; diagnostics omit payloads/property values. | `test_initializer_metadata_faults.py`, callback and serialization failure tests |
| IM-09 | Omitted/None/empty equivalence; weak-reference lifetime checks; failed export cleanup; all-skip Function reuse. | `test_initializer_metadata_contract.py`, `test_initializer_metadata_function_identity.py` |
| IM-10 | Required positive cases, framework fixtures, deterministic generated cases; exceptions fail positive tests. | Scope/materialization suites, `test_initializer_metadata_generated.py`, existing NNX/Equinox regressions |

Existing maintained tests are under `tests/extra_tests/framework/`:
`test_initializer_metadata.py`, `test_initializer_reshape_metadata.py`,
`test_initializer_metadata_optimizations.py`,
`test_initializer_metadata_rope.py`, and `test_conv_transpose_bias_folding.py`.
`test_initializer_metadata_broadcast.py` covers stored singleton-only reshape
views, the graph-operation fallback for genuine expansion, exact payloads in
nested Functions, shared-source reuse, and independently editable source uses
with equal-valued annotated or unannotated neighbors.
`test_initializer_metadata_optimizer.py` covers the shared verified computation
rewrites, parity fixtures for ordinary and annotated exports, independent source
edits, nested scopes, and injected changes before or during an approved rewrite.
Unsupported rewrite candidates must remain unchanged.
`test_parameter_metadata.py` covers the separate model manifest and must not be
counted as proof of the callback's final provenance. In particular, its
external-data-reference test does not establish callback file-mode coverage.

## Mandatory successful fixtures

Retain explicit public-API success assertions for:

- Direct dense JAX captures and directly used NumPy captures. For NumPy, inspect
  the object actually supplied to the callback; do not require original-object
  identity before JAX canonicalization.
- Supported dtype/shape boundaries from IM-04 and pure C-order reshapes.
- Reused NNX `LinearGeneral`, with runtime parity and correct shared-source uses;
  Equinox Conv bias reshaping; and a Linen Dense fixture with source annotations.
- Scalar broadcast and `jnp.full`, including changed source values to ensure the
  tag is attached to the storage the computation actually consumes.
- Root, `jit`, nested ONNX Functions, Function-local reshape, and supported
  control-flow captures. Inspect a complete expected capture set instead of
  asserting merely that some annotation exists somewhere. Include constant
  predicates and zero-trip cases to enforce structural reachability consistently.
- Distinct sources with equal payloads and equal annotations, plus one accepted
  source next to an unannotated equal-valued capture or generated constant.

For independent-source fixtures, use a model with separate outputs for two
equal-valued sources. Edit only one final stored source and verify that only
its corresponding output changes, using a JAX reference with just that source
substituted. Exercise identical annotations as well as an unannotated neighbor.
This public-artifact oracle checks use association independently of the
exporter's registry, metadata strings, and tensor counts. These fixtures use
one representation per source; they do not imply an automatic update protocol
for arbitrary models with several representations.

Determine IM-10 success eligibility from the traced program's structural
output dependencies before exporter rewrites. A rewrite that drops a required
live source must not turn its positive test into an allowed error. Test an
already-disconnected accepted capture separately: that must fail under IM-06.

Use checker/runtime parity where the runtime supports the involved dtype. A
runtime kernel gap does not excuse missing metadata conformance tests for a
dtype whose export storage is promised. The callback feature does not promise
new inference kernels.

ONNX Runtime 1.24.1's optimizer fails an internal `RemoveNode` check on some
empty-tensor fixtures. Empty-input runtime tests therefore use
`ORT_DISABLE_ALL`; other cases retain default optimization. Exact storage and
full ONNX checking still run for every promised dtype, including BF16 where
ordinary NumPy runtime feeds are unavailable. See the
[runtime gap table](../../about/dependencies.md#onnx-runtime-kernel-gaps).

The existing RoPE fixture restores a known architecture and applies a caller's
AdamW mask. Keep that as integration evidence, without treating it as proof of
a general importer, source alias reconstruction, or arbitrary gradient parity.
`custom_jvp`/`custom_vjp` wrapper tests likewise establish forward/capture
behavior, not preservation of those derivative rules.

## Negative and generated coverage

Inject failures both at individual transformations and immediately before the
final validator. Required mutations include removing one key, changing a value,
swapping source tags, swapping payloads, changing dtype/count/order, merging
equal sources, fabricating an equal untracked replacement, retaining an orphan
tagged tensor, losing only one of several live representations, and deleting
the final accepted carrier. Tests must expect a contract exception and must not
allow logging followed by success.

Bounded, reproducible generated cases use a fixed seed set, ranks 0–3,
singleton/empty legal dimensions, contiguous/strided NumPy inputs, up to three
equal/distinct/shared sources, and up to two nested scopes. Compose approved
reshapes with graph operations. They exercise valid cases that must succeed and
single-fault mutations that must fail. Persist each discovered counterexample
as a small named regression. Large model demonstrations supplement these tests;
they cannot replace the invariant tests.

## Optional optimization measurements

Use representative framework models as an opt-in diagnostic alongside the
maintained regressions. Keep checkpoints, ONNX files, and measurement reports
outside git. Record the source revision, dependency versions, model
configuration, seeds, prepared parameter values, precision settings, input
shapes, opset, and runtime settings. Adapter examples should include nonzero
adapter scales so output parity exercises their contribution.

Compare exports without a callback, with an all-skip callback, and with accepted
annotations. For a controlled local diagnosis, a separate run that applies only
unused-node removal to an unannotated export can distinguish lowering overhead
from the ordinary optimizer sweep; do not change the production optimizer
policy to obtain that comparison. Count nodes across root and nested graphs
and Functions, and report payload bytes separately from serialized bytes and
tensor metadata bytes. Universal count or size thresholds are inappropriate
for different traces and dependency versions.

For each original saved artifact, run the full ONNX checker after save/reload,
compare runtime outputs with the prepared JAX model, and inventory actual
annotations across every tensor-bearing scope. Check the expected captured
source set and every stored representation's dtype and flat C-order payload,
including aliases and reshaped carriers. Checking only labels or counts cannot
detect merged independent parameters. Retain independent-source edit tests as
described above, including equal-valued sources with identical annotations and
an unannotated equal-valued neighbor.

Measure runtime optimization separately, with the optimization level and
runtime version recorded. If ONNX Runtime writes an optimized model, compare
its node counts, numerical outputs, metadata, and source associations against
the original artifact rather than assuming inference parity proves metadata
preservation. Runtime serialization and optimization fall outside IM-07;
retention or loss in a particular runtime version does not redefine the public
contract. The original serialized jax2onnx export remains the authoritative
annotated artifact.

## Release decision

All items below are required for GO:

- Every public requirement has implementation enforcement and positive or
  negative tests as applicable. All IM-10 success cases pass.
- Both confirmed silent-loss regressions are fixed and covered. No contract
  counterexample is waived as an undocumented model-specific limitation.
- The complete final check is mandatory, covers every promised scope and return
  mode, and cannot be bypassed by ordinary optimizer error policy.
- Focused contract tests pass on both locked JAX stacks and the declared minimum
  dependency stack. Full relevant repository checks and the strict docs build
  pass. Existing CI results from before these tests were added are not evidence
  for this gate.
- User/API documentation and the actual public exception/behavior agree. Remove
  release claims if a regression breaks this specification.

The standard CI jobs run these maintained tests as part of the full suite. The
minimum-dependency job explicitly includes the initializer metadata suites and
their reshape/ConvTranspose regression coverage. A local locked-stack pass does
not replace that minimum-version gate.

Test completion is evidence for the enforced invariant; a large green test
count alone is not the invariant. Changes to tracing, lowering, optimization,
serialization, supported dtypes, or dependencies must continue to run this
suite. A narrower future contract requires an explicit documented API decision,
not a new silent fallback.
