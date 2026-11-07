# Initializer Metadata Contract

This page defines the public contract for
`to_onnx(..., initializer_metadata=callback)`, introduced in 0.19.0.
The exporter enforces it through capture records, guarded lowering and graph
assembly, and final artifact validation. The maintained tests and release
requirements are described in
[Validation and release gate](../developer_guide/advanced_topics/initializer_metadata_validation.md).

**A successful export preserves every accepted annotation on the correct
exported source representations. Otherwise, the export raises an error.**

The feature has a deliberately bounded input: constants exposed to jax2onnx by
JAX tracing. It does not promise to discover every parameter of a Python model.
Within that boundary, preservation is mandatory; it is not conditional on
whether a particular model happens to be covered by an example test.

The numbered requirements below are normative. “Must” identifies a requirement
for implementation and tests. They apply under the project's supported
dependency versions and export settings. They add obligations to the ordinary
export contract; they do not make unsupported JAX operations exportable.

## IM-01 — Source and capture boundary

A **source** is the actual constant object that JAX supplies to jax2onnx when
the exporter binds a traced constant. A **capture** is a binding of that source
in a particular graph or function body. Multiple captures can refer to the
same source object.

The exporter must call the callback for each capture before converting that
object to NumPy. This applies to root graphs, nested graphs, and local ONNX
Functions. Ordinary model inputs, JAX literals, and exporter-generated helper
constants are not captures for this API.

Source identity is object identity within one export. Array contents,
`source_path`, other user properties, and generated ONNX names must not be used
to infer identity. Different captured objects remain different sources even
when their bytes and annotations are identical.

JAX may replace or transform objects before exposing them to the exporter.
In particular, a captured NumPy object need not be the original NumPy array.
The callback receives the object at this boundary, not a reconstructed Python
model leaf. Objects never captured create no annotation obligation.

## IM-02 — Callback input, result, and acceptance

The caller must keep source values stable during export. The callback must not
mutate them or the model, and must return a stable result for a given source.

The callback returns either:

| Result | Meaning |
| --- | --- |
| `None` or an empty mapping | Do not annotate this source. These results are equivalent. |
| A nonempty `Mapping[str, str]` | Accept these entries as an annotation obligation. |

Keys must be nonempty strings. Values may be empty strings. Both must be
encodable as UTF-8. The exporter must copy the mapping when processing the
callback result. Subsequent changes to the caller's mapping must not alter the
accepted snapshot. Keys and values are opaque and must retain their exact text;
there is no interpretation of `trainable`, `source_path`, or application prefixes.

A noncallable argument, a non-mapping result, or a non-string key/value raises
`TypeError`. Empty keys and unencodable text raise `ValueError`. No annotation
is accepted from an invalid result. Exceptions raised by the callback itself
must propagate without conversion into a warning or a successful export.

## IM-03 — Repeated calls and export isolation

Callback order and total invocation count are unspecified. A source may be
captured in several scopes, and the exporter may process a capture more than
once. Every observed result for the same source must have the same normalized
snapshot: `None` and `{}` normalize to “skip”; all other results normalize to
their copied entries.

If observed results disagree, including skip-to-annotation or annotation-to-skip
changes, the exporter must raise `InitializerMetadataError` with reason
`inconsistent_callback`. It need not call a callback an extra time to test
determinism.

Snapshots and source tracking belong to one export. A later export, including
one following a failed export, must not inherit the earlier callback, snapshots,
or source records. No internal tracking identifiers may leak into user metadata.

## IM-04 — Reference storage

For an accepted source, the exporter must record its reference tensor at the
capture boundary: its JAX-traced dtype, shape, and element payload after normal
typed constant binding, before plugin folding or optimization. The reference
is not a hypothetical representation from before JAX tracing.

This first contract supports dense tensors with traced dtype `bool`, `int8`,
`uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`, `float16`,
`bfloat16`, `float32`, or `float64`. Normal JAX precision requirements still
apply. Rank-zero and zero-element tensors are included.

A nonempty annotation on another representation, such as packed complex,
quantized sub-byte, sparse, or opaque storage, must raise
`InitializerMetadataError` with reason `unsupported_representation`. An
unannotated value remains subject to the ordinary exporter behavior.

An allowed stored representation has the same dtype and the same logical
element bit patterns in C-order as its reference. Its shape may differ only
by a reshape with unchanged element count and element order. Signed zero and
NaN payload bits must not be silently replaced. Physical protobuf encoding
and external-data placement may differ; the decoded tensor must satisfy these
requirements.

## IM-05 — Stored representations and transformations

A **stored representation** is a dense initializer or a tensor-valued node
attribute (`TENSOR`, or an element of `TENSORS`). This includes `Constant.value`
and, for example, `ConstantOfShape.value`. These locations can occur in any
root graph, nested graph, or local Function.

Every stored representation of an accepted source must carry that source's
complete accepted entries in `TensorProto.metadata_props`, or the corresponding
IR tensor properties. Unrelated metadata may coexist, but must not replace an
accepted entry. Runtime results of graph operations do not need copies of
source annotations.

The exporter may retain an accepted source and represent casts, transposes,
broadcasts, and other computations as graph operations. It may fold a pure
reshape into stored data if IM-04 remains true. Any other materialization of
data derived from an accepted source must remain a graph computation or cause
a preservation error. Copying the label onto transformed numerical data is not
an allowed substitute.

Every representation and its uses must retain a tracked association with the
correct source and capture. An accepted source must not share a stored
representation with an independent captured or generated source. This includes
unannotated sources with equal contents. Reusing storage for the same source
is permitted; one global copy per model is not promised. Multiple shapes or
scopes may require multiple representations.

## IM-06 — Complete, live preservation

A successful export must satisfy both requirements:

1. Every accepted capture has at least one live, valid stored representation.
2. Every live stored representation derived from an accepted capture satisfies
   IM-04 and IM-05, and remains connected to that capture's corresponding uses.

“Live” means structurally reachable through dependencies of a model output,
including the relevant reachable function calls and control-flow bodies.
Traverse the bodies of a reachable control-flow node regardless of a constant
predicate or trip count; do not try to prove branch feasibility or execution.
Keeping a detached or unused annotated tensor does not satisfy the obligation.
Structural reachability does not assert a nonzero gradient or numerical
influence for every possible input.

Loss of the last live representation after acceptance is a preservation error,
including removal during dead-code elimination. There is no silent “unused”
exception after acceptance in this first contract. Objects removed by JAX
before capture remain outside IM-01.

Annotations swapped between sources, annotations on fabricated replacements,
or a tagged survivor masking an untagged derived copy must also fail. Merely
finding the expected strings or counting annotated tensors is insufficient.

## IM-07 — Return modes and serialization

The same obligations apply to `return_mode="ir"`, `"proto"`, and `"file"`.
Validation must cover the final result after all exporter-owned transformations.

- `ir`: the returned IR model must satisfy the contract.
- `proto`: the returned protobuf model must satisfy it after serialization.
- `file`: loading the returned path, including external data when used, must
  produce a model that satisfies it. Tensor properties stay in the ONNX model
  when tensor payloads reside in a data file.

Callback, metadata-validation, and preservation failures must be detected before
publishing destination files. Existing destination files must remain unchanged
for those failures. Ordinary filesystem/serialization exceptions still fail
the call. Atomic publication of several files across I/O failures or process
termination is not promised by this feature.

External graph optimizers, manual edits, later mutation of returned objects,
and third-party viewers are outside this boundary. A consumer that serializes
an unmodified returned model with the supported ONNX/IR serializers must retain
the properties; that round trip is part of conformance testing. Netron's display
behavior and its optional JSON attachment are separate from this contract.

## IM-08 — Detectable contract failures

`InitializerMetadataError` is the required public exception, a subclass of
`ValueError` exported from `jax2onnx`. It exposes a stable `reason` string:

| Reason | Meaning |
| --- | --- |
| `inconsistent_callback` | Observed snapshots for one source disagree. |
| `unsupported_representation` | An annotated source cannot use IM-04 storage. |
| `preservation_failed` | Final entries, provenance, representation, liveness, or use association cannot be verified. |

The error must identify the failed obligation, export stage, and relevant
scope/source using exporter-generated diagnostic handles where available.
Exact message wording and handle values are not stable API. Diagnostics must
not dump array payloads or arbitrary callback property values. Caller-raised
exceptions retain the caller's own message.

Missing evidence must fail, rather than mean “probably preserved”. There is no
warning-only or best-effort mode for nonempty accepted annotations. Permissive
handling of ordinary optimizer failures must not swallow these errors or allow
an unchecked return. I/O errors and failures unrelated to metadata retain their
ordinary exception types; no failure path returns a purportedly successful model.

The current verified profile uses the lowering implementations shipped with
jax2onnx. If an export accepts annotations and also uses an unreviewed external
lowering, it raises `preservation_failed`; the same external lowering remains
usable when no annotation is accepted. User model functions and
`@onnx_function` still use the shipped lowering machinery. This restriction
prevents an extension from silently bypassing the materialization checks.

## IM-09 — Disabled and empty behavior

Omitting the argument and passing `initializer_metadata=None` must be
equivalent. A callback whose observed results are all `None` or empty mappings
must leave the graph, tensor payloads, types, and metadata equivalent to the
ordinary export with the same configuration. Callback execution and validation
still occur. There must be no persistent bookkeeping metadata or cross-export
state introduced by this case.

## IM-10 — Required successful behavior

Preserve-or-fail must not be implemented by rejecting every annotation. The
following valid cases are required to succeed when the ordinary exporter
supports the involved operations and the caller meets IM-01 through IM-04:

| Required case | Required result |
| --- | --- |
| Direct captured dense tensors of the listed dtypes, including scalar and empty tensors | Exact entries and reference storage survive. |
| Different sources with identical payloads, including identical annotations | Separate source associations survive. |
| One source reused within a graph | Its uses remain associated with that source. |
| Pure C-order reshapes | The reshaped representation satisfies IM-04 and retains its entries. |
| A captured scalar used in `jnp.full`, or an annotated broadcast | The scalar remains correctly represented and connected to the fill/broadcast. |
| Captures in `jit`, nested Functions, and supported `cond`, `scan`, `while`, and `fori_loop` bodies | Each accepted capture is preserved in the relevant scope. |

For this success requirement, each accepted capture must participate in the
structural output dependencies of the traced program at capture time, using
IM-06's reachability definition. Exporter rewrites must retain the source
storage and dataflow needed to satisfy these cases. Removing that dataflow
does not turn a required success into an allowed rejection. Captures already
disconnected from those dependencies when accepted must instead cause an
IM-06 preservation error.

These requirements are exercised through direct JAX code and maintained
Flax NNX, Linen, and Equinox integration fixtures. Success remains mandatory
for these cases; an integrity exception is a test failure, not a passing
alternative. Other exportable programs may be rejected if preservation cannot
be established. That rejection must follow IM-08.

## Separate APIs and non-goals

`annotate_parameters` and `read_parameter_metadata` describe final root-graph
names through the separate `pkg.jax2onnx.parameter_metadata` model manifest.
They neither satisfy nor replace this callback contract and must not infer
their records from opaque tensor properties. Their existing
[annotation contract](parameter_metadata.md#annotation-contract) remains separate.

This feature does not infer parameter inventories, framework paths, freeze
policies, or the meaning of arbitrary metadata keys. It does not guarantee
gradient preservation, optimizer state, training support, a general ONNX-to-JAX
importer, or automatic synchronization of multiple exported representations
after a consumer edits weights. A source path alone does not encode an alias
or update protocol.

The [validation specification](../developer_guide/advanced_topics/initializer_metadata_validation.md)
defines the evidence required to release this contract.
