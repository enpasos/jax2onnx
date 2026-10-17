# Precision Contract Experiment

Status: internal integration proof of concept, based on `main` at `fae17822`
(2026-10-17). This is a runnable discussion aid for the
[target design](../precision_target_state.md), with a deliberately small scope.
The public exporter and its legacy precision behavior are unchanged.

## Run it

From the repository root, using the project's installed Poetry environment:

```sh
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run python -m internal_docs.precision.poc.demo

JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run pytest -q internal_docs/precision/poc/test_source_export.py
```

The demo prints the environment, graph operator/result types, actual results,
and expected failures. It checks its examples against JAX and writes no model
files. The tests also cover constants, shared values, configuration isolation,
and the experiment's rejection boundaries. These internal tests are invoked
explicitly; the default repository suite collects `tests/`.

## The question it answers

Can the existing lowering pipeline preserve the dtypes JAX actually traced,
including mixed FP64/FP32 and FP16/FP32 computation, when legacy promotion is
removed from the selected path?

For the cases below, yes. The experiment also shows why this is a numerical
contract decision rather than merely changing ONNX type labels.

```python
def exact_region(x):
    y = x.astype(jnp.float32)
    return (y + 1) - y
```

With FP64 input `100_000_000`, JAX performs the arithmetic in FP32 and returns
`0`. The merged legacy double export preserves the initial FP32 conversion,
widens subsequent arithmetic, and returns `1` in FP64. The experiment returns
`0` in FP32, matching JAX for this case. The lower result is the intended
consequence of preserving that model's arithmetic, not a request to maximize
accuracy by choosing a different calculation.

To request a local minimum instead, the model can choose
`jnp.promote_types(x.dtype, jnp.float32)`. The same expression then keeps FP64
for FP64 input, while selecting FP32 for FP16 input. The export call supplies
typed input specifications; the rule itself belongs to JAX code.

## Observed examples

The demo was run on Python 3.12.9, JAX 0.11.2, ONNX 1.23.2, ONNX IR 1.0.0, and
ONNX Runtime 1.31.0. Execution used `CPUExecutionProvider`, one thread, and
`ORT_DISABLE_ALL`. JAX x64 was enabled by the demo before tracing; the exporter
does not change it. All successful demo samples passed the full ONNX checker
and matched the JAX reference's result types and values.

All 26 focused tests also passed on both repository-supported JAX stacks:

| Python | JAX / JAXLIB | ONNX | ONNX Runtime | Result |
| --- | --- | --- | --- | --- |
| 3.12.9 | 0.11.2 | 1.23.2 | 1.31.0 | 26 passed |
| 3.11.12 | 0.10.2 | 1.22.0 | 1.29.0 | 26 passed |

Both used ONNX IR 1.0.0. This includes expected rejection tests and BF16
structural tests; it does not mean 26 different runtime capabilities were
established.

The unchanged public-export regression suite also passed on the Python 3.12
stack: **4460 passed**, using CPU execution and four pytest workers. The
repository's lint, formatting, typing, builder-policy checks, and strict MkDocs
build passed. Internal Markdown links were checked separately because this
folder is not part of the published documentation build.

| Model choice | Experimental result | What it makes visible |
| --- | --- | --- |
| Exact FP32 arithmetic after FP64 input `1e8` | `0`, FP32; legacy double export gives `1`, FP64 | The source rounding and arithmetic boundaries matter. |
| Minimum FP32 arithmetic after FP64 input `1e8` | `1`, FP64 | JAX code can deliberately retain higher precision. |
| The same minimum rule after FP16 input `2048` | `1`, FP32 | Typed inputs instantiate the model's rule. |
| FP64 arithmetic plus a shared FP32 cast and its negation | FP64, FP32, FP32 outputs | Higher and lower precision coexist; one consumer does not rewrite another. |
| FP16 input `[2048, 1]`, selected FP32 addition, cast back | `[2048, 2]` FP16; `[2049, 2]` FP32; `[2048, 2]` FP16 | Model code chooses where extra precision and subsequent rounding occur. |
| FP32 values `[1e8, 1, -1e8]` explicitly converted for an FP64 reduction, then returned as FP32 | `1`, FP32, with DOUBLE ReduceSum | A wider intermediate calculation can have a lower-precision interface. |
| Shared FP32 value used directly, negated, and bitcast to UINT32 | All three agree with JAX at opset 26, including signed-zero bits | Representation-sensitive uses retain the original FP32 binding. |

Exact equality on these chosen samples is a regression criterion, not a
cross-runtime bitwise-equality promise. The reduction example does not establish
all reduction orders or accumulator implementations. FP16 graph types and these
results do not prove native FP16 hardware execution or a speed advantage.

## Where it stops

The experiment distinguishes an ONNX representation limit from a limit of its
own implementation:

- A same-width FP32-to-UINT32 bitcast is rejected at opset 23 with a diagnostic
  naming the equation, types, and required opset. An explicit selection of
  opset 26 succeeds on the tested CPU runtime. The experiment never raises the
  opset automatically. See the official
  [BitCast schema](https://onnx.ai/onnx/operators/onnx__BitCast.html).
- A different-width bitcast needs another representation; this experiment
  rejects it rather than inserting a numeric cast.
- An explicit FP64 input specification with JAX x64 disabled fails before
  tracing. Already-truncated arrays cannot reveal a former FP64 intention.
- Functions, control flow, JIT call regions, symbolic shapes, effects, and
  operations outside the small allowlist raise `PrototypeLimit`. That says
  nothing about whether ONNX or a runtime could support them.

The allowlist is `add`, `sub`, `mul`, `neg`, `convert_element_type`, `reduce_sum`,
and `bitcast_convert_type`, at opsets 23 or 26. Inputs must be explicitly typed
static floating tensors; UINT32 is included for bitcast results. BF16 has
structural test coverage only: BF16 runtime execution and numerical acceptance
remain unverified here. This list is an exploration boundary, not a product
support matrix for every composition, shape, or attribute combination.

The harness establishes schema validity and checks selected semantic examples.
The demo and tests establish CPU execution and numerical agreement only for
their recorded cases. It has no runtime capability database, deployment target
API, automatic fallback selection, latency benchmark, or memory benchmark.
A failed CPU kernel load would be a runtime failure; passing the ONNX checker
alone would not establish deployability.

## How it connects to the implementation

[source_export.py](source_export.py) uses the actual internal tracing helper,
existing primitive plugins and dispatch ownership checks, IR optimizer,
post-processing, and ONNX serialization. It does not construct hand-written
expected graphs or replace module functions to make a test pass.

A local `IRContext` subclass changes the type of a writable output placeholder
to its traced dtype. It first uses the existing allocator to preserve ownership
rules. Typed inputs and constants use the existing non-promoting bindings;
graph outputs refer to produced values. Post-processing runs with constant
promotion disabled. The selected plugin paths receive the legacy flag `False`
only to suppress promotion; this is not a general implementation of the target
contract through that Boolean.

The public entry point is bypassed because it also overrides JAX x64. The
experiment requires typed inputs and leaves the caller's tracing configuration
in place. It also bypasses public input normalization, names/layout options,
parameter materialization, and file-return handling. Those surfaces still need
integration work. Concurrent export and arbitrary tracing side effects have
not been validated by this experiment.

Child contexts currently construct the production `IRContext` directly; they
would not inherit this subclass's allocation rule. Functions and control flow
are therefore rejected before lowering. A production path needs an explicit
internal policy threaded through all scopes, constants, and finalization, with
an audit of plugins that consult the existing Boolean directly.

## Recommended next step

Use these examples to agree on observable behavior, especially the `0` versus
`1` result and mixed output types. Then replace the local subclass with a
private, explicit conversion policy and validate it first through one nested
function and one control-flow boundary. Retain legacy regressions beside it.
Only widen the supported subset after its source semantics, schema validity,
and claimed runtime behavior have evidence.

The experiment begins [migration Stage 3](../precision_migration_plan.md#stage-3-build-an-internal-path-that-preserves-source-dtypes);
it does not complete that stage or determine the eventual public argument name.
