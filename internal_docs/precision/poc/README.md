# Precision Contract Experiments

Status: bounded private integration, 2026-10-18, beyond `main` at `fae17822`.
The converter carries SOURCE policy through selected real lowerings and scopes.
Public options and legacy defaults retain their meanings; no public source mode
is available. These experiments inform the
[target design](../precision_target_state.md) and
[migration plan](../precision_migration_plan.md).

| Experiment | Question | Evidence |
| --- | --- | --- |
| [Source entry point](source_export.py), [demo](demo.py) | Can real lowerings preserve the traced calculation? | [Integration findings](source_integration_findings.md): inputs, scopes, call-time settings, cache isolation and current tests. |
| [Numerical contract](numerical_contract.py) | Do result types, wider sums and intermediate rounding survive execution? | [Numerical findings](numerical_contract_findings.md): actual exports, independent controls and failed runtime acceptance. |
| [Migration calls](migration_ux.py) | Can callers predict legacy/source differences and diagnostics? | [UX findings](migration_ux_findings.md): input defaults, tracing configuration, weak inputs and option conflicts. |

## Run it

From the repository root in the installed Poetry environment:

```sh
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run python -m internal_docs.precision.poc.demo

JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run python -m internal_docs.precision.poc.numerical_contract

JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run python -m internal_docs.precision.poc.migration_ux

JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run pytest -q tests/extra_tests/converter/test_source_*.py \
    internal_docs/precision/poc
```

The demos keep models in memory. Source and UX implementations now live in
[private test support](../../../tests/extra_tests/converter/precision_test_support.py)
and [UX test support](../../../tests/extra_tests/converter/precision_ux_support.py);
the old module paths remain runnable compatibility entry points. The durable
source regressions live in `tests/extra_tests/converter` and run in normal CI.
Supplemental experiments in this folder still need an explicit test invocation.
See [recorded validation](source_integration_findings.md#recorded-validation)
for suite counts, versions and actual results.

## Observable source behavior

For exact FP32 work after FP64 input `1e8`:

```python
def exact_region(x):
    y = x.astype(jnp.float32)
    return (y + 1) - y
```

JAX and the source experiment return FP32 `0`. Legacy double export preserves
the initial conversion but widens subsequent arithmetic and returns FP64 `1`.
The source contract preserves that model's chosen computation. To choose a
local minimum instead, model code can use `jnp.promote_types(x.dtype, jnp.float32)`;
typed export inputs instantiate that rule.

| Model choice | Current experimental evidence |
| --- | --- |
| Mixed FP64 calculation and shared rounded FP32 value | Direct mixed outputs, arithmetic and same-width bitcast retain the intended types and selected values. |
| FP16 work with selected FP32 arithmetic and cast-back | Chosen input, intermediate and output boundaries are retained on the tested samples. |
| Weak scalar plus explicitly FP16 one | Weak input `2048` returns FP16 `2048`; the strong input with the same shape/dtype returns FP32/FP64 `2049`. |
| Local x64 or matrix-precision setting around a named function | Captured settings survive body retracing and distinguish specialization. Cached source traces cannot leak into ordinary/legacy calls. |
| FP16 matrix inputs with preferred FP32 result | Bounded default-precision matrix lowering preserves FP16 inputs and returns FP32 `2049`. |
| FP32 squares followed by selected FP64 variance work | Actual source exports preserve exact variance `8388609` inline, in a Function and in both conditional branches. |
| FP16 square then sum | The graph is valid and executes, but the tested ORT targets miss the required rounding. Sample acceptance rejects it. |

The [review findings](source_integration_findings.md#review-findings) explain
three repaired defects: lost local tracing context, cached branch leakage
between policies, and legacy algebraic fusion changing source arithmetic.
Source now retains separate operations where those fusions lacked numerical
justification. The known FP16 runtime failure persists without converter fusion.

The [follow-up review](source_integration_findings.md#follow-up-review-on-2026-10-18)
closed two additional gaps: the shared optimizer now honors SOURCE eligibility,
and function specialization includes weak metadata in captured state and static
arguments. Both fixes have regular-suite regression coverage on both stacks.
The new exact FP32 reciprocal sample still differs under ORT optimization;
that profile fails its exact sample acceptance even with the correct source
graph. See the recorded numerical evidence before choosing a runtime profile.

## Scope and limitations

The private harness uses the actual tracing, IR context, plugins, optimizer,
post-processing and serialization. It checks static floating inputs and a small
primitive set at opsets 23/26, including selected Function/If cases. It rejects
loops, scan, JIT regions, effects, symbolic shapes and unaudited operations.
Bodies are validated recursively. Same-width BitCast requires explicit opset26;
different-width bitcasts need a different representation and are rejected.
No opset is raised automatically. These are experiment limits, not universal
ONNX support classifications.

Matrix support is limited to positive static rank-2/rank-3 standard matrix axes,
FP16 operands, an explicitly preferred FP32 result and default precision.
Explicit algorithms and other combinations are rejected. The wider computation
has no measured latency/memory advantage and does not promise a native
low-precision hardware algorithm. Numerical coverage is for selected samples.

BF16 structural validity does not establish runtime kernels. Function coverage
does not establish all nested combinations or every JAX setting. Names/layouts,
parameter materialization, other return modes and concurrent export remain
outside the harness. The normal public pipeline still uses its legacy policy.
The [next implementation work](../precision_migration_plan.md#next-implementation-work)
addresses a finite initial subset and shared export-boundary integration.

## Numerical acceptance is a separate result

The numerical demo reports actual source exports separately from changed-source
candidates and independent hand-built graphs. A candidate passing one example
is not an automatic exporter repair. In particular, wider arithmetic with an
explicit FP16 intermediate cast passes the square/sum sample, while the original
composition fails on both recorded CPU runtimes and both optimization settings.

The [private CPU sample gate](../../../tests/extra_tests/converter/precision_runtime_support.py)
checks a supplied artifact, input/reference sample and execution profile.
`require_sample_acceptance` rejects failed and unverified results. The demo
shows this rejection for the original FP16 sample. Kernel execution failure is
distinct from numerical failure. An accepted sample does not certify all inputs
or every composition of allowed primitives.

The calling experiment uses `precision="source"` only as a private spelling.
Its supplied/normalized/traced properties, actual warnings and legacy comparison
remain useful before any public interface is finalized. Neither it nor the
CPU sample gate is a general deployment capability engine. Public availability
requires the bounded acceptance gates in the migration plan.
