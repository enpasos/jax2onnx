# Migration Calling Experiment

Status: internal UX candidate, 2026-10-17, updated for the
[private source-policy integration](source_integration_findings.md). Private
converter code has changed; the public API and legacy defaults are unchanged.
[`migration_ux.py`](migration_ux.py) compares the actual public
`to_onnx` entry point with the existing bounded
[`export_source_dtypes`](source_export.py) experiment. The candidate spelling
`precision="source"` is an experimental spelling. The shape-only FP32 fallback
implements the default proposed in the target documents; this experiment does
not settle the public interface or release that behavior.

```sh
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run python -m internal_docs.precision.poc.migration_ux

JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run pytest -q internal_docs/precision/poc/test_migration_ux.py \
    tests/extra_tests/converter/test_source_inputs.py \
    tests/extra_tests/converter/test_source_function_context.py
```

The demo delegates to the shared
[UX test helper](../../../tests/extra_tests/converter/precision_ux_support.py)
and writes no model artifacts. Its export comparison reports the supplied
input kinds/dtypes and weak flags, source-normalized inputs and weak flags,
ambient/tracing/restored JAX x64, actual traced input types and weak flags,
traced output types, final ONNX boundary types, and captured
warnings. Observation wraps the callable used by the real exporter; it does not
perform an extra trace. Python scalar outputs use JAX's effective dtype rules.

## Candidate call behavior

| Call | Route and configuration |
| --- | --- |
| No precision argument | Call public `to_onnx` with the old argument omitted; its existing default temporarily disables JAX x64. |
| `enable_double_precision=False` | Call public `to_onnx` with explicit `False`; retain the same legacy behavior and record that the argument was explicit. |
| `enable_double_precision=True` | Call public `to_onnx` with explicit `True`; retain its x64 override and legacy promotion policy. |
| `precision="source"` | Normalize supported inputs to typed specifications and call the real bounded source exporter with the caller's JAX configuration. |
| `precision="source"` plus either explicit legacy Boolean | Reject before input normalization or model tracing; advise omitting the legacy argument. |

A sentinel distinguishes the first two calls. The candidate retains their
current common behavior while making that distinction available for discussion
of a future migration. Unknown selections and non-Boolean legacy values also
fail before tracing. No generic export policy engine is introduced.

For the source route, shape tuples/lists receive FP32 in both ambient x64
settings. `jax.ShapeDtypeStruct`, NumPy arrays, and JAX arrays keep their observed
floating dtype. Static, nonnegative integer dimensions are required. Concrete
array contents are not baked into the model; they supply tracing properties.
Weak JAX arrays and typed specifications retain `weak_type`; NumPy arrays are
strong, and shape-only source inputs normalize to strong FP32 specifications.
Legacy inputs are passed unchanged to the public entry point, which owns its
own normalization.

Both the wrapper and shared tracing helper now preserve `weak_type` when the
private source policy is selected. Legacy normalization retains its observed
compatibility rules. The earlier source defect and its correction are described
below; ONNX receives the concrete calculation JAX selects, without a weak tensor
type in the exported graph.

An explicit FP64 specification or existing FP64 array with disabled x64 fails
before the callable runs. The message explains how to configure JAX before
preparing FP64 inputs and exporting, or deliberately choose FP32. An array
already truncated by JAX to FP32 supplies FP32; the exporter cannot reconstruct
its former requested dtype.

## Observations

For the model's `y = x.astype(jnp.float32); return (y + 1) - y` and FP64 input
`1e8`, both routes observe an FP64 input and an FP32 traced output. The old
`True` route exports a DOUBLE output and returns `1` in FP64. The source route
exports a FLOAT output and returns `0` in FP32. Separate CPU execution uses
`CPUExecutionProvider`, one thread, and `ORT_DISABLE_ALL`.

The same dtype-dependent minimum-FP32 model receives different typed inputs:
FP16 input `2048` selects FP32 work; FP64 input `1e8` retains FP64 work. Both
samples return `1` in the selected type. This rule belongs to the model. The
typed export inputs instantiate it, and the selected opset constrains ONNX
representation. The facade does not choose runtime providers or math settings.

With an FP32 input and disabled x64, a model's `.astype(jnp.float64)` is
truncated by JAX to FP32. Export succeeds for that actual trace and the report
includes the original JAX warning. This differs from an explicit FP64 input,
whose contradiction is detectable before tracing. With ambient x64 enabled,
source tracing also retains JAX's FP64 default for a separately returned
`jnp.asarray(1.0)` while the shape-only input remains FP32; legacy omission
temporarily changes that default to FP32.

Input, option, and JAX configuration failures are separate categories. Model
tracing failures retain their original cause. The existing source exporter's
prototype-scope and ONNX-representation limits retain distinct categories;
other export exceptions remain generic export failures. These categories do
not infer runtime kernel support or numerical acceptance. Runtime execution
in the demo is separate and runtime exceptions are not recast as export errors.

## Weak input properties preserved on the source path

A review probe originally found that input normalization changed JAX's type
promotion before lowering. A weakly typed scalar can combine with an explicitly
typed operand differently from a strong input with the same shape and dtype.
The source integration now preserves that distinction:

```python
import jax.numpy as jnp
from internal_docs.precision.poc.migration_ux import export_candidate


def add_fp16_one(x):
    return x + jnp.asarray(1.0, dtype=jnp.float16)


x = jnp.asarray(2048.0)  # weak_type=True
jax_result = add_fp16_one(x)
candidate = export_candidate(add_fp16_one, [x], precision="source")
```

The regression recreates weak and strong inputs under each ambient x64 setting
and runs the serialized model with the stored input dtype. On both installed
stacks below, CPU execution with one thread and `ORT_DISABLE_ALL` now agrees
with the original JAX inputs:

| Ambient x64 | Input dtype and property | JAX result | Source runtime result |
| --- | --- | --- | --- |
| Disabled | FP32, `weak_type=True` | FP16 `2048` | FP16 `2048` |
| Disabled | FP32, `weak_type=False` | FP32 `2049` | FP32 `2049` |
| Enabled | FP64, `weak_type=True` | FP16 `2048` | FP16 `2048` |
| Enabled | FP64, `weak_type=False` | FP64 `2049` | FP64 `2049` |

`_summarize_input` / `_source_inputs` in the
[shared UX helper](../../../tests/extra_tests/converter/precision_ux_support.py)
carry weak flags into source
specifications, and `_as_sds_list` in
[`conversion_api.py`](../../../jax2onnx/converter/conversion_api.py) preserves
them under `PrecisionPolicy.SOURCE`. The direct
[`export_source_dtypes`](source_export.py) helper selects that same policy.
The report and compact display expose supplied, source-normalized, and actual
traced flags so a lost property cannot be hidden by matching changed dtypes.

The [46 input regressions](../../../tests/extra_tests/converter/test_source_inputs.py)
cover weak/strong JAX arrays
and specifications, both x64 settings, direct helper and UX routes, and original
JAX references that retain weakness. They also cover two distinct source
function definitions for weak/strong inputs and source-to-legacy-to-source
isolation. Legacy cases pass the original input sequence unchanged and compare
serialized CPU results with actual public calls. The source correction does not
change legacy input normalization or release a public source mode.

## Call-site settings and sequential exports

Review found that a named function could lose a caller's local JAX settings
when its body was traced again during export. With FP32 input `1e8`, scoped
FP64 cancellation should return FP32 `1`; the earlier implementation retraced
under ambient x64-off and returned FP32 `0`. The reverse setting also changed
the result. The source call now records x64 and default matrix precision,
replays them for body tracing, and separates cache entries for different
settings. Existing FP64 formals are retained even if the local body disables
creation of new FP64 values. Result signatures are checked after retracing.

The same review found that JAX could reuse a conditional branch traced during
source export inside a later public legacy export, leaking internal source
metadata into model kwargs. Ordinary JAX, legacy export, and source export now
have distinct tracing-cache contexts. The
[context regressions](../../../tests/extra_tests/converter/test_source_function_context.py)
exercise both export orders, omitted/false/true legacy settings, both ambient
x64 states, ordinary eager/JIT calls before and after export, nested functions,
and restoration after failure. Scoped matrix precision is preserved even when
it causes an explicit rejection by the bounded source matrix lowering.

The source-input, scope, and context regressions are collected by the normal
repository suite. The documentation helpers remain convenient entry points;
their implementation and durable regression coverage live under
`tests/extra_tests/converter/`.

## Validation and boundaries

The focused tests cover all three historical flag forms, shape-only/typed/
concrete inputs, both ambient x64 settings, actual public-call comparisons,
the numerical difference, dtype-dependent model choices, warning visibility,
scalar-output observation, pre-trace rejections, and configuration restoration
after successful and failed calls. At the original UX checkpoint, all **71 UX
tests** and the initial 26 straight-line source-export tests passed on both
installed stacks:

| Python | JAX / JAXLIB | ONNX | ONNX IR | ONNX Runtime | Combined result |
| --- | --- | --- | --- | --- | --- |
| 3.12.9 | 0.11.2 | 1.23.2 | 1.0.0 | 1.31.0 | 97 passed |
| 3.11.12 | 0.10.2 | 1.22.0 | 1.0.0 | 1.29.0 | 97 passed |

The later 46 source-input cases pass on both stacks and cover the repaired
weak-input defect. Audited nested-scope evidence is recorded separately in the
[integration findings](source_integration_findings.md); the historical 97-test
total above is not the current full-suite total.

Both demos completed successfully with CPU execution restricted to one thread.
Ruff, Black, and an explicit mypy check of the facade passed on the Python 3.12
stack. The tests compare warnings with the real public call: JAX 0.10.2 silently
truncates an FP64 `ShapeDtypeStruct` on the old omitted/False route, whereas JAX
0.11.2 emits a warning. The model's explicit FP64 `astype` under disabled x64
warns on both stacks. The report does not invent warnings absent from JAX.

The source subset includes audited `onnx_function` and two-branch `cond`
bodies, floating `gt`, and Boolean-to-INT32 selector conversion. It now also
covers the bounded preferred-FP32 matrix cases and the complete selected-FP64
normalization example, including required division, broadcasting, and square
root. Source arithmetic retains the tested addition/division and square/sum
boundaries. Recursive validation still excludes loops, JIT regions, and other
unaudited operations. Opsets remain 23/26.

The separate
[CPU sample acceptance check](../../../tests/extra_tests/converter/precision_runtime_support.py)
records numerical success, failure, or an unverified sample; execution errors
remain distinct. It rejects acceptance of the known FP16 square/sum mismatch
on the recorded runtimes. This is a test utility for supplied samples, not a
general runtime capability matrix, an automatic repair, or a public export API.

The configuration tests cover sequential calls and ordinary tracing/export
failures, plus the bounded scoped x64 and matrix-precision cases above.
Concurrent exports, arbitrary model side effects, and other configuration
changes remain outside this experiment. The report's top-level x64 observation
does not enumerate each nested call-site setting. Warning capture is local to
this sequential harness and is surfaced in its report. This is not a replacement
for future production warning and error design.
