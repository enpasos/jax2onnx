# Bounded numerical-contract findings

Local implementation and CPU evidence, updated 2026-10-18.
[numerical_contract.py](numerical_contract.py) compares eight cases: public
exports, changed-source examples, independent hand-built controls, and actual
source exports. Artifact origins remain explicit. The
[private integration findings](source_integration_findings.md) record the
implementation and complete validation; no public source mode is available.

## Reproduce

From the repository root:

```sh
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run python -m internal_docs.precision.poc.numerical_contract

JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run pytest -q internal_docs/precision/poc/test_numerical_contract.py
```

The demo retains models in memory and prints versions, graph types, JAX eager
and JIT results, ONNX ReferenceEvaluator results, and separate execution and
numerical status. It enables x64 during model preparation and restores the
caller setting on success and failure. Runtime checks use
`CPUExecutionProvider`, one intra/inter-op thread, opset 23, and optional ORT
optimization disabled and enabled. Every model passes the full ONNX checker.
The installed stacks use JAX 0.11.2 / ORT 1.31.0 on Python 3.12 and JAX 0.10.2 /
ORT 1.29.0 on Python 3.11, on this x86_64 host.

## Matrix product: preserved preferred FP32 result

With FP16 `x=[[[2048,1]]]`, FP16 `w=ones((1,2,1))`, contracting axes `((2,),(1,))`,
batch axes `((0,),(0,))`, and `preferred_element_type=float32`, JAX eager and JIT
return FP32 `2049`. Public legacy export with `enable_double_precision=False`
produces FP16 `MatMul` and returns FP16 `2048`. That public gap remains.

The private source path now exports the unchanged JAX callable as local FP32
operand casts followed by FP32 `MatMul`. Its FP16 input interface and other
uses of the original operands remain intact. ReferenceEvaluator and both ORT
settings return FP32 `2049`. Casting only the final result would not repair
lost arithmetic.

This lowering admits positive static rank-2 matrices or rank-3 matrices with
batch axis 0, the usual matrix contracting axes, FP16 operands, an explicitly
preferred FP32 result, and default precision. Other combinations, including
explicit HIGH/HIGHEST or algorithm requests, are rejected as private coverage
limits. See the [dot regressions](../../../tests/extra_tests/converter/test_source_dot_precision.py).
They also cover selected overflow, underflow, cancellation, longer reductions,
special values, shared operands, and serialized artifacts on both stacks.

Finite FP16 values and their products are exactly representable in FP32. This
supports widening before multiplication for the admitted contract; it does not
prove every reduction order or runtime algorithm. JAX distinguishes
[preferred result/accumulation hints from algorithm requirements](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot.html).
The lowering does not promise native FP16 hardware, performance, or a general
translation of those algorithm requirements. The changed-source public-export
example remains a separate comparison, not the implementation of this source
path.

## Normalization: selected FP64 variance through the real exporter

The FP32 input has shape `(2,16)`: one row is `[8192,-8192] + [1,-1]*7`, and the
other contains sixteen `1e8` values. Centering and squares use FP32; summing the
rounded squares and dividing by sixteen use FP64. Variance is explicitly
rounded to FP32 before epsilon, square root, and final FP32 normalization.

The actual JAX callable now exports through the bounded source pipeline using
`div`, `broadcast_in_dim`, and the patched `jax.numpy.sqrt` lowering. The
hand-built ONNX graph remains an independent control. Both agree with JAX and
the staged NumPy/`math.fsum` reference on exact variance `8388609` and `0`.
A NumPy FP32-only sum gives `8388608` for the first row. Centered values, squares,
variance, and the constant row's zero result are checked exactly; only final
normalized values use `rtol=3e-7, atol=1e-7`.

The [durable normalization regressions](../../../tests/extra_tests/converter/test_source_normalization.py)
check the actual export inline, inside an ONNX Function, and in both conditional
branches after saving/reloading, with ORT optimization off/on. This is bounded
end-to-end evidence for model-chosen FP64 work. It does not adopt the separate
legacy LayerNorm accuracy policy or establish arbitrary normalization coverage.

## Source operations retain their arithmetic boundaries

Review found that legacy algebraic fusion could change newly admitted source
arithmetic: `(x+x)/float32(2.00001)` became `Mean(x,x)`, yielding `1` instead of
approximately `0.999995` for `x=1`. Source lowering now retains Add and Div.
Source also retains Mul followed by ReduceSum, and a separate Sqrt where traced,
instead of using unvalidated Mean, LpNormalization, ReduceSumSquare, or ReduceL2
substitutions. Legacy behavior is unchanged. The
[fusion regressions](../../../tests/extra_tests/converter/test_source_arithmetic_fusion.py)
check these boundaries with values that distinguish the computations.

The shared optimizer is now policy-aware too. SOURCE excludes four unaudited
arithmetic/reduction rewrites even when a legacy environment flag requests one.
The near-one reciprocal witness `x * (float32(1.000001) / sqrt(y))` at `x=y=1`
retains `1.0000009536743164` in the serialized source artifact and both tested
ORT profiles. Its former conversion to `1` is repaired. See the
[optimizer regressions](../../../tests/extra_tests/converter/test_source_optimizer_precision.py).

## Further exact-sample runtime failures

For FP32 `x=3`, `y=2`, the source graph retains `x * (1 / sqrt(y))`.
JAX eager, saved/reloaded ReferenceEvaluator and ORT 1.29/1.31 with optimization
disabled return `2.1213202476501465`. Both runtimes with optimization enabled
return `2.1213204860687256`. That profile fails the selected exact-sample
criteria. Tests reject its failed acceptance while permitting a future runtime
correction. This is runtime evidence, not a remaining converter reassociation
or a universal bitwise-equality requirement. Returning the reciprocal as
another output preserves this sample in both profiles; changing the model
interface is not proposed as an automatic fix.

An ad-hoc captured-state probe with x64 disabled uses FP16 input `1`, weak
FP32 bias `4096`, then converts the sum to FP32 and scales by `2**-20`.
JAX eager/JIT and ReferenceEvaluator return `0.00390625`. ORT 1.29/1.31 return
`0.003907203674316406` with optimization disabled and enabled: the FP16 Add
rounding is lost. This probe is not a durable acceptance regression and does
not establish the underlying cause or a general repair. Positive captured-state
cache tests use FP32 arithmetic to keep cache correctness separate from this
runtime limitation.

## FP16 intermediate rounding: failed target acceptance

For FP16 `x=full(8,2**-13)`, the model returns both `square=x*x` and its FP16 sum.
JAX eager/JIT and ReferenceEvaluator return zero for both. The source graph now
contains ordinary Mul and ReduceSum. Both tested ORT versions, with optional
optimization disabled and enabled, expose zero squares but return sum `2**-23`.
The independent unfused control has the same discrepancy. Removing converter
fusion therefore does not repair the runtime sample; these observations do not
identify the underlying runtime cause.

A changed-source candidate explicitly uses
`Cast(FP32) -> Mul(FP32) -> Cast(FP16) -> Cast(FP32) -> ReduceSum(FP32) -> Cast(FP16)`.
It retains the intermediate and result FP16 types and returns zero on this
sample. This is bounded evidence, not a validated automatic transformation.

The private [CPU sample gate](../../../tests/extra_tests/converter/precision_runtime_support.py)
checks exact output types, shapes, and values for supplied artifact/input/reference
samples and records actual runtime settings. `require_sample_acceptance` rejects
both failed and unverified reports. A missing execution profile stays unverified;
a kernel load/run error is reported separately. The original square/sum sample
fails this gate on both installed runtimes. The explicit wider candidate passes
for this sample only. See the [acceptance tests](../../../tests/extra_tests/converter/test_source_runtime_acceptance.py).

Diagnostic tests permit a future runtime correction to pass the original
sample. Recognizing the existing discrepancy still reports failed acceptance.
This gate validates samples; it is not a general capability database, automatic
fallback, or public deployment-target interface. The original composition is
excluded from numerical acceptance on the recorded targets until revalidated.
