# Work Notes: ONNX Coverage Approach

## Goal

Build and maintain an ONNX-operator-first coverage view for `jax2onnx`, then use it to drive implementation priorities (quick wins first).

Target release for the quick-win batch in this cycle: `0.12.1`.

## Scope

- Start from the official ONNX operator index.
- Compare against what plugins currently expose and lower.
- Show concrete gaps.
- Attach actionable recommendations.
- Map ONNX gaps to likely JAX entry points to make implementation planning easier.

## Source of Truth

- Official ONNX index:
  - `https://onnx.ai/onnx/operators/index.html`
- Plugin code scanned:
  - `jax2onnx/plugins/**/*.py`
- Generated report:
  - `docs/user_guide/onnx_operator_coverage.md`
- Generator script:
  - `scripts/generate_onnx_operator_coverage.py`

## Coverage Detection Strategy

For each ONNX operator from the official index:

1. `Metadata` signal
   - Found when plugin metadata contains `onnx__<Op>.html`.
2. `Lowering` signal
   - Found when plugin code contains `builder.<Op>(...)`.
3. `In Plugins`
   - True if either metadata or lowering signal exists.

This gives three useful states:

- Metadata + Lowering: implemented and documented.
- Metadata only: mapping declared, lowering signal missing or indirect.
- Lowering only: runtime emission exists, metadata missing.

## Matrix Columns

Current table columns in `docs/user_guide/onnx_operator_coverage.md`:

- `ONNX Operator`
- `In Plugins`
- `Metadata`
- `Lowering`
- `Plugin Modules`
- `Potential JAX Ops`
- `Suggested Next Step`

Extra (non-index) names from plugins are tracked separately to catch alias/helper names.

## Quick-Win Heuristics

Quick wins were treated as:

- Simple unary math ops with direct JAX primitive equivalents.
- Low plugin complexity.
- Clear `expect_graph` assertions.
- Limited risk to control-flow and layout-heavy paths.

## Coverage Snapshot (Current)

Latest generated matrix (`docs/user_guide/onnx_operator_coverage.md`) shows:

- Operators in official index: `200`
- Operators referenced in plugins: `155`
- Coverage: `77.5%`

## Implemented Quick Wins (0.12.1)

New plugins:

- `jax2onnx/plugins/jax/lax/acos.py`
- `jax2onnx/plugins/jax/lax/acosh.py`
- `jax2onnx/plugins/jax/lax/asin.py`
- `jax2onnx/plugins/jax/lax/asinh.py`
- `jax2onnx/plugins/jax/lax/atan.py`
- `jax2onnx/plugins/jax/lax/atanh.py`
- `jax2onnx/plugins/jax/lax/ceil.py`
- `jax2onnx/plugins/jax/lax/round.py`
- `jax2onnx/plugins/jax/lax/tan.py`
- `jax2onnx/plugins/jax/random/categorical.py` (`Multinomial`)
  - Added rank-3 logits stress testcases:
    - `random_categorical_logits_rank3` (opset 21, `Multinomial`)
    - `random_categorical_logits_rank3_opset23` (Gumbel-max fallback path)
  - Added symbolic batch testcase and lowering support:
    - `random_categorical_logits_symbolic_batch` (`run_only_dynamic`)
    - dynamic output-shape reconstruction via `Shape -> Slice -> Reshape`.

Extended existing plugins:

- `jax2onnx/plugins/jax/lax/integer_pow.py`
  - Added `Reciprocal` path for `y == -1`.
  - Added testcase `integer_pow_reciprocal`.
- `jax2onnx/plugins/jax/image/resize.py`
  - Added legacy `Upsample` lowering path for `opset <= 9`.
  - Added testcase `resize_nearest_opset9_upsample`.
  - Added additional legacy stress testcases:
    - `resize_linear_opset9_upsample` (structural-only due legacy interpolation drift vs JAX)
    - `resize_nearest_rank3_opset9_upsample`
- `jax2onnx/plugins/jax/lax/div.py`
  - Added fusion `(x + y) / 2 -> Mean`.
  - Added symbolic-shape fusion testcase:
    - `div_add_half_fuses_to_mean_symbolic`
  - Added fusion `x / ||x|| -> LpNormalization` for supported L1/L2 denominator patterns.
  - Added `Mean` fusion guard testcases:
    - `div_add_third_no_mean`
    - `div_add_half_f64_no_mean`
  - Added stress/guard testcases:
    - `div_lpnorm_l1_axis1`
    - `div_lpnorm_l2_axis2`
    - `div_sqrt_of_norm_no_lpnorm_fusion` (prevents over-fusion).
- `jax2onnx/plugins/jax/lax/reduce_window_sum.py`
  - Added `LpPool` lowering for supported `reduce_window_sum(abs(x), ...)` patterns (opset >= 22).
  - Added testcase `reduce_window_sum_abs_lppool_opset23`.
  - Added testcase `reduce_window_sum_abs_lppool_dilated_opset23`.
- `jax2onnx/plugins/jax/lax/dynamic_update_slice.py`
  - Added `TensorScatter` lowering for a strict cache-like `dynamic_update_slice` subset (opset >= 24).
  - Added testcase `dus_tensorscatter_axis1_opset24` (`skip_numeric_validation=True` due missing ORT kernel).
- `jax2onnx/plugins/jax/lax/scatter.py` + `jax2onnx/plugins/jax/lax/scatter_utils.py`
  - Added explicit `Scatter` / `ScatterElements` coverage and lowering path for supported 1D `PROMISE_IN_BOUNDS` cases.
- `jax2onnx/plugins/jax/lax/scan.py`
  - Corrected metadata mapping from `Scan` to `Loop` (actual lowering used).

## JAX Mapping Layer

`Potential JAX Ops` was added to make implementation planning explicit.

Examples:

- `Acos` -> `jax.lax.acos`, `jax.numpy.arccos`
- `Atanh` -> `jax.lax.atanh`, `jax.numpy.arctanh`
- `Reciprocal` -> `jax.numpy.reciprocal`, `jax.lax.integer_pow(x, -1)`, `1.0 / x`
- `ReduceLogSumExp` -> `jax.nn.logsumexp`, `jax.scipy.special.logsumexp`

## Runtime Compatibility Notes (ORT)

- In this environment, selected kernels are not consistently available for all dtype/opset combinations.
- `Mean` fusion in `div` is intentionally guarded away from the `float64` path due ORT kernel availability in this setup.
- `random_categorical` keeps `skip_numeric_validation=True` in metadata (stochastic op + runtime kernel differences across opsets/providers).
- `TensorScatter(24)` has no ORT kernel in this environment; dedicated testcase uses structural validation (`expect_graph`) plus allowlisted numeric skip.

## Validation Commands Used

Regenerate coverage matrix:

```bash
poetry run python scripts/generate_onnx_operator_coverage.py
```

Regenerate tests after plugin metadata/testcase changes:

```bash
poetry run python scripts/generate_tests.py
```

Focused regression runs used in this phase:

```bash
poetry run pytest -q tests/primitives/test_jax_image.py -k "resize_nearest_opset9_upsample"
poetry run pytest -q tests/primitives/test_random.py -k "categorical"
poetry run pytest -q tests/primitives/test_lax.py -k "Test_div"
poetry run pytest -q tests/primitives/test_lax.py -k "Test_reduce_window_sum"
poetry run pytest -q tests/primitives/test_lax.py -k "Test_dynamic_update_slice"
poetry run pytest -q tests/extra_tests/framework/test_do_not_skip_numeric_validation.py
```

## Current Remaining Quick Wins

Top low-hanging follow-ups from the uncovered table:

- Additional shape/opset stress tests for newly covered ops (`Mean`, `Multinomial`, `Upsample`)
- Review uncovered legacy/control-flow ops (`Scan`) and explicitly classify as roadmap vs non-goal.

## Follow-Up Plan

1. Expand targeted `expect_graph` checks for remaining new quick wins with dynamic-shape variants.
2. Keep metadata/lowering parity strict (remove stale metadata when lowering is absent).
3. Re-run coverage generation and focused regressions after each quick-win change.
