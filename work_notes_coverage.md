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

## Implemented Quick Wins (0.12.1)

Added new plugins:

- `jax2onnx/plugins/jax/lax/acos.py`
- `jax2onnx/plugins/jax/lax/acosh.py`
- `jax2onnx/plugins/jax/lax/asin.py`
- `jax2onnx/plugins/jax/lax/asinh.py`
- `jax2onnx/plugins/jax/lax/atan.py`
- `jax2onnx/plugins/jax/lax/atanh.py`
- `jax2onnx/plugins/jax/lax/ceil.py`
- `jax2onnx/plugins/jax/lax/round.py`
- `jax2onnx/plugins/jax/lax/tan.py`

Extended existing plugin:

- `jax2onnx/plugins/jax/lax/integer_pow.py`
  - Added `Reciprocal` path for `y == -1`.
  - Added testcase `integer_pow_reciprocal`.

## JAX Mapping Layer

`Potential JAX Ops` was added to make implementation planning explicit.

Examples:

- `Acos` -> `jax.lax.acos`, `jax.numpy.arccos`
- `Atanh` -> `jax.lax.atanh`, `jax.numpy.arctanh`
- `Reciprocal` -> `jax.numpy.reciprocal`, `jax.lax.integer_pow(x, -1)`, `1.0 / x`
- `ReduceLogSumExp` -> `jax.nn.logsumexp`, `jax.scipy.special.logsumexp`

## Runtime Compatibility Note (ORT)

Some newly added ONNX ops are not fully covered by ONNX Runtime in `float64` on this setup.

For affected quick-win plugins (`Acos`, `Acosh`, `Asin`, `Asinh`, `Atan`, `Atanh`, `Tan`), testcase metadata uses:

- `disable_float64_test: True`

This keeps conversion coverage while respecting runtime kernel availability in CI-like test runs.

## Validation Commands Used

Regenerate coverage matrix:

```bash
python scripts/generate_onnx_operator_coverage.py
```

Focused quick-win regression run:

```bash
poetry run pytest -q tests/primitives/test_lax.py -k "acos or acosh or asin or asinh or atan or atanh or ceil or round or tan or integer_pow_reciprocal"
```

Policy sanity check:

```bash
poetry run pytest -q tests/extra_tests/framework/test_no_onnx_in_converter_plugins.py
```

## Current Remaining Quick Wins

Still marked quick win in the matrix:

- `ReduceL1`
- `ReduceL2`
- `ReduceLogSum`
- `ReduceLogSumExp`
- `ReduceSumSquare`

Each now includes `Potential JAX Ops` in the table to accelerate implementation.

## Follow-Up Plan

1. Implement remaining reduce-family quick wins in `primitives.lax` or `primitives.jnp`.
2. Add targeted `expect_graph` checks and keep metadata in sync.
3. Re-run generator and focused tests.
4. Re-evaluate any `metadata only` rows and either add concrete lowering signals or clean metadata aliases.
