# Typing Improvements Roadmap

Long-running effort to keep the converter + plugin stack type-safe under mypy. This
mirrors the guardrails in `AGENTS.md` (PRNG discipline, ONNX-only IR builders) and
keeps metadata/regressions visible earlier in CI.

## Goals

- Strengthen static typing in converter + plugin subsystems so shape/dtype drift,
  PRNG misuse, and metadata mismatches fail fast.
- Use shared protocols/helpers (`LoweringContextProtocol`, `SymbolicDimOrigin`,
  `PrimitiveLowering`, `FunctionLowering`, etc.) instead of ad-hoc `dict`/`Any`.
- Keep strict mypy coverage rolling forward via `scripts/check_typing.sh`.

## Strategy

1. Review mypy config regularly and identify the next converter/plugin hot paths
   to bring under strict overrides.
2. Introduce shared typed helpers for IR builders, lowering contexts, and PRNG
   metadata so new modules inherit concrete signatures.
3. Tighten mypy settings (per-package strict overrides + CI helper) so coverage
   ratchets forward without regressing performance.
4. Annotate prioritized modules, land the strict override, and resolve new typing
   errors immediately.

## Completed Milestones

- mapped the converter/plugin modules most affected by stricter mypy in
  `pyproject.toml`.
- Added shared typing helpers (`SymbolicDimOrigin`, `LoweringContextProtocol`,
  `AxisOverrideInfo`, `RngTrace`, `PrimitiveLowering`, `FunctionLowering`) and
  migrated converter + registry infrastructure to use them.
- Verified the converter no longer touches ONNX protobufs directly; IR-only flow
  stays intact after removing the old serde shim.
- Hardened typing around the plugin registry + Issue52 fixtures/sandboxes; the
  scatter/broadcast/loop-concat helpers exercise the new protocols.
- Added `scripts/check_typing.sh` so CI runs `poetry run mypy --config-file
  pyproject.toml`; wired in `scripts/report_rng_traces.py` so RNG helpers stay visible.
- Simplified mypy config so `files` currently tracks `jax2onnx/converter`, while
  follow-up strict coverage continues through the curated helper/override work.
- Extended strict coverage across high-traffic numpy/nn/random/attention plugins
  (reshape/tile/einsum/stack/split/take/transpose/squeeze/select/prod/outer/sort,
  activations, RNG primitives, attention shims, etc.) by annotating lowers, binding
  specs, and batch rules with shared protocols.
- Brought image/linear/conv/layer_norm/equinox helpers under strict overrides so
  conversion guardrails also apply to Equinox modules.
