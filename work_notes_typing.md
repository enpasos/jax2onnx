# Work Notes: Typing Improvements

## Goal
- Strengthen static typing in the converter and plugin subsystems so the JAX→ONNX pipeline catches shape/dtype drift, PRNG misuse, and metadata mismatches earlier (mirrors guardrails in `AGENTS.md`).

## Plan
- Review current mypy configuration and identify insertion points for stricter checks (converter + plugin hot paths).
- Introduce shared typed protocols/helpers for IR builders, lowering contexts, and PRNG metadata to reduce ad-hoc dict usage.
- Tighten mypy settings (per-package strict overrides + CI helper) to keep coverage high as changes land.
- Annotate prioritized modules and resolve new typing errors to exercise the stricter configuration.

## Progress
- ✅ Inspected `pyproject.toml` mypy settings and converter/plugin modules to map where protocols and stricter flags will have the most impact.
- ✅ Added shared typing helpers (`SymbolicDimOrigin`, `LoweringContextProtocol`) and updated converter + representative plugins to consume them, expanding the mypy target set with the new converter modules.
- ✅ Confirmed documentation already points engineers to `ir.to_proto(...)` and verified every `jax2onnx/converter` module relies solely on `onnx_ir`, so removing the old serde shim does not introduce new typing surface area or protobuf touch points inside the converter.
- ✅ Hardened typing around the plugin registry + helper sandboxes (`jax2onnx/plugins/plugin_system.py`, Issue52 fixtures/sandbox) by adding the `PrimitiveLowering`/`FunctionLowering` protocols, the `AxisOverrideInfo`/`RngTrace` helpers, and extending mypy’s strict coverage; updated the scatter/broadcast/loop-concat fixtures and tests to exercise the new annotations.
- ✅ Added `scripts/check_typing.sh` so CI (and humans) can run `./scripts/check_typing.sh` to execute `poetry run mypy --config-file pyproject.toml` with the expanded target set.
- ✅ Extended strict mypy coverage to plugin infrastructure (`jax2onnx/plugins/_axis0_utils.py`, `_loop_extent_meta.py`) and surfaced RNG factory metadata in CI via `scripts/report_rng_traces.py` + the updated `scripts/check_typing.sh`, which now invokes plugin discovery and prints the registered RNG helpers.
