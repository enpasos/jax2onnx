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
