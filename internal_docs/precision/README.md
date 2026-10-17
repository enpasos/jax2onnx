# Computation Precision

These documents describe current precision behavior, the proposed target, and
the proposed transition. Each topic has a user view focused on model preparation,
export options, and observable results, plus a technical view for implementation
work.

JAX model code and configuration define computation, accumulation, output types,
and deliberate conversions. The export call supplies the input shapes and dtypes
used for tracing and selects the ONNX representation. Runtime configuration
controls the execution environment. The target preserves the resulting numerical
contract within documented support limits. Examples include FP16 or BF16 work,
selective wider arithmetic for stability, and scientific calculations requiring
FP64.

| Topic | User view | Technical view |
| --- | --- | --- |
| Current behavior | [Precision today](precision_current_state_user_view.md) | [Current state](precision_current_state.md) |
| Proposed behavior | [Precision in the target design](precision_target_state_user_view.md) | [Target state](precision_target_state.md) |
| Proposed transition | [Moving to the target behavior](precision_migration_plan_user_view.md) | [Migration plan](precision_migration_plan.md) |

The current-state documents describe `main` at `fae17822` on 2026-10-17, including
the merged fix from [PR #275](https://github.com/enpasos/jax2onnx/pull/275).
The target and migration documents are proposals, not descriptions of an
available new export mode. Read down either column for a complete account.

A runnable [internal proof of concept](poc/README.md) explores the proposal using
the existing lowering pipeline for a small set of operations. It compares JAX
with ONNX execution, including a case where the legacy and proposed contracts
produce different answers. It also demonstrates an explicit ONNX opset limit.
Its coverage and exclusions are documented separately from the target contract.

Both views distinguish ONNX validity, preservation of the intended calculation,
runtime execution, and numerical validation. The proposed handling of limits is
to use a validated alternative within the selected target, require an explicit
choice for target or precision changes, and report a concrete failure when no
suitable implementation exists. An untested deployment target remains untested.
