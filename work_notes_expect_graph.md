# Expect Graph Coverage Notes

## Goal
Attach auto-generated `expect_graph` structural assertions to every plugin and example testcase so each converter path is validated against the ONNX IR it produces.

## Current Checklist
- [x] `plugins/examples/eqx/simple_linear.py`
- [x] `plugins/flax/nnx/elu.py`
- [x] `plugins/flax/nnx/avg_pool.py`
- [x] `plugins/flax/nnx/relu.py`
- [ ] Remaining `flax/nnx` linear-family modules (e.g., `linear_general.py`, `conv.py`)
- [x] `plugins/flax/nnx/linear.py`
- [x] `plugins/flax/nnx/sigmoid.py`
- [x] `plugins/flax/nnx/tanh.py`
- [x] `plugins/flax/nnx/softplus.py`
- [x] `plugins/flax/nnx/leaky_relu.py`
- [x] `plugins/flax/nnx/gelu.py`
- [x] `plugins/flax/nnx/softmax.py`
- [ ] (No remaining flax/nnx activations without coverage)
- [ ] Other plugin families (JAX lax, Equinox, remaining examples)

## Standard Workflow
1. Pick the next component from the TODO portion of the checklist.
2. For each testcase in that component:
   - Run `poetry run python scripts/emit_expect_graph.py <testcase_name>` to capture the current graph snippet.
   - Add (or merge into existing) `post_check_onnx_graph` entries using the emitted snippet. When merging with existing logic, wrap both checks in a helper to preserve prior expectations.
3. Re-run the targeted pytest class, e.g. `poetry run pytest -q tests/primitives/test_nnx.py::Test_<Component>`.
4. Update this checklist before moving on.

## Context for New Chats
When starting a new session, run through the **Standard Workflow** above, using this file as the single source of truth for which components still need coverage.
