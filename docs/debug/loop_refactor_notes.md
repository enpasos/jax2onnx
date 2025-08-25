## Quick-start Summary for a *fresh* chat / branch

> **Goal:** Export a CNN-v2 model that hits `jax.lax.while_loop` without breaking the existing battery of 24 while-loop unit-tests.

---

### 1. Starting point (✅ all green)

* **Baseline plugin** converted `lax.while_loop` → ONNX `Loop` in \~200 LOC.
* All 24 while-loop tests in `tests/primitives/test_lax.py` passed.
* **CNN-v2 export still failed** (captured 4-D tensor + scalar counter inside loop).

---

### 2. Refactor sprint (current WIP branch)

Key changes made to the plugin (≈ 1000 LOC now):

| Area                             | What was added / changed                                                                                                               | Why                                                      |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Scalar-promotion**             | Detect `int{8,16,32}` loop counters → Cast to **INT64** on entry, auto-cast literals, fix mixed-dtype binops.                          | ONNX Loop spec mandates `INT64` iteration counter.       |
| **Captured tracers**             | Unified handling of `body.constvars` & `cond.constvars`; turn dynamic tracers into real sub-graph inputs; de-duplicate by ONNX symbol. | CNN-v2 body closes over 4-D tensor not in state tuple.   |
| **Name-collision guard**         | Generate fresh symbols when Loop outputs collide with inputs, & alias conflicts map.                                                   | Avoid ONNX “output reused as input” errors.              |
| **Value-info hardening**         | After every eqn add missing `value_info`; extra pass to ensure *every* tensor used by body/cond has shape+dtype metadata.              | Previous CNN bug hit `KeyError: 'var_95'` during export. |
| **Patched `jax.lax.while_loop`** | Monkey-patched at import time to bind custom primitive + closed jaxprs.                                                                | Needed for captured-jaxpr plumbing.                      |
| **Regression helpers**           | `_repro_nnx_scalar_and_captured_tensor_bug` etc. added as dedicated failing tests.                                                     | Keep the failure reproducible.                           |

---

### 3. New outcomes

| Status             | Tests                                                                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **✅ 18 / 24 pass** | Most scalar/vector/multi-state cases, INT64 promotion, bool passthrough…                                                                                 |
| **❌ 6 fail**       | `while_loop_with_closure(_f64)`, `while_loop_captured_tracer`, `while_loop_no_loop_output_reused_as_input`, and both `while_loop_nnx_repro(_f64)` cases. |
| **CNN-v2**         | Still fails (now during ONNX shape-inference).                                                                                                           |

Typical error now:

```
onnx.shape_inference.InferenceError:
(op_type:Loop, node name: while_loop_0): Output 1 is out of bounds
```

*Implication:* Number of outputs registered on the `Loop` node doesn’t match body-graph outputs **or** ONNX expects the first output slot (iteration condition) that we’re not providing correctly when captured tracers are present.

---

### 4. Known pain points in current code

1. **`extra_body_inputs` bookkeeping**
   – was `UnboundLocalError`; later fixed, but still fragile.

2. **Passthrough tracers added as body outputs**
   We create `Identity` nodes & `add_output`, but outer `state_out` list may **miss** the corresponding slot alignment.

3. **Loop output ordering vs ONNX spec**
   ONNX `Loop` outputs must *exactly* match loop-carried variables order; any additional scan outputs must come *after* them and be listed in the `body` graph’s `output` list in the same order.

4. **Shape-inference crash path**
   Happens *after* model build, so graph is syntactically valid; but shape-inference walks body graph, sees fewer outputs than referenced (index 1 OOB).

---

### 5. Suggested next steps (for new branch / chat)

1. **Minimal repro first:**
   *Use `while_loop_no_loop_output_reused_as_input`* – simplest failing case.

2. **Inspect final ONNX graph**
   Dump `model.graph.node` for the Loop and `body_graph.output` lengths; confirm mismatch.

3. **Verify ordering & counts**
   Ensure `state_out` is *exactly* `len(state_in)` long and indexes align with `body_builder.outputs[1:]` (output 0 is the condition).

4. **Add assert guards**
   During `to_onnx`, assert:

   ```python
   assert len(state_out) == len(body_output_syms)
   assert body_builder.outputs[0].type.tensor_type.elem_type == TensorProto.BOOL
   ```

   to trigger earlier and give clearer traces.

5. **Once green, re-run CNN-v2**
   Most likely it will pass once Loop output bookkeeping is fixed.

---

### 6. How to park & resume

* Create branch `feat/onnx-cnn2-bughunt` with all current work + this summary in `docs/debug/loop_refactor_notes.md`.
* Revert `main` (or mark CNN-v2 test `@pytest.mark.xfail`) to restore green CI.
* Open a *new* chat thread starting with:

  > “I have a mismatched Loop output/shape-inference error, simplest repro is …, here’s the ONNX dump ↓ … how do I reconcile outputs?”

 