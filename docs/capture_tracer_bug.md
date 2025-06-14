# Lessons Learned – Fixing the `while_loop`-Captured-Tracer Bug

---

## 0  Overview

Our mission started with **one** failing test –
`test_while_loop_tracer_passthrough`.
After a tour-de-force through `TypeError` → `TracerArrayConversionError` → `RuntimeError (Topological Sort)` → `ONNX TypeInferenceError`, and finally a **regression failure**, we have a much clearer picture.

The root cause is how we handle *captured tracers* (variables from an outer scope used inside a `lax.while_loop`). The converter struggles to correctly identify these tracers and wire them into the ONNX `Loop` operator's body graph without breaking the graph's structure.

Below is the fully updated post-mortem and a new, more robust implementation plan.

---

## 1  Key Lessons

| #     | Take-away                                                                                                                                                                                                  | Why it matters                                                                                                                                                                                                   |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Sub-graph tracing creates fresh Python objects.**<br> A tracer that appears inside the `while` body is a **different object** from the outer `Var` that produced the same logical value, even if they are semantically identical. | Dictionary lookups keyed by the tracer object (`var_to_name[tracer]`) will fail. We must look up the tracer's underlying `Var` from the original trace to establish its identity. |
| **2** | **Don’t patch core utilities for a niche bug.**                                                                                                                                                            | An early attempt to modify `get_var_name()` globally broke unrelated plugins. The fix *must* be fully contained within the `WhileLoopPlugin`.                                                         |
| **3** | **Heuristics for tracer identity are fragile.**                                                                                                                                                           | Comparing abstract values (`cval.aval == loop_state_var.aval`) is not a reliable way to check if two variables are the same. It led to regression failures. The check must be based on the object identity of the underlying `Var`. |
| **4** | **Loop inputs must be correct and complete.** <br> The ONNX `Loop` op is strict. The number of tensors passed to it must exactly match the number of inputs defined in its body subgraph. | An intermediate fix produced an `ONNX TypeInferenceError` because we passed 3 inputs when the body graph expected 4. Every captured variable must be accounted for as a loop-invariant input. |

---

## 2  Revised Implementation Roadmap

This plan is more robust and avoids the fragile heuristics that caused regression failures.

| Step  | Action                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Outcome check                                                                             |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **1** | **Revert to Green.** Revert `jax2onnx/plugins/jax/lax/while_loop.py` to its state before any of our changes. Keep the `test_while_loop_tracer_passthrough` test case (using `input_values`) so it continues to fail. | Baseline: only the target test case fails. |
| **2** | **Inside `WhileLoopPlugin.to_onnx`**, when iterating over captured variables (`all_consts`):<br>• Handle non-Tracer constants first and `continue`.<br>• For a Tracer `cval`, find its underlying `Var` from the outer graph: `underlying = s.get_var_from_tracer(cval)`.<br>• Check if we already have a name for it: `outer_name = s.var_to_name.get(underlying)`.<br> | A clean separation of logic. |
| **3** | **If `outer_name` is `None` (the tracer is unknown):**<br>• **This is the critical step.** Raise a clear `RuntimeError`. We should not invent names. This condition signifies that a value used inside the loop (e.g., `y = x*2`) was not properly passed as either a loop state variable or a constant. The JAX jaxpr should have made it a `constvar`. Forcing a fix here indicates a deeper issue. *The `passthrough` test should NOT trigger this error*. | This enforces correctness. The converter should not have to guess graph connections. The passthrough test should work because `x` is already a known input. |
| **4**| **Process the Known Tracer**: <br>• Map the inner `cvar` to the found `outer_name`: `body_conv.var_to_name[cvar] = outer_name`.<br>• Ensure `outer_name` is an input to the body graph: `body_builder.add_input(...)`.<br>• **If `outer_name` is NOT one of the loop state inputs (`state_in`)**, add it to `extra_body_inputs` to be passed to the `Loop` op. | This correctly identifies loop-invariant dependencies that are not part of the state, preventing duplicate inputs and ensuring the ONNX graph is valid. |
| **5**| **Build the ONNX `Loop` node**: <br>• Final `Loop` inputs are: `[max_trip, init_cond] + state_in + extra_body_inputs`. <br>• Final `Loop` outputs are just `state_out`. | The inputs to the ONNX op now correctly match the inputs defined in its body subgraph, satisfying the type inferencer. |
| **6**| **Run the single failing test** (`...::test_while_loop_tracer_passthrough`). | Should now pass. ✔️ |
| **7**| **Full regression sweep** (`pytest -q`). | No other tests should be broken. ✔️ |

---

## 3  Green-Bar Checklist

* [ ] `test_while_loop_tracer_passthrough` passes.
* [ ] All other `while_loop_*` variants pass.
* [ ] No regressions in other plugins (e.g., `nnx.dropout`, `lax.cond`).
* [ ] All ONNX models produced are valid (`onnx.checker.check_model`).