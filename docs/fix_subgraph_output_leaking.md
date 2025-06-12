
### Fixing the `jax2onnx` Converter: A Summary of Solutions

This document outlines the series of fixes required to address several cascading issues in the `jax2onnx` converter, starting from an output mismatch error and leading to subsequent attribute and value errors.

#### 1. Fix Subgraph Output "Leaking"

This is the primary and most critical fix, addressing the initial bug where the converted model had more outputs than the source JAX function (e.g., `JAX: 11, ONNX: 23`).

* **Symptom:** The ONNX model has more outputs than the original JAX function.
* **Root Cause:** The converter processes nested subgraphs (from primitives like `pjit` and `remat2`) by recursively calling its main processing logic. This caused the output variables of every subgraph to be incorrectly added to the final model's output list.
* **Key Concept:** The function that processes a `jaxpr` (`_process_jaxpr`) must be able to distinguish between the main graph and an inlined subgraph. It should only add final outputs to the ONNX model when processing the top-level main graph.

**Solution (`jax2onnx/converter/jaxpr_converter.py`):**

This solution requires a generic handler for inlining subgraphs and a modification to the core JAXPR processing function.

**A. Create a Generic Subgraph Inlining Handler:**
A single, reusable method should handle all primitives that contain a nested `jaxpr`, such as `pjit` and `remat2`.

```python
# In jaxpr_converter.py, add a new handler method
def _inline_subgraph(self, eqn, params):
    """Generic handler for inlining subgraphs from primitives like pjit or remat2."""
    # ① Fetch the closed jaxpr from the primitive's parameters
    closed = params.get("call_jaxpr") or params.get("jaxpr")
    if isinstance(closed, ClosedJaxpr):
        inner_jaxpr, consts = closed.jaxpr, closed.consts
    else:
        inner_jaxpr, consts = closed, params.get("consts", ())

    # ② Map inputs from the outer scope to the inner scope
    for outer_invar, inner_invar in zip(eqn.invars, inner_jaxpr.invars):
        self.set_var_name(inner_invar, self.get_name(outer_invar))

    # ③ Recursively process the subgraph, telling it it's NOT the main graph
    self._process_jaxpr(inner_jaxpr, consts, is_main_graph=False)

    # ④ Map outputs from the inner scope back to the outer scope
    for outer_outvar, inner_outvar in zip(eqn.outvars, inner_jaxpr.outvars):
        self.set_var_name(outer_outvar, self.get_name(inner_outvar))

# In _register_primitive_handlers, assign this method to the relevant primitives:
self.primitive_handlers["pjit"] = self._inline_subgraph
self.primitive_handlers["remat2"] = self._inline_subgraph
```

**B. Modify `_process_jaxpr`:**
The function needs to accept the new `is_main_graph` flag and use it to conditionally add outputs.

```python
# In jaxpr_converter.py, replace the existing _process_jaxpr

def _process_jaxpr(self, jaxpr: Any, consts: list[Any], *, is_main_graph: bool = True) -> None:
    """Processes a JAXPR's constants, inputs, and equations."""
    # ... (Logic for processing constants and inputs) ...
    
    for eqn in jaxpr.eqns:
        self._process_eqn(eqn)

    # FIX: Only define graph outputs for the top-level jaxpr call
    if is_main_graph:
        self.logger.debug("Defining graph outputs.")
        for i, var in enumerate(jaxpr.outvars):
            self.add_graph_output(var, i)

```

---
#### 2. Fix Downstream `AttributeError` and `ValueError`

The initial changes to fix the output leaking revealed that the converter's state was not being managed correctly, causing two subsequent errors during regression testing.

* **Symptom 1:** `AttributeError: 'Jaxpr2OnnxConverter' object has no attribute 'enable_double_precision'`
* **Symptom 2:** `ValueError: Expected kernel to be a constant tensor...`
* **Root Cause:** The `enable_double_precision` flag was not being stored on the converter instance. Furthermore, the logic for processing `jaxpr.constvars` was flawed, failing to populate the `name_to_const` dictionary that many plugins rely on to get weight and bias values.

**Solution:**

A corrected, robust implementation of `__init__` and `_process_jaxpr` is required to manage state correctly.

**A. Modify `__init__`:**
The converter must accept and store the `enable_double_precision` flag.

```python
# In jaxpr_converter.py, update the __init__ method

def __init__(
    self,
    builder: OnnxBuilder,
    # ... other params
    enable_double_precision: bool = False,
):
    # ...
    self.enable_double_precision = enable_double_precision
    # ...
```

**B. Finalize `_process_jaxpr` Logic:**
The function must process the JAXPR in a specific order: constants first, then inputs, then equations, and finally (and conditionally) outputs. This ensures plugins called during equation processing have access to all the necessary constant information.

```python
# In jaxpr_converter.py, the final, robust version of _process_jaxpr

def _process_jaxpr(self, jaxpr: Any, consts: list[Any], *, is_main_graph: bool = True) -> None:
    """Processes a JAXPR in the correct order: consts -> inputs -> eqns -> outputs."""
    self.logger.debug(f"Processing JAXPR. is_main_graph={is_main_graph}")

    # 1. Bind consts to ONNX initializers and populate the name_to_const map
    for var, const_val in zip(jaxpr.constvars, consts):
        # Logic to promote precision if self.enable_double_precision is True...
        const_name = self.get_constant_name(processed_val)
        self.set_var_name(var, const_name)
        self.name_to_const[const_name] = processed_val

    # 2. Add graph inputs from invars (only for main graph)
    if is_main_graph:
        for var in jaxpr.invars:
            self.add_graph_input(var)

    # 3. Process all equations
    for eqn in jaxpr.eqns:
        self._process_eqn(eqn)

    # 4. Define graph outputs (only for the main graph)
    if is_main_graph:
        self.logger.debug("Defining graph outputs.")
        for i, var in enumerate(jaxpr.outvars):
            self.add_graph_output(var, i)
```

This complete set of changes creates a robust converter that correctly handles nested subgraphs, manages internal state properly, and provides a reliable foundation for the plugins to build upon.