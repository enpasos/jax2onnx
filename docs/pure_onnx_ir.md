# Plan: Make `converter2/` and `plugins2/` **100% ONNX-free**

> **Goal**
> All Python modules under `jax2onnx/converter2` and `jax2onnx/plugins2` must **not import** the ONNX protobuf library (`onnx`, `onnx.helper`, `onnx.onnx_ml_pb2`, `onnx.numpy_helper`, etc.). These packages should work purely on **onnx-ir** (IR-PY) abstractions. Any ModelProto work happens **outside** these directories.

---

## Why (brief)

* **Cleaner architecture**: IR-only in the compiler core; serialization & ModelProto tweaks happen at the edge.
* **Fewer deps & faster import**: no heavy protobuf on hot paths.
* **Fewer bugs**: eliminate our custom FunctionProto serializer; use native `onnx_ir.Function`.

---

## Current failing imports (from the policy test)

```
- jax2onnx/converter2/conversion_api.py: import onnx
- jax2onnx/converter2/conversion_api.py: from onnx import helper as oh, OperatorSetIdProto
- jax2onnx/converter2/conversion_api.py: from onnx import numpy_helper as _nh
- jax2onnx/converter2/function_scope.py: import onnx
- jax2onnx/converter2/function_scope.py: import onnx.helper as oh
- jax2onnx/converter2/function_scope.py: import onnx.onnx_ml_pb2 as onnx_ml
- jax2onnx/converter2/function_scope.py: from onnx import numpy_helper
- jax2onnx/converter2/function_scope.py: from onnx import TensorProto
- jax2onnx/converter2/ir_optimizations.py: import onnx
- jax2onnx/converter2/ir_optimizations.py: from onnx import helper as oh
- jax2onnx/plugins2/_post_check_onnx_graph.py: import onnx
```

---

## Design principles

1. **IR in the core**
   `converter2/*` and `plugins2/*` operate only on **onnx\_ir** (`import onnx_ir as ir`).
2. **Functions are first-class**
   Build function bodies as `ir.Function` and attach them to the **IR** model (let `onnx_ir` serialize to `FunctionProto`).
3. **One serialization edge**
   Create a single adapter outside converter2 (`jax2onnx/serde_onnx.py`) that converts **IR** → **ONNX ModelProto** and performs protobuf-level tweaks if any.
4. **ModelProto checks/optimizations live outside** converter2/plugins2 (e.g., `jax2onnx/post_onnx_opt.py`, `jax2onnx/onnx_checks.py`).

---

## Target architecture (high level)

```
plugins2/*  ─┐
             ├─→ converter2/*  ──→  onnx_ir.Model + [onnx_ir.Function...]
plugins2/*  ─┘

onnx_serde.py:  ir.Model → onnx.ModelProto      (ONLY place that imports onnx)
post_onnx_opt.py: protobuf-level fixes/passes   (imports onnx)
onnx_checks.py:  graph assertions, Netron checks (imports onnx)
```

---

## Refactor tasks (step-by-step)

### 0) Keep the policy test in CI

Already added:

```
tests/policy/test_no_onnx_in_converter2_plugins2.py
```

Run:

```bash
poetry run pytest -q tests/policy/test_no_onnx_in_converter2_plugins2.py
```

### 1) `converter2/ir_builder.py` → **IR only**

* ✅ You added `to_ir_model()`; **remove** any `onnx` imports & temp file conversion.
* Ensure all node creation uses `onnx_ir` types: `ir.Value`, `ir.Node`, `ir.Graph`, `ir.Model`.

### 2) `converter2/function_scope.py` → native `ir.Function`

* **Delete** custom FunctionProto serializer (`attach_functions_to_model`, any `onnx.helper`, `onnx_ml_pb2`, `numpy_helper` usage).

* Extend `FunctionScope` with:

  * `begin()` / `end()` (keep IR wiring in child ctx).
  * **`to_ir_function()`** that wraps the child IR buffers into an `ir.Function`:

    ```python
    def to_ir_function(self) -> ir.Function:
        g = ir.Graph(
            inputs=list(self.ctx.builder.inputs),
            outputs=list(self.ctx.builder.outputs),
            nodes=list(self.ctx.builder.nodes),
            initializers=list(self.ctx.builder.initializers),
            name=self.name + "_body",
            opset_imports={"": self.ctx.builder.opset, (self.domain or ""): 1},
        )
        return ir.Function(domain=self.domain, name=self.name, graph=g, attributes=[])
    ```

* **No ONNX imports** remain in this file.

### 3) `plugins2/plugin_system.py` (FunctionPlugin)

* Lower the callee **inside** the function scope using `fscope.ctx` (you’ve done most of this).

* After lowering, **collect** the `ir.Function`:

  ```python
  ir_func = fscope.to_ir_function()
  funcs = getattr(ctx, "_ir_functions", None)
  if funcs is None:
      setattr(ctx, "_ir_functions", [])
      funcs = ctx._ir_functions
  # Dedup by (domain, name)
  key = (ir_func.domain or "", ir_func.name)
  if key not in {(f.domain or "", f.name) for f in funcs}:
      funcs.append(ir_func)
  ```

* Emit the callsite as a normal node with `domain=ir_func.domain, op_type=ir_func.name`.

* **No ONNX imports** here.

### 4) `converter2/conversion_api.py`

* Remove **all** `onnx` imports from this module.
* Build IR as before; then:

  ```python
  ir_model = ctx.builder.to_ir_model(name=model_name, ir_version=10)
  # Attach functions collected on ctx
  for f in getattr(ctx, "_ir_functions", []) or []:
      if not hasattr(ir_model, "functions"):
          ir_model.functions = []
      ir_model.functions.append(f)
  ```
* Run **IR-level** passes only (if needed): rely on onnx\_ir passes (`TopologicalSort`, `NameFix`, `RemoveUnusedFunctions`).
* **Return the IR model** to the caller (change signature upstream), OR move **protobuf conversion** to a separate adapter (see step 5).

> If you must continue returning `onnx.ModelProto` for now, create a small adapter in a **new module** (outside converter2) and call it from `user_interface.py`:
>
> ```python
> from jax2onnx.serde_onnx import ir_to_onnx
> model_proto = ir_to_onnx(ir_model)
> ```

### 5) Add `jax2onnx/serde_onnx.py` (the **only** place that imports ONNX)

* Contents:

  ```python
  import onnx, onnx_ir as ir, tempfile, os

  def ir_to_onnx(ir_model: "ir.Model") -> onnx.ModelProto:
      tmp = None
      try:
          with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
              tmp = f.name
          ir.save(ir_model, tmp)
          return onnx.load_model(tmp)
      finally:
          if tmp and os.path.exists(tmp):
              os.remove(tmp)
  ```

* Any protobuf-level post-processing (OperatorSetIdProto, numpy\_helper, etc.) should live in **another** module (e.g., `post_onnx_opt.py`), **not** in converter2.

### 6) `converter2/ir_optimizations.py`

* Replace any protobuf-level logic with **IR-level** routines (operate on `ir.Model`, `ir.Graph`, `ir.Node`).
* If a specific pass truly requires protobuf, **move the pass** to `post_onnx_opt.py`.

### 7) `plugins2/_post_check_onnx_graph.py`

* This is a *test helper* that imports `onnx`. Move it **out of** `plugins2/`:

  * New path: `tests/_post_check_onnx_graph.py` or `jax2onnx/utils/onnx_checks.py` (outside converter2/plugins2).
  * Update plugin testcases to import from the new location.

---

## Incremental migration checklist

* [ ] `ir_builder.py`: rename `to_model_proto` → `to_ir_model`, remove ONNX imports.
* [ ] `function_scope.py`: add `to_ir_function()`, remove **all** ONNX imports & serializer code.
* [ ] `plugin_system.py`: collect `ir.Function` in `_ir_functions`, emit callsite with function domain/op.
* [ ] `conversion_api.py`: stop importing ONNX; attach `_ir_functions` to `ir_model`; return `ir_model` (or call `serde_onnx.ir_to_onnx` outside converter2).
* [ ] Add `serde_onnx.py` and move protobuf conversion there.
* [ ] `ir_optimizations.py`: keep only IR passes here; move protobuf passes to `post_onnx_opt.py`.
* [ ] Move `_post_check_onnx_graph.py` out of `plugins2/`.
* [ ] Run the policy test until green.

---

## Example: returning an ONNX model without violating the policy

In `user_interface.py` (or wherever your public API lives), do:

```python
from jax2onnx.converter2.conversion_api import to_onnx as to_onnx_ir   # returns ir.Model (rename later)
from jax2onnx.serde_onnx import ir_to_onnx

ir_model = to_onnx_ir(...)            # converter2 returns IR model
onnx_model = ir_to_onnx(ir_model)     # single place where ONNX protobuf is used
return onnx_model
```

This way **converter2** never imports `onnx`, but the public API still returns `onnx.ModelProto` for existing callers.

---

## Re-run the policy test

```bash
poetry run pytest -q tests/policy/test_no_onnx_in_converter2_plugins2.py
```

Expected: **PASS**.
If you still see offenders, the error list points to the exact line/file; move or refactor those imports per the plan.

---

## (Optional) Enforce via Ruff

Add to `pyproject.toml`:

```toml
[tool.ruff.per-file-ignores]
"jax2onnx/converter2/**.py" = ["ANN101","ANN102"]  # example; tune as needed
"jax2onnx/plugins2/**.py"   = ["ANN101","ANN102"]

[tool.ruff.flake8-type-checking]
strict = true
```

(We already have the stronger **pytest policy**; Ruff is just extra guardrails.)

---

## Outcome

* `converter2` & `plugins2` become **pure IR**.
* Functions are represented as **onnx\_ir.Function**; no custom FunctionProto code.
* One tiny **serde** module handles protobuf conversion.
* The “extra Constant” artifacts from hand serialization will go away with the custom serializer.
* The policy test prevents regressions.
