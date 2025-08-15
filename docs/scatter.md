# JAX ↔ ONNX Scatter Rebuild: Design & Plan

**v1.3 — “Today Mode++” (complete replacement)**

> Clean-room rebuild of the scatter plugin family with **zero changes to tests or registry surfaces**. The implementation lives in **one file** (`jax2onnx/plugins/jax/lax/scatter_utils.py`) and introduces strict **Where/If guardrails**, **depth-2/3 strategies**, **policy gates**, and **dtype harmonization**.

---

## 1) Goals

* Pass **all existing registry testcases** as-is (no edits to their declarations).
* Eliminate ONNX Runtime shape/type inference errors (esp. `Where` / `If`).
* Implement JAX scatter semantics via ONNX `ScatterND` with:

  * Default `(N,K)` path
  * Depth-2 batched window path (B×L, single scatter axis)
  * Depth-3 H×W window path (B×H×W, two scatter axes)
* Respect JAX **scatter modes** (`PROMISE/None`, `CLIP`, `FILL_OR_DROP`) and **reductions** (`none/replace`, `add`, `mul`, `min`, `max`).
* Keep a narrow, testable surface: one public prep function + one emit function.

---

## 2) Non-Goals (for today)

* No refactors across multiple files (we keep a single module).
* No changes to the registry decorator, harness, or testcase payloads.
* No new public APIs; everything happens behind the existing callsite.

---

## 3) Public Surface (unchanged)

```python
# Call path used by all scatter plugins
data_name, idx_name, upd_name = _prepare_scatter_inputs_for_onnx(
    s,
    operand_v,
    indices_v,
    updates_v,
    dimension_numbers,
    scatter_mode,   # PROMISE/CLIP/FILL_OR_DROP
    reduction,      # "none"|"add"|"mul"|"max"|"min"
)

# Then emit the ONNX node
out = emit_scatternd(
    s, data_name, idx_name, upd_name,
    reduction=reduction, out_name=optional
)

# or a convenience wrapper
out = prepare_and_emit_scatternd(...)
```

The **tests and primitive registry stay exactly the same**.

---

## 4) File Layout (one module)

`jax2onnx/plugins/jax/lax/scatter_utils.py` contains:

* **Shape Helpers**
  `_ensure_np_dtype`, `_manually_ensure_shape_env_entry`, `_is_dim_symbolic`,
  `_are_dims_equal`, `_are_shapes_equal`, `_dims_concrete_equal_or_symbol_equal`,
  `_make_shape_concrete_for_prod`, `compute_expected_updates_shape`,
  `_map_operand_axis_to_updates_pos`, `_reduce_min_last_axis`

* **DType Harmonization**
  `_harmonize_float_dtypes` (no mixed float dtypes reach ORT)

* **Policies (modes)**
  PROMISE/None (scalar clamp), CLIP (vector clip), FILL\_OR\_DROP (mask + neutral)

* **Strategies**
  Default `(N,K)`; Depth-2 (B×L); Depth-3 (B×H×W)

* **Guardrails**
  `emit_where_with_guardrails` (explicit expand + dtype promote)
  If/cond branch reconciliation (shape+dtype equalization)

* **ONNX Emitters**
  `emit_scatternd`, `prepare_and_emit_scatternd`

* **Utilities**
  `_auto_pad_updates_if_smaller`, `_get_neutral_value`

All nodes added **must** register shape/dtype in `s.shape_env` using `ShapeDtypeStruct`.

---

## 5) Core Contracts & Invariants

1. **Where-contract (strict):**
   ONNX `Where` inputs must be **explicitly expanded** to a **single, identical shape** at build time.

   * Cast `cond → BOOL`.
   * Promote `x/y` dtypes via `np.promote_types`.
   * Choose a **reference** shape (prefer `x`, else `y`).
   * Use `Shape(ref)` + `Expand` to make `cond`, `x`, and `y` all match that shape.
   * Register output with that shape and the promoted dtype.

2. **If/cond branch reconciliation (pre-merge):**
   Before emitting `If`, the **then/else** outputs must have **identical** shapes & dtypes.
   Allowed transforms pre-merge: `Cast`, `Squeeze`, `Unsqueeze`, `Reshape` (≤1 `-1`), and `Expand` from `1 → N`.

   * **DType:** common promoted dtype (e.g., f32+f64 → f64).
   * **Shape:** if shapes are broadcast-compatible, select the **more informative** (replace `1` with other side’s concrete dim). If not compatible → **fail early** with a precise error.

3. **No mixed float dtypes into ORT:**
   If operand/updates floats differ, **cast updates → operand dtype**. Repeat per branch when inside an `If`.

4. **Reshape safety:**
   At most **one `-1`** per `Reshape`. If element counts mismatch, prefer **Pad** (when per-dim ≤ target and both are concrete) or **keep original** if strategy dictates (depth-2 preserving L-from-updates).

5. **Opset guards:**

   * `ReduceMin` axes: attribute (≤17) vs **input** (≥18).
   * ScatterND reduction attrs require **opset ≥ 16**; otherwise warn and use `"none"`.

---

## 6) Strategies

### 6.1 Default `(N,K)` ScatterND

* Compute the **expected ONNX updates shape** with:

  ```python
  compute_expected_updates_shape(
      dnums: ScatterDimensionNumbers,
      operand_shape: Sequence[int],
      indices_shape: Sequence[int],
  ) -> Tuple[int, ...]
  ```

  Supports both conventions for `update_window_dims`:

  * window = all operand dims except `inserted_window_dims`
  * or additionally excluding scatter dims

* If actual vs expected updates have the **same element count**, reshape once (≤1 `-1`).

* If strictly smaller **per-dim** and concrete: right-**Pad** with neutral to target shape.

* If element counts mismatch and can’t be padded → fail with a precise message listing shapes and `dnums`.

### 6.2 Depth-2 (B×L window; one scatter axis)

**Preconditions:**

* `|sdod| == 1`, `inserted_window_dims == []`, `operand_batching_dims == []`
* `upd_rank == op_rank` **or** `upd_rank == op_rank + 1 and updates[0] == 1`
* `indices.shape ∈ {(), (1,), (1,1)}` (K=1) or effectively equivalent

**Rules:**

* **L source:** try to map operand scatter axis → updates axis via `_map_operand_axis_to_updates_pos`; if found, **take L from updates**. If not found, fall back to operand L (log a warning).

* **Degenerate L==1 pick:** allowed only when:

  1. window covers all operand dims (|update\_window\_dims|==op\_rank)
  2. mapped updates axis exists and has length 1
  3. updates have a leading `N=1` arm
     Then `Gather` along that updates axis to remove it.

* **Indices grid:** build `(B, L, 2)` with:

  * batch grid: 0..B-1
  * column grid: `start + (0..L-1)`

* **Policies:**

  * PROMISE/None: clamp **scalar** `start` to `[0, max(0, dim−L)]`.
  * CLIP: do **not** clamp scalar; **Clip** the full index vector to `[0, dim−1]`.
  * FILL\_OR\_DROP: see §7 (masking).

* **Leading `N=1`** updates case `(1,B,L,…)`: drop singleton via `Squeeze([0])` **only when** the rest matches the expected `(B,L,…)` shape; otherwise keep to preserve the updates’ L semantics.

### 6.3 Depth-3 (B×H×W window; two scatter axes)

**Preconditions:**

* `|sdod| == 2`, `inserted_window_dims == []`, `operand_batching_dims == []`
* `len(update_window_dims) == op_rank`
* `upd_rank == op_rank + 1`
* `indices.shape == (1, 2)`

**Rules:**

* Build `(B,H,W,3)` index grids and **reshape** to `(-1,3)`.
* `updates` → **reshape** to `(-1,1)`.
* Shapes registered explicitly to satisfy ORT’s shape inference.

---

## 7) Policies (scatter modes)

### 7.1 PROMISE / None

* Keep semantics of “indices are valid”, but to keep ORT safe:

  * **Clamp scalar start** so that the window fits: `start ∈ [0, max(0, dim−L)]`.
  * Do not change valid callers; invalid callers get safe behavior instead of ORT load failure.

### 7.2 CLIP

* Do not clamp scalar start.
* **Clip the full index vector** element-wise to `[0, dim−1]`.

### 7.3 FILL\_OR\_DROP

* Build **per-row** validity over **K**:

  * `low_ok = (idx >= 0)`
  * `high_ok = (idx < dim_limits)` for the K scatter dims (and implicit batch if present)
  * `both_ok = low_ok & high_ok`  (shape = indices.shape)
  * Cast to `i64`, `ReduceMin` on last axis (K) to emulate `ALL(K)` → `(B,L)`
  * Cast back to bool: `row_ok` (shape `(B,L)`)

* **Broadcast** `row_ok` to updates rank: `(B,L)` → `(B,L,1,1,…)` via `Unsqueeze([2..])`.

* **Safe tensors**:

  * `safe_indices = Where(row_ok, indices, 0)`
  * `safe_updates = Where(row_ok, updates, neutral_or_fallback)` where:

    * reductions {add,mul,max,min} use **neutral** constants per dtype (`_get_neutral_value`)
    * replace/none uses `GatherND(data, safe_indices)` as fallback (“old value”)

* Then feed `safe_*` to `ScatterND`.

All `Where` calls go through **guardrails** (cond cast + expand to a single reference shape).

---

## 8) DType Rules

* **Global:** no mixed float dtypes reach ORT. Harmonize floats (updates → operand).
* **Where:** promote `x/y` dtypes via `np.promote_types` and cast both sides to the promoted dtype before the op.
* **FILL\_OR\_DROP:** neutral values computed per **final** updates dtype.
* **If:** reconcile branch output dtypes before merging.

---

## 9) Opset & Builder Rules

* `ReduceMin` axes:

  * ≤ 17: `axes=[-1]` attribute
  * ≥ 18: `axes` is a 2nd input tensor (int64 `[-1]`)
* `ScatterND` reduction:

  * ≥ 16: `reduction` attribute allowed in {"none","add","mul","min","max"}
  * < 16: warn and fall back to `"none"`
* Every node addition must:

  * emit a deterministic, contextual name (e.g., `Depth2_*`, `FoD_*`, `WhereGuardrails_*`, `IfRecon_*`)
  * **register** shape/dtype via `_manually_ensure_shape_env_entry`

---

## 10) Error Handling (fail fast, precise)

* **If reconciliation:** incompatible shapes (non-broadcast-compatible) → raise with both shapes, dtypes, and node context.
* **Default path:** updates element-count mismatch (and cannot pad) → raise with operand/indices shapes and `dnums`.
* **Reshape safety:** more than one non-concrete dim that would need `-1` → raise.

---

## 11) Test Plan

### A) Keep the existing registry tests (unchanged)

They validate real paths end-to-end.

### B) Add micro-tests (local, surgical)

1. **Where Guardrails**

   * `(B,L)` cond vs `(B,L,1,1)` data → both inputs expanded; ORT shape inference passes.
   * Mixed float dtypes → promoted; output dtype as expected.

2. **If Reconciliation**

   * `(B,L)` vs `(B,L,1)` branches → reconciles to `(B,L,1)`.
   * f32 vs f64 → reconciles to f64.
   * Incompatible `(B,L)` vs `(B,M)` → early error.

3. **Depth-2**

   * L derived from updates when mapping exists; else fallback with warning.
   * Leading `N=1` squeeze only when the rest matches; otherwise preserved.
   * Degenerate `L==1` pick removes the mapped axis.

4. **FILL\_OR\_DROP**

   * Out-of-range rows → indices zeroed and neutral/fallback applied, with proper broadcast of `(B,L)` → `(B,L,1,…)`.

5. **Depth-3**

   * `indices (1,2)` → `(-1,3)`, updates → `(-1,1)`; ORT loads.

6. **Opset Boundaries**

   * `ReduceMin` axes as attribute vs input works at 17/18.

---

## 12) Rollout Order (to drain the 25 reds fast)

1. **Where guardrails** (explicit `Expand` → single reference shape)
   *Fixes “Incompatible dimensions” in fluids & cond paths.*

2. **If branch reconciliation** (pre-merge shape+dtype)
   *Removes remaining `If`-wrapped inference failures.*

3. **Depth-2 (“L from updates”)** + degenerate L==1 pick + leading `N=1` handling.

4. **FILL\_OR\_DROP** row mask broadcast + neutral/fallback updates.

5. **Depth-3** flatten.

6. **Default path**: reshape/pad guardrails (one `-1`, pad if concrete ≤ target).

Re-run the suite after each step; keep the micro-tests green throughout.

---

## 13) Logging/Telemetry

* Node names prefixed with context:

  * `WhereGuardrails_*`, `IfRecon_*`, `Depth2_*`, `Depth3_*`, `FoD_*`, `DefaultUpdates_*`
* One-line summaries for strategy selection and L source:

  * `Depth-2 selected: B=?, L=? (from updates|operand), start policy=<...>`
* Error messages include exact shapes/dtypes and `dnums`.

---

## 14) Risks & Mitigations

* **Broadcast surprises at Where/If:**
  Mitigation: explicit `Expand` to a single reference shape; branch reconciliation before `If`.

* **Symbolic dims in Reshape:**
  Mitigation: “one `-1` only” rule; otherwise Pad or keep; detailed error with the offending dims.

* **Mixed float dtypes:**
  Mitigation: global harmonization (updates → operand), re-applied per branch.

---

## 15) Definition of Done

* Single-file implementation in `scatter_utils.py` with sections above.
* 4–6 micro-tests added (Where, If, Depth-2, FILL\_OR\_DROP, Depth-3, Opset boundary).
* Full registry suite **passes** on target opset(s) without changing any test declarations.
* No ORT `Where/If` shape/type inference errors; no mixed float dtypes in the graph.

---

## 16) Appendix — Helper Behaviors (concise)

* **`emit_where_with_guardrails(s, cond, x, y, out_name?, context?) -> str`**

  * Casts `cond→BOOL`, promotes `x/y` dtypes, **expands all three** to a single reference shape, emits `Where`, registers out.

* **`compute_expected_updates_shape(dnums, operand_shape, indices_shape)`**

  * Derives ONNX-spec updates shape; supports both window conventions; raises if `update_window_dims` length is inconsistent.

* **`_auto_pad_updates_if_smaller(...)`**

  * Right-pads when every dim `orig ≤ target` and both sides are **concrete**; otherwise no-op.

* **`_get_neutral_value(reduction, dtype)`**

  * Returns neutral element for add/mul/min/max; special-cases bool; default 0 for replace/none.

* **`_reduce_min_last_axis(s, inp, out, keepdims=0)`**

  * Emits ReduceMin on last axis across opsets (attr vs input).

* **`emit_scatternd(..., reduction="none")`**

  * Sets reduction attr if opset ≥ 16; result inherits `data` shape/dtype.

---

**Version tag to set in code:**
`SCATTER_UTILS_VERSION = "v1.3 TodayMode++ (Where+If guardrails, Depth2/3, Policies, Harmonize)"`

 
