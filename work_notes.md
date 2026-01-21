# Work Notes: Integration of Native onnx-ir Cloning

## 1. Objective
Replace the custom `_PreservingCloner` class in `jax2onnx` with the native `onnx_ir.clone()` function (introduced in `onnx-ir` PR #313), while ensuring strict preservation of metadata required for JAX tracing (loop stack extents, etc.).

## 2. Context & Investigation
*   **Upstream Status**: `onnx-ir` added a public `clone()` method in PR #313. However, metadata preservation issues were noted, with a follow-up PR #314 pending.
*   **Initial Findings**:
    *   The `tmp/ir-py` version (containing PR #313) introduced API breaking changes: `ir.Graph` constructor arguments changed, and convenience methods like `add_input` were removed.
    *   Verification with `check_native_clone.py` confirmed that the distinct `ir.clone()` implementation **failed** to preserve `metadata_props`, `type`, and `shape` correctly for fresh values, breaking JAX tracing compatibility.
    *   **Environment Mismatch**: `pytest` executed against a system-installed `onnx-ir` that lacked the `allow_outer_scope_values` argument, causing crashes.

## 3. Implementation Plan
### 3.1. Local Patching of `onnx-ir`
To enable native cloning immediately without waiting for upstream releases, we identified and applied the necessary fixes locally in `tmp/ir-py`.

*   **File**: `tmp/ir-py/src/onnx_ir/_cloner.py`
*   **Changes**:
    *   Modified `_clone_or_get_value` to copy `metadata_props` and the internal `meta` dictionary.
    *   Modified `clone_node` to explicitly copy `type`, `shape`, `metadata_props`, and `meta` for output values.
*   **Artifact**: See `suggested_pr_changes.md` for the exact diffs.

### 3.2. Refactoring `jax2onnx`
*   Updated `jax2onnx/converter/ir_clone.py` to attempt using `graph.clone(allow_outer_scope_values=True)`.
*   **Robust Fallback**: Restored `_PreservingCloner` as a fallback mechanism. This ensures the code still works in environments with older `onnx-ir` versions (which caused `invalid graph` errors with a naive fallback) and guarantees correctness if the native clone fails or is missing features.

## 4. Verification Results
We validated the changes with the full test suite and specific verification scripts.

| Test Script | Status | Description |
| :--- | :--- | :--- |
| `check_native_clone.py` | **PASS** | Verifies that `graph.clone()` (patched) preserves all metadata and handles outer scopes. |
| `tests/extra_tests/converter/test_ir_clone.py` | **PASS** | Ensures no regressions in existing cloning scenarios. |
| **Full Suite** (`pytest`) | **PASS** | **2002 passed**, 3 skipped, 3 xfailed. Confirms full stability. |

## 5. Deliverables / Commit Messages

### Commit for `jax2onnx`
```text
Refactor ir_clone to use native onnx-ir clone with robust fallback

This change refactors `converter/ir_clone.py` to utilize the native `graph.clone(allow_outer_scope_values=True)` method from `onnx-ir` when available. This aligns `jax2onnx` with the upstream library's direction.

Crucially, we retain the custom `_PreservingCloner` class as a robust fallback mechanism. This ensures that:
1.  **Metadata Preservation**: JAX tracing metadata (loop extents, shapes, types) is preserved even if the installed `onnx-ir` version's native clone lacks this support.
2.  **Outer Scope Handling**: We correctly handle outer-scope values (scanning over captured variables) which is essential for JAX primitives like `scan` and `loop`.
3.  **Backward Compatibility**: The system remains stable regardless of whether the environment has the latest patched `onnx-ir` or an older version.

Verified with full test suite (2002 tests passed).
```

### Commit for `ir-py` (Submitted as [onnx/ir-py#318](https://github.com/onnx/ir-py/pull/318))
```text
Fix metadata preservation in Cloner for graph copies

The current `Cloner` implementation (used by `Graph.clone()`) did not preserve critical metadata properties when creating new values and nodes. This caused downstream issues for tools (like jax2onnx) that rely on `metadata_props` for tracing (e.g., preserving loop stack extents).

Changes:
-   **_clone_or_get_value**: Explicitly copy `metadata_props` and the internal `meta` dictionary when creating new Values.
-   **clone_node**: Explicitly copy `type`, `shape`, `metadata_props`, and `meta` for output values. The default behavior previously produced fresh values with `None` types/shapes, losing this information.

These changes ensure that `graph.clone()` produces a high-fidelity copy suitable for compiled graph transformations.
```
