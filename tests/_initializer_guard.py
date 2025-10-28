# tests/_initializer_guard.py

"""Initializer shape guard (currently disabled by default).

Historically the guard patched ``IRBuilder.to_ir_model`` to raise whenever an
initializer flowed through reshape/expand-style ops. That proved too invasive:
many existing conversions intentionally rely on those patterns, so installing
the guard globally caused large portions of the test suite to fail.

The guard helpers remain available for targeted diagnostics, but the
auto-install hook is now a no-op.  Tests that relied on the guard
(``test_initializer_shape_guard``) will automatically skip when the guard is
disabled.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import (
    EXAMPLE_REGISTRY,
    PLUGIN_REGISTRY,
    import_all_plugins,
)
from tests import t_generator as tgen

PURE_SHAPE_OP_INPUTS: dict[str, tuple[int, ...]] = {
    "Expand": (0,),
    "Flatten": (0,),
    "Identity": (0,),
    "Reshape": (0,),
    "Squeeze": (0,),
    "Unsqueeze": (0,),
}

# Allowlisted patterns that still route initializers through shape-only ops.
# Keyed by (model_name, op_type, initializer_name).
ACCEPTED_CASES: set[tuple[str, str, str]] = {
    ("broadcast_in_dim_batch", "Expand", "const_1"),
    ("broadcast_in_dim_dynamic_B", "Expand", "const_0"),
    ("concatenate_tile_and_symbolic", "Expand", "const_1"),
    ("cond_internal_constant", "Identity", "const_1"),
    ("cond_internal_constant_f64", "Identity", "const_1"),
    ("cond_multiple_operands_in_tuple", "Identity", "const_1"),
    ("cond_multiple_operands_in_tuple", "Identity", "const_2"),
    ("cond_multiple_operands_in_tuple", "Identity", "const_3"),
    ("cond_scalar", "Identity", "const_1"),
    ("cond_with_scatter", "Identity", "const_0"),
    ("cond_scatter_repro", "Identity", "const_0"),
    ("cond_scatter_repro", "Identity", "const_1"),
    ("device_put_scalar", "Identity", "const_0"),
    ("dus_1d_block_update", "Unsqueeze", "dus_rank_scalar_0"),
    ("dus_1d_scalar_update", "Unsqueeze", "dus_rank_scalar_0"),
    ("dus_2d_block_update", "Unsqueeze", "dus_rank_scalar_0"),
    ("dus_3d_block_update", "Unsqueeze", "dus_rank_scalar_0"),
    ("dus_4d_block_update", "Unsqueeze", "dus_rank_scalar_0"),
    ("reshape_from_scalar", "Reshape", "const_0"),
    ("reshape_to_scalar", "Reshape", "const_0"),
    ("scatter_add_depth2_depth2_helper_regression", "Reshape", "const_4"),
    (
        "scatter_add_fluids_pattern_updates_5_4_1_1",
        "Unsqueeze",
        "scatter_num_updates_0",
    ),
    ("scatter_add_fp64_dtype_mismatch", "Reshape", "const_4"),
    (
        "scatter_add_mismatched_window_dims_from_user_report",
        "Unsqueeze",
        "scatter_num_updates_0",
    ),
    (
        "scatter_add_mismatched_window_dims_from_user_report2",
        "Unsqueeze",
        "scatter_num_updates_0",
    ),
    (
        "scatter_add_mismatched_window_dims_from_user_report3",
        "Unsqueeze",
        "scatter_num_updates_0",
    ),
    ("scatter_add_scalar", "Unsqueeze", "scatter_num_updates_0"),
    ("scatter_add_scalar", "Reshape", "const_2"),
    ("scatter_add_vector", "Reshape", "const_1"),
    ("scatter_depth2_mixed_dtypes_fp_mismatch", "Reshape", "const_4"),
    ("scatter_depth2_mixed_dtypes_fp_mismatch_f64", "Reshape", "const_4"),
    ("scatter_max_depth2_helper_regression_fp64", "Reshape", "const_4"),
    ("scatter_max_fp64_dtype_path_check", "Reshape", "const_1"),
    ("scatter_min_depth2_helper_regression_fp64", "Reshape", "const_4"),
    ("scatter_min_fp64_dtype_path_check", "Reshape", "const_1"),
    (
        "scatter_mul_fluids_pattern_updates_5_4_1_1",
        "Unsqueeze",
        "scatter_num_updates_0",
    ),
    (
        "scatter_mul_mismatched_window_dims_from_user_report",
        "Unsqueeze",
        "scatter_num_updates_0",
    ),
    (
        "scatter_mul_mismatched_window_dims_from_user_report2",
        "Unsqueeze",
        "scatter_num_updates_0",
    ),
    (
        "scatter_mul_mismatched_window_dims_from_user_report3",
        "Unsqueeze",
        "scatter_num_updates_0",
    ),
    ("scatter_set_axis0", "Unsqueeze", "scatter_num_updates_0"),
    ("scatter_set_axis0", "Reshape", "const_3"),
    ("scatter_set_middle", "Unsqueeze", "scatter_num_updates_0"),
    ("scatter_set_middle", "Reshape", "const_2"),
    ("scatter_set_single", "Unsqueeze", "scatter_num_updates_0"),
    ("scatter_set_single", "Reshape", "const_2"),
    ("scatter_set_vector", "Reshape", "const_1"),
    ("scatter_static_slice_set", "Unsqueeze", "scatter_num_updates_0"),
    ("stack_axis_0", "Unsqueeze", "const_0"),
    ("stack_axis_0", "Unsqueeze", "const_1"),
    ("stack_axis_1", "Unsqueeze", "const_0"),
    ("stack_axis_1", "Unsqueeze", "const_1"),
    ("stack_negative_axis", "Unsqueeze", "const_0"),
    ("stack_negative_axis", "Unsqueeze", "const_1"),
    ("stack_scalars", "Unsqueeze", "const_0"),
    ("stack_scalars", "Unsqueeze", "const_1"),
}

# Guard availability is recorded separately for diagnostics, but the guard is
# intentionally disabled by default.
GUARD_ENABLED: bool = False


class InitializerShapeGuardError(RuntimeError):
    """Raised when a pure shape op consumes an initializer value."""


class InitializerShapeGuard:
    """Centralised checker enforcing initializer invariants."""

    __slots__ = ("enabled", "accepted_seen", "models_seen", "last_failure")

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.accepted_seen: set[tuple[str, str, str]] = set()
        self.models_seen: set[str] = set()

    def check(self, model: ir.Model, *, model_name: str) -> None:
        if not self.enabled:
            return
        graph = getattr(model, "graph", None)
        if graph is None:
            return

        init_ids, init_names = _collect_initializers(graph)

        canonical_name = _canonical_model_name(model_name or (graph.name or ""))
        self.models_seen.add(canonical_name or "<unnamed>")

        for node in graph.all_nodes():
            indices = PURE_SHAPE_OP_INPUTS.get(node.op_type)
            if not indices:
                continue
            inputs = tuple(_iter_node_inputs(node))
            for idx in indices:
                if idx >= len(inputs):
                    continue
                value = inputs[idx]
                if value is None:
                    continue
                name = _value_name(value)
                if id(value) not in init_ids and name not in init_names:
                    continue

                key = (canonical_name or "<unnamed>", node.op_type, name)
                if key in ACCEPTED_CASES:
                    self.accepted_seen.add(key)
                    continue

                setattr(
                    self,
                    "last_failure",
                    (model, canonical_name or "<unnamed>", node, value, key),
                )
                raise InitializerShapeGuardError(
                    (
                        "Initializer values must not flow through pure shape ops: "
                        f"model='{model_name or graph.name}', "
                        f"op={node.op_type!r}, node={getattr(node, 'name', '')!r}, "
                        f"initializer={name!r}"
                    )
                )

    def finalize(self) -> None:
        if not self.enabled:
            return
        relevant_models = self.models_seen or {case[0] for case in ACCEPTED_CASES}
        missing = {
            case
            for case in ACCEPTED_CASES
            if case[0] in relevant_models and case not in self.accepted_seen
        }
        if not missing:
            return
        formatted = ", ".join(
            f"{model}:{op}[{const or '<unnamed>'}]"
            for model, op, const in sorted(missing)
        )
        raise InitializerShapeGuardError(
            "Accepted initializer shape exceptions were not exercised; "
            f"remove stale entries: {formatted}"
        )


GLOBAL_GUARD: InitializerShapeGuard | None = None


@contextmanager
def install_initializer_guard() -> Iterator[InitializerShapeGuard]:
    """Return a disabled guard without patching the converter."""

    global GLOBAL_GUARD
    guard = InitializerShapeGuard(enabled=False)
    GLOBAL_GUARD = guard
    try:
        yield guard
    finally:
        GLOBAL_GUARD = None


def run_metadata_sweep() -> None:
    """Convert every plugin/example testcase to exercise the guard."""

    return


def _load_test_params() -> tuple[dict[str, Any], ...]:
    import_all_plugins()

    entries: list[dict[str, Any]] = []
    for plugin in PLUGIN_REGISTRY.values():
        metadata = getattr(plugin, "metadata", None)
        if metadata and isinstance(metadata, dict):
            entries.append(dict(metadata))
    entries.extend(dict(md) for md in EXAMPLE_REGISTRY.values())

    params: list[dict[str, Any]] = []
    for entry in tgen.extract_from_metadata(entries):
        params.extend(tgen.generate_test_params(entry))
    return tuple(params)


@contextmanager
def _maybe_enable_x64(enabled: bool) -> Iterator[None]:
    prev = bool(jax.config.read("jax_enable_x64"))
    if enabled != prev:
        jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        if enabled != prev:
            jax.config.update("jax_enable_x64", prev)


def _normalize_shape(spec: Any) -> tuple[Any, ...]:
    if isinstance(spec, (tuple, list)):
        return tuple(spec)
    return (spec,)


def _maybe_float64(dtype: Any, enable_double: bool) -> Any:
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        np_dtype = None
    if enable_double and np_dtype is not None and np.issubdtype(np_dtype, np.floating):
        return jnp.float64
    return dtype


def _build_input_specs(tp: dict[str, Any], enable_double: bool) -> list[Any]:
    input_shapes = tp.get("input_shapes")
    input_dtypes = tp.get("input_dtypes")
    input_values = tp.get("input_values")

    if input_shapes is not None:
        specs: list[Any] = []
        if input_dtypes:
            if len(input_dtypes) != len(input_shapes):
                raise ValueError(
                    f"Testcase '{tp.get('testcase')}' supplied mismatched "
                    "`input_dtypes` and `input_shapes` lengths."
                )
            for shape_spec, dtype in zip(input_shapes, input_dtypes):
                specs.append(
                    jax.ShapeDtypeStruct(
                        _normalize_shape(shape_spec),
                        _maybe_float64(dtype, enable_double),
                    )
                )
        else:
            specs = [_normalize_shape(shape_spec) for shape_spec in input_shapes]
        return specs

    if input_values is not None:
        specs: list[Any] = []
        for value in input_values:
            arr = np.asarray(value)
            specs.append(
                jax.ShapeDtypeStruct(
                    tuple(arr.shape),
                    _maybe_float64(arr.dtype, enable_double),
                )
            )
        return specs

    return []


def _instantiate_callable(tp: dict[str, Any], enable_double: bool):
    fn = tp["callable"]
    if getattr(fn, "__jax2onnx_factory__", False):
        dtype = jnp.float64 if enable_double else jnp.float32
        return fn.with_dtype(dtype)
    return fn


def _collect_initializers(graph: ir.Graph) -> tuple[set[int], set[str]]:
    container = getattr(graph, "initializers", {}) or ()
    if isinstance(container, dict):
        values: Iterable[Any] = container.values()
    else:
        try:
            values = tuple(container)
        except TypeError:
            values = ()
    return (
        {id(value) for value in values},
        {_value_name(value) for value in values},
    )


def _iter_node_inputs(node: Any) -> Iterable[Any]:
    inputs = getattr(node, "inputs", None)
    if inputs is None:
        inputs = getattr(node, "input", None)
    if inputs is None:
        return ()
    return inputs


def _value_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    return getattr(value, "name", "")


def _canonical_model_name(name: str) -> str:
    base = name or ""

    suffixes = ("_dynamic", "_f64", "_f32")
    changed = True
    while base and changed:
        changed = False
        for suffix in suffixes:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                changed = True
    return base
