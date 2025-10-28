# tests/extra_tests/loop/test_loop_scatter_payload_extent_regression.py

from __future__ import annotations

import pytest

from typing import Callable, Optional

from jax2onnx.sandbox import issue52_scatter_payload_repro as repro

AXIS_META_KEY = "loop_axis0_override"
STACK_WIDTH = 5


def _value_dims(value) -> tuple:
    shape = getattr(getattr(value, "shape", None), "dims", None)
    if not shape:
        return ()
    dims = []
    for dim in shape:
        if hasattr(dim, "value"):
            dims.append(dim.value)
        else:
            dims.append(dim)
    return tuple(dims)


def _find_value(
    ir_model,
    prefix: str,
    predicate: Optional[Callable] = None,
):
    for node in ir_model.graph.all_nodes():
        for out in node.outputs:
            name = getattr(out, "name", "")
            if not name.startswith(prefix):
                continue
            if predicate is not None and not predicate(out):
                continue
            return out
    raise AssertionError(f"No value starting with '{prefix}' found in IR")


@pytest.mark.filterwarnings("ignore:.*Removing initializer.*:UserWarning")
def test_scatter_payload_propagates_loop_extent():
    if "square" not in repro._primitive_registry():
        pytest.skip("square primitive missing under this JAX build")
    _, ir_model, *_ = repro.export_models(trace_axis0=False)

    loop_out = _find_value(ir_model, "loop_out_")
    concat_out = _find_value(
        ir_model,
        "concat_out_",
        predicate=lambda v: _value_dims(v)[:2] == (2, STACK_WIDTH),
    )
    reshape_out = _find_value(
        ir_model,
        "bcast_reshape_out_",
        predicate=lambda v: _value_dims(v)[:2] == (1, STACK_WIDTH),
    )
    scatter_out = _find_value(
        ir_model,
        "scatter_out_",
        predicate=lambda v: _value_dims(v)[:1] == (STACK_WIDTH,),
    )
    expand_out = _find_value(
        ir_model,
        "bcast_out_",
        predicate=lambda v: _value_dims(v)[:2] == (STACK_WIDTH, STACK_WIDTH),
    )

    assert loop_out.meta.get(AXIS_META_KEY) == STACK_WIDTH
    assert scatter_out.meta.get(AXIS_META_KEY) == STACK_WIDTH
    assert reshape_out.meta.get(AXIS_META_KEY) == STACK_WIDTH
    assert expand_out.meta.get(AXIS_META_KEY) == STACK_WIDTH
    assert concat_out.meta.get(AXIS_META_KEY) == STACK_WIDTH
