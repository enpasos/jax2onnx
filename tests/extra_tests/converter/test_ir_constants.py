# tests/extra_tests/converter/test_ir_constants.py

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from jax2onnx.converter.ir_constants import ConstantFolder


def _eqn(
    *,
    primitive_name: str,
    invars: list[object],
    outvars: list[object],
    params: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        primitive=SimpleNamespace(name=primitive_name),
        invars=invars,
        outvars=outvars,
        params=params or {},
    )


def test_constant_folder_caches_multi_output_results_by_outvar_index() -> None:
    folder = ConstantFolder()
    invar = object()
    first_out = object()
    second_out = object()
    jaxpr = SimpleNamespace(
        eqns=[
            _eqn(
                primitive_name="multi",
                invars=[invar],
                outvars=[first_out, second_out],
                params={"offset": 3},
            )
        ]
    )
    calls = 0

    def _handler(value: np.ndarray[Any, np.dtype[Any]], *, offset: int) -> tuple[
        np.ndarray[Any, np.dtype[Any]],
        np.ndarray[Any, np.dtype[Any]],
    ]:
        nonlocal calls
        calls += 1
        return value + offset, value - offset

    folder.register_const(invar, np.asarray([1, 2], dtype=np.int64))
    folder.register_handler("multi", _handler)
    folder.install_producers(jaxpr)

    np.testing.assert_array_equal(
        folder.try_evaluate(second_out),
        np.asarray([-2, -1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        folder.try_evaluate(first_out),
        np.asarray([4, 5], dtype=np.int64),
    )
    assert calls == 1


def test_constant_folder_rejects_multi_output_handler_arity_mismatch() -> None:
    folder = ConstantFolder()
    invar = object()
    first_out = object()
    second_out = object()
    jaxpr = SimpleNamespace(
        eqns=[
            _eqn(
                primitive_name="bad_multi",
                invars=[invar],
                outvars=[first_out, second_out],
            )
        ]
    )

    def _handler(
        value: np.ndarray[Any, np.dtype[Any]],
    ) -> tuple[np.ndarray[Any, np.dtype[Any]]]:
        return (value,)

    folder.register_const(invar, np.asarray([1, 2], dtype=np.int64))
    folder.register_handler("bad_multi", _handler)
    folder.install_producers(jaxpr)

    assert folder.try_evaluate(second_out) is None
    assert folder.try_evaluate(first_out) is None


def test_constant_folder_producer_scope_restores_previous_producers() -> None:
    folder = ConstantFolder()
    outer_out = object()
    inner_out = object()
    outer_jaxpr = SimpleNamespace(
        eqns=[
            _eqn(
                primitive_name="outer",
                invars=[],
                outvars=[outer_out],
            )
        ]
    )
    inner_jaxpr = SimpleNamespace(
        eqns=[
            _eqn(
                primitive_name="inner",
                invars=[],
                outvars=[inner_out],
            )
        ]
    )

    with folder.producer_scope(outer_jaxpr):
        assert id(outer_out) in folder._producer
        assert id(inner_out) not in folder._producer

        with folder.producer_scope(inner_jaxpr):
            assert id(inner_out) in folder._producer
            assert id(outer_out) not in folder._producer

        assert id(outer_out) in folder._producer
        assert id(inner_out) not in folder._producer

    assert folder._producer == {}
