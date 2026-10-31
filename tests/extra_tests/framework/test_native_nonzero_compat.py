# tests/extra_tests/framework/test_native_nonzero_compat.py

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx import allclose, to_onnx
from jax2onnx._compat import jax as compat


@pytest.mark.parametrize(
    ("fn", "inputs"),
    [
        (
            lambda x: jnp.nonzero(x, size=2),
            [np.asarray([1, 0, 2, 3], dtype=np.int32)],
        ),
        (
            lambda x: jnp.nonzero(x, size=4, fill_value=(-1, -2)),
            [np.asarray([[0, 1], [2, 0]], dtype=np.int32)],
        ),
        (
            lambda x, fill: jnp.nonzero(x, size=4, fill_value=fill),
            [np.asarray([0, 1, 0, 2], dtype=np.int32), np.asarray(-3, dtype=np.int32)],
        ),
        (
            lambda x, fills: jax.vmap(
                lambda row, fill: jnp.nonzero(row, size=3, fill_value=fill)[0]
            )(x, fills),
            [
                np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int32),
                np.asarray([-2, -3], dtype=np.int32),
            ],
        ),
        (
            lambda x: jnp.nonzero(x, size=3),
            [np.asarray([[False, True], [True, False]], dtype=np.bool_)],
        ),
        (
            lambda x: jnp.nonzero(x, size=3),
            [np.empty((0, 2), dtype=np.int32)],
        ),
        (
            lambda x: (jnp.nonzero(x, size=2), jnp.nonzero(x, size=3, fill_value=-1)),
            [np.asarray([0, 1, 0, 2], dtype=np.int32)],
        ),
    ],
    ids=[
        "truncate",
        "tuple-fill",
        "dynamic-fill",
        "vmap-fill",
        "bool",
        "empty",
        "repeat",
    ],
)
def test_native_nonzero_preserves_composition_outputs(
    tmp_path: Path, fn: Callable[..., Any], inputs: list[np.ndarray]
) -> None:
    path = tmp_path / "nonzero.onnx"
    to_onnx(fn, inputs=inputs, return_mode="file", output_path=path)
    matches, message = allclose(fn, str(path), inputs, atol=0, rtol=0)
    assert matches, message


def test_nonzero_adapter_rejects_foreign_and_modified_carriers_without_callbacks() -> (
    None
):
    class ForeignCarrier:
        def expand(self, *_args: Any) -> Any:
            raise AssertionError("foreign HiJAX callback ran")

        def __getattribute__(self, name: str) -> Any:
            raise AssertionError("foreign HiJAX attribute hook ran")

    template = jax.make_jaxpr(lambda x: x + x)(
        jax.ShapeDtypeStruct((2,), np.int32)
    ).jaxpr.eqns[0]
    equation = template.replace(
        primitive=compat._NATIVE_HI_CALL or compat.Primitive("call_hi_primitive"),
        params={"_prim": ForeignCarrier()},
    )
    with pytest.raises(NotImplementedError, match="native Nonzero"):
        compat.native_nonzero_expansion(equation)

    # Older supported JAX versions already expose nonzero as its composition.
    # If a native HiJAX carrier is emitted, its exact parameter schema is owned.
    closed = jax.make_jaxpr(lambda x: jnp.nonzero(x, size=2))(
        jax.ShapeDtypeStruct((2,), np.int32)
    )
    for equation in closed.jaxpr.eqns:
        if equation.primitive is not compat._NATIVE_HI_CALL:
            continue
        original = equation.params["_prim"]
        for fault in ("extra-field", "extra-parameter", "noninteger-size"):
            carrier = object.__new__(type(original))
            vars(carrier).update(vars(original))
            if fault == "extra-field":
                vars(carrier)["expand"] = ForeignCarrier()
            else:
                parameters = dict(vars(original)["params"])
                if fault == "extra-parameter":
                    parameters["unknown"] = 1
                else:
                    parameters["size"] = ForeignCarrier()
                vars(carrier)["params"] = parameters
            malformed = equation.replace(params={"_prim": carrier})
            with pytest.raises(TypeError, match="native Nonzero"):
                compat.native_nonzero_expansion(malformed)
