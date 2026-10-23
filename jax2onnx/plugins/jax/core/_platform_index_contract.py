# jax2onnx/plugins/jax/core/_platform_index_contract.py

"""Pure contract for the native Mosaic/default selector in ordinary lowering."""

from __future__ import annotations

from typing import Any

import numpy as np


def is_mosaic_default_platforms(value: Any) -> bool:
    return (
        type(value) is tuple
        and len(value) == 2
        and type(value[0]) is tuple
        and len(value[0]) == 1
        and type(value[0][0]) is str
        and value[0][0] == "mosaic"
        and value[1] is None
    )


def mosaic_default_platform_index(eqn: Any) -> int:
    """Validate the exact native selector and return its ordinary-domain index.

    Mosaic is a Pallas compiler domain with a separate native lowering rule.
    This converter does not lower Pallas/Mosaic programs. For this exact pair,
    ordinary CPU, CUDA, ROCm and TPU lowering all select default index 1.
    Device-dependent selectors need explicit shared platform provenance; the
    export host's current backend cannot determine their execution semantics.
    """
    if (
        type(eqn.params) is not dict
        or set(eqn.params) != {"platforms"}
        or not is_mosaic_default_platforms(eqn.params["platforms"])
    ):
        raise NotImplementedError(
            "platform_index supports only the native Mosaic/default selector "
            "in ordinary ONNX lowering; device dispatch requires explicit "
            "execution-platform provenance"
        )
    if eqn.invars or len(eqn.outvars) != 1 or eqn.effects:
        raise ValueError("platform_index requires no inputs, one output and no effects")
    aval = eqn.outvars[0].aval
    if (
        aval.shape != ()
        or np.dtype(aval.dtype) != np.dtype(np.int32)
        or aval.weak_type is not False
    ):
        raise ValueError("platform_index requires a strong scalar int32 output")
    return 1
