# jax2onnx/converter/ir_constants.py

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, cast
import numpy as np
from numpy.typing import NDArray


class ConstantFolder:
    def __init__(self) -> None:
        self._known: Dict[int, NDArray[np.generic]] = {}
        self._producer: Dict[int, tuple[Any, int]] = {}
        self._handlers: Dict[str, Callable[..., Any]] = {}

    def register_const(self, var: Any, value: np.ndarray) -> None:
        arr = np.asarray(value)
        self._known[id(var)] = cast(NDArray[np.generic], arr)

    def install_producers(self, jaxpr: Any) -> None:
        self._producer.clear()
        for eqn in jaxpr.eqns:
            for output_index, out in enumerate(eqn.outvars):
                self._producer[id(out)] = (eqn, output_index)

    def register_handler(
        self, primitive_name: str, handler: Callable[..., Any]
    ) -> None:
        self._handlers[str(primitive_name)] = handler

    def try_evaluate(self, var: Any) -> Optional[NDArray[np.generic]]:
        vid = id(var)
        if vid in self._known:
            return self._known[vid]

        literal = getattr(var, "val", None)
        if literal is not None:
            arr = np.asarray(literal)
            cast_arr = cast(NDArray[np.generic], arr)
            self._known[vid] = cast_arr
            return cast_arr

        producer = self._producer.get(vid)
        if producer is None:
            return None
        eqn, output_index = producer

        handler = self._handlers.get(eqn.primitive.name)
        if handler is None:
            return None

        inputs: list[NDArray[np.generic]] = []
        for invar in eqn.invars:
            val = self.try_evaluate(invar)
            if val is None:
                return None
            inputs.append(val)

        try:
            out = handler(*inputs, **eqn.params)
        except Exception:
            return None

        output_count = len(eqn.outvars)
        if output_count == 1:
            outputs = (out,)
        elif isinstance(out, (tuple, list)):
            outputs = tuple(out)
            if len(outputs) != output_count:
                return None
        else:
            return None

        for outvar, raw_output in zip(eqn.outvars, outputs):
            arr = np.asarray(raw_output)
            self._known[id(outvar)] = cast(NDArray[np.generic], arr)
        return self._known.get(id(eqn.outvars[output_index]))
