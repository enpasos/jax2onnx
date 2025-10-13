from __future__ import annotations
from typing import Any, Callable, Dict, Optional
import numpy as np

class ConstantFolder:
    def __init__(self) -> None:
        self._known: Dict[int, np.ndarray] = {}
        self._producer: Dict[int, Any] = {}

    def register_const(self, var: Any, value: np.ndarray) -> None:
        self._known[id(var)] = np.asarray(value)

    def install_producers(self, jaxpr) -> None:
        self._producer.clear()
        for eqn in jaxpr.eqns:
            for out in eqn.outvars:
                self._producer[id(out)] = eqn

    def try_evaluate(self, var: Any, handler: Callable[..., Any]) -> Optional[np.ndarray]:
        vid = id(var)
        if vid in self._known:
            return self._known[vid]

        literal = getattr(var, "val", None)
        if literal is not None:
            arr = np.asarray(literal)
            self._known[vid] = arr
            return arr

        eqn = self._producer.get(vid)
        if eqn is None:
            return None

        inputs: list[np.ndarray] = []
        for invar in eqn.invars:
            val = self.try_evaluate(invar, handler)
            if val is None:
                return None
            inputs.append(val)

        try:
            out = handler(eqn.primitive, *inputs, **eqn.params)
        except Exception:
            return None

        arr = np.asarray(out)
        self._known[vid] = arr
        return arr
