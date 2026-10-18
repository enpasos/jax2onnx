# jax2onnx/converter/precision_policy.py

"""Private conversion policy; the public exporter retains its legacy rules."""

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from collections.abc import Iterator

from jax2onnx._compat.jax import export_trace_scope


class PrecisionPolicy(Enum):
    LEGACY = "legacy"
    SOURCE = "source"


_ACTIVE_PRECISION_POLICY: ContextVar[PrecisionPolicy] = ContextVar(
    "jax2onnx_precision_policy", default=PrecisionPolicy.LEGACY
)


def active_precision_policy() -> PrecisionPolicy:
    """Policy for plugin abstract evaluation, which has no lowering context."""
    return _ACTIVE_PRECISION_POLICY.get()


@contextmanager
def precision_policy_scope(policy: PrecisionPolicy) -> Iterator[None]:
    token = _ACTIVE_PRECISION_POLICY.set(policy)
    try:
        with export_trace_scope(policy.value):
            yield
    finally:
        _ACTIVE_PRECISION_POLICY.reset(token)
