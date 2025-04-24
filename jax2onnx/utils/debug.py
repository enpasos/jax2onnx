import logging, os

# enable via:  JAX2ONNX_DEBUG_SYMBOLICS=1  (any truthy value)
SYMBOLIC_DEBUG = bool(int(os.getenv("JAX2ONNX_DEBUG_SYMBOLICS", "0")))


def sdebug(msg, *a, **kw):
    if SYMBOLIC_DEBUG:
        logging.getLogger("jax2onnx.symbolic").debug(msg, *a, **kw)


__all__ = ["SYMBOLIC_DEBUG", "sdebug"]
