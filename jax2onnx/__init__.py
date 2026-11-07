# jax2onnx/__init__.py

from jax2onnx.initializer_metadata import InitializerMetadataError  # noqa: F401
from jax2onnx.parameter_metadata import (  # noqa: F401
    ParameterMetadata,
    annotate_parameters,
    read_parameter_metadata,
)
from jax2onnx.user_interface import (  # noqa: F401
    to_onnx,
    onnx_function,
    allclose,
    allclose_onnxruntime_web,
)
