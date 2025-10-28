# jax2onnx/user_interface.py

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
    runtime_checkable,
)
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from jax import core

from jax2onnx.converter.conversion_api import (
    InputSpec,
    ShapeDimSpec,
    ShapeTupleSpec,
    to_onnx as to_onnx_impl,
)
from jax2onnx.converter.ir_postprocess import postprocess_ir_model
from jax2onnx.plugins.plugin_system import onnx_function as onnx_function_impl
from jax2onnx.serde_onnx import ir_to_onnx
import onnxruntime
from onnx_ir import Model as IRModel  # type: ignore


ReturnMode = Literal["proto", "ir", "file"]
_VALID_RETURN_MODES = {"proto", "ir", "file"}


PathLikeStr = Union[str, os.PathLike[str]]


def _normalize_return_mode(value: str) -> ReturnMode:
    mode = value.lower().strip()
    if mode not in _VALID_RETURN_MODES:
        raise ValueError(
            f"Unsupported return_mode '{value}'. Expected one of: {sorted(_VALID_RETURN_MODES)}"
        )
    return cast(ReturnMode, mode)


@runtime_checkable
class _SupportsShapeAndDtype(Protocol):
    shape: Tuple[Any, ...]
    dtype: Any


UserInputSpec = Union[
    jax.ShapeDtypeStruct,
    core.ShapedArray,
    _SupportsShapeAndDtype,
    Sequence[ShapeDimSpec],
]


def _normalize_shape_tuple(candidate: Sequence[Any]) -> ShapeTupleSpec:
    dims: list[ShapeDimSpec] = []
    for dim in candidate:
        if isinstance(dim, (int, np.integer)):
            dims.append(int(dim))
        elif isinstance(dim, str):
            dims.append(dim)
        else:
            raise TypeError(
                "Shape tuples may only contain int or str entries; "
                f"received {type(dim)}."
            )
    return tuple(dims)


def _normalize_input_specs(raw_inputs: Sequence[UserInputSpec]) -> List[InputSpec]:
    normalized: List[InputSpec] = []
    for item in raw_inputs:
        if isinstance(item, jax.ShapeDtypeStruct):
            dims = tuple(
                int(dim) if isinstance(dim, np.integer) else dim
                for dim in tuple(item.shape)
            )
            normalized.append(jax.ShapeDtypeStruct(dims, item.dtype))
            continue
        if isinstance(item, core.ShapedArray):
            dims = tuple(
                int(dim) if isinstance(dim, np.integer) else dim for dim in item.shape
            )
            normalized.append(jax.ShapeDtypeStruct(dims, item.dtype))
            continue
        if isinstance(item, _SupportsShapeAndDtype):
            dims = tuple(
                int(dim) if isinstance(dim, np.integer) else dim
                for dim in tuple(item.shape)
            )
            normalized.append(jax.ShapeDtypeStruct(dims, item.dtype))
            continue
        if isinstance(item, (list, tuple)):
            normalized.append(_normalize_shape_tuple(item))
            continue
        raise TypeError(
            "Invalid 'inputs' entry. Expected a jax.ShapeDtypeStruct, "
            "jax.core.ShapedArray, array-like object, or shape tuple/list. "
            f"Received {type(item)}."
        )
    return normalized


@overload
def to_onnx(
    fn: Callable,
    inputs: Sequence[UserInputSpec],
    input_params: Optional[Mapping[str, object]] = ...,
    model_name: str = ...,
    opset: int = ...,
    *,
    enable_double_precision: bool = ...,
    record_primitive_calls_file: Optional[str] = ...,
    return_mode: Literal["proto"] = ...,
    output_path: None = ...,
) -> onnx.ModelProto: ...


@overload
def to_onnx(
    fn: Callable,
    inputs: Sequence[UserInputSpec],
    input_params: Optional[Mapping[str, object]] = ...,
    model_name: str = ...,
    opset: int = ...,
    *,
    enable_double_precision: bool = ...,
    record_primitive_calls_file: Optional[str] = ...,
    return_mode: Literal["ir"],
    output_path: Optional[PathLikeStr] = ...,
) -> IRModel: ...


@overload
def to_onnx(
    fn: Callable,
    inputs: Sequence[UserInputSpec],
    input_params: Optional[Mapping[str, object]] = ...,
    model_name: str = ...,
    opset: int = ...,
    *,
    enable_double_precision: bool = ...,
    record_primitive_calls_file: Optional[str] = ...,
    return_mode: Literal["file"],
    output_path: PathLikeStr,
) -> str: ...


def to_onnx(
    fn: Callable,
    inputs: Sequence[UserInputSpec],
    input_params: Optional[Mapping[str, object]] = None,
    model_name: str = "jax_model",
    opset: int = 21,
    *,  # All arguments after this must be keyword-only
    enable_double_precision: bool = False,
    record_primitive_calls_file: Optional[str] = None,
    return_mode: ReturnMode = "proto",
    output_path: Optional[PathLikeStr] = None,
) -> Union[onnx.ModelProto, IRModel, str]:
    """
    Converts a JAX function or model into an ONNX model.

    This function serves as the main entry point for converting JAX/Flax models to ONNX format.
    It supports dynamic shapes and additional runtime parameters.

    Args:
        fn: The JAX function or Flax module to convert.
        inputs: Sequence of input specifications. Each entry may be:
            * a `jax.ShapeDtypeStruct` (or `jax.core.ShapedArray`);
            * any array-like object exposing `.shape` and `.dtype`
              (e.g. `jax.Array`, `np.ndarray`);
            * a tuple/list of ints/strs describing the desired shape.
        input_params: Optional mapping of string keys to runtime parameters that
            should be exposed as inputs in the ONNX model rather than baked into
            the export (e.g. `"deterministic"` flags).
        model_name: Name to give the ONNX model. Defaults to "jax_model".
        opset: ONNX opset version to target. Defaults to 21.
        enable_double_precision: If True, export tensors as tensor(double). Defaults to False (use tensor(float)).
        record_primitive_calls_file: Optional path to a file. If provided,
            details of each JAX primitive encountered during conversion will be
            recorded to this file. This log can be used by developers to manually
            create new test cases. Defaults to None (disabled).
        return_mode: Output mode. `"proto"` (default) returns an ONNX ModelProto,
            `"ir"` returns the intermediate onnx_ir.Model, and `"file"`
            serialises directly to disk.
        output_path: Destination path (str or PathLike) required when `return_mode` is
            `"file"`. Ignored otherwise.

    Returns:
        Depending on `return_mode`, either an ONNX ModelProto, an onnx_ir.Model,
        or the filesystem path written to when `return_mode="file"`.

    Example:
        >>> import jax.numpy as jnp
        >>> from flax import nnx
        >>> from jax2onnx import to_onnx
        >>>
        >>> model = MyFlaxModel(...)
        >>> onnx_model = to_onnx(model, inputs=[('B', 32, 32, 3)])
        >>> import onnx
        >>> onnx.save(onnx_model, "model.onnx")
    """

    logging.info(
        f"Converting JAX function to ONNX model with parameters: "
        f"model_name={model_name}, opset={opset}, input_shapes={inputs}, "
        f"input_params={input_params}, "
        f"enable_double_precision={enable_double_precision}, "
        f"record_primitive_calls_file={record_primitive_calls_file}, "
        f"return_mode={return_mode}, output_path={output_path}"
    )

    # Determine the nature of the 'inputs' argument to prepare for to_onnx_impl
    normalized_mode = _normalize_return_mode(return_mode)

    file_path: Optional[str] = None
    if normalized_mode == "file":
        if output_path is None:
            raise ValueError(
                "`output_path` must be provided when return_mode is 'file'."
            )
        path_value = os.fspath(output_path)
        if isinstance(path_value, bytes):
            path_value = path_value.decode()
        file_path = cast(str, path_value)

    normalized_inputs: List[InputSpec] = []
    if inputs:
        normalized_inputs = _normalize_input_specs(inputs)

    param_map: Dict[str, object] = {}
    if input_params:
        for key, value in input_params.items():
            if not isinstance(key, str):
                raise TypeError(
                    "input_params must use string keys; "
                    f"received key of type {type(key)}."
                )
            param_map[key] = value

    result = to_onnx_impl(
        fn=fn,
        inputs=normalized_inputs,
        input_params=param_map,
        model_name=model_name,
        opset=opset,
        enable_double_precision=enable_double_precision,
        record_primitive_calls_file=record_primitive_calls_file,
    )

    def _attach_input_params(model_proto: onnx.ModelProto) -> None:
        if not param_map:
            return

        graph = model_proto.graph
        existing_inputs = {vi.name for vi in graph.input}
        provided_names = set(existing_inputs)
        provided_names.update(init.name for init in graph.initializer)
        sparse_inits = getattr(graph, "sparse_initializer", None)
        if sparse_inits is not None:
            provided_names.update(init.name for init in sparse_inits)

        referenced: set[str] = set()
        for node in graph.node:
            for inp_name in node.input:
                if inp_name:
                    referenced.add(inp_name)
        for output in graph.output:
            name = output.name
            if name:
                referenced.add(name)

        for name, value in param_map.items():
            if not name or name in existing_inputs:
                continue
            if name in provided_names:
                continue
            if name not in referenced:
                continue

            arr = np.asarray(value)
            elem_type = None
            try:
                elem_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(arr.dtype))
            except Exception:
                elem_type = None
            if elem_type is None and arr.dtype == np.bool_:
                elem_type = onnx.TensorProto.BOOL
            if elem_type is None:
                continue
            shape = list(arr.shape)
            vi = onnx.helper.make_tensor_value_info(name, elem_type, shape)
            graph.input.extend([vi])
            existing_inputs.add(name)
            provided_names.add(name)

    def _save_model_proto(model_proto: onnx.ModelProto, dest: str) -> str:
        dest_dir = os.path.dirname(dest)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        onnx.save(model_proto, dest)
        return dest

    postprocess_ir_model(
        result,
        promote_to_double=enable_double_precision,
    )
    if normalized_mode == "ir":
        return result

    model_proto = ir_to_onnx(result)
    _attach_input_params(model_proto)
    if normalized_mode == "file":
        assert file_path is not None
        return _save_model_proto(model_proto, file_path)
    return model_proto


def onnx_function(
    target: Optional[Union[Callable, type]] = None,
    *,
    unique: bool = False,
    namespace: Optional[str] = None,
) -> Union[Callable, type]:
    """
    Decorator to mark a function or class as an ONNX function.

    This decorator is used to indicate that a function or class should be converted to
    an ONNX function node when included in a model. It allows the function to be traced
    and exported as a reusable component with its own namespace in the ONNX graph.

    Args:
        target: The target function or class to decorate. When omitted, the decorator
            must be called with parentheses.
        unique: If True, reuse a single ONNX Function definition for all call sites
            that share the same callable type and captured parameters.
        namespace: Custom domain prefix for the emitted FunctionProto. Defaults to
            ``"custom"`` when omitted.

    Returns:
        The decorated function or class with ONNX function capabilities.

    Example:
        >>> from jax2onnx import onnx_function
        >>> from flax import nnx
        >>>
        >>> @onnx_function
        >>> class MLPBlock(nnx.Module):
        >>>     def __init__(self, features, rngs):
        >>>         self.dense = nnx.Linear(features, rngs=rngs)
        >>>         self.activation = nnx.relu
        >>>
        >>>     def __call__(self, x):
        >>>         return self.activation(self.dense(x))
    """

    return onnx_function_impl(target, unique=unique, namespace=namespace)


def allclose(
    fn: Callable,
    onnx_model_path: str,
    inputs: List[Any],
    input_params: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Tuple[bool, str]:
    """
    Checks if JAX and ONNX Runtime outputs remain numerically close.

    Args:
        fn: JAX callable to compare against the exported ONNX model.
        onnx_model_path: Path to a serialized model that ORT can execute.
        inputs: Concrete input arrays (or shape tuples, which will be sampled).
        input_params: Optional keyword arguments applied to both call sites.
        rtol: Relative tolerance for floating-point comparisons.
        atol: Absolute tolerance for floating-point comparisons.

    Returns:
        `(is_match, message)` where `is_match` indicates success and `message`
        provides context when a mismatch occurs.
    """

    logging.info(
        "Comparing JAX and ONNX outputs (path=%s, rtol=%s, atol=%s)",
        onnx_model_path,
        rtol,
        atol,
    )

    def _is_shape(x: Any) -> bool:
        return isinstance(x, (tuple, list)) and all(
            isinstance(dim, (int, str)) for dim in x
        )

    xs: List[Any]
    if all(_is_shape(x) for x in inputs):
        xs = [
            np.random.rand(*[d if isinstance(d, int) else 2 for d in shape]).astype(
                np.float32
            )
            for shape in inputs
        ]
    else:
        xs = list(inputs)

    params = dict(input_params or {})
    return _run_allclose(fn, onnx_model_path, xs, params, rtol=rtol, atol=atol)


def _run_allclose(
    fn: Callable,
    model_path: str,
    xs: List[Any],
    params: Dict[str, Any],
    *,
    rtol: float,
    atol: float,
) -> Tuple[bool, str]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        model_path,
        providers=ort.get_available_providers(),
    )

    ort_inputs = _build_ort_inputs(session, xs, params)
    ort_outputs = session.run(None, ort_inputs)

    jax_args = [_to_jax_array(x) for x in xs]
    jax_kwargs = {k: _to_jax_kwarg(v) for k, v in params.items()}

    try:
        jax_result = fn(*jax_args, **jax_kwargs)
    except Exception as exc:  # pragma: no cover - defensive aid
        return False, f"Failed to evaluate JAX function: {exc}"

    jax_host = jax.device_get(jax_result)
    jax_flat, _ = jax.tree_util.tree_flatten(jax_host)
    jax_outputs = [_to_numpy_output(val) for val in jax_flat]

    if len(jax_outputs) != len(ort_outputs):
        return (
            False,
            f"Output count mismatch (JAX={len(jax_outputs)} vs ORT={len(ort_outputs)})",
        )

    for idx, (expected, got) in enumerate(zip(jax_outputs, ort_outputs)):
        expected_arr = np.asarray(expected)
        got_arr = np.asarray(got)

        if expected_arr.shape != got_arr.shape:
            return (
                False,
                "Output {} shape mismatch (JAX {} vs ORT {})".format(
                    idx, expected_arr.shape, got_arr.shape
                ),
            )

        if _is_floating_dtype(expected_arr) or _is_floating_dtype(got_arr):
            if not np.allclose(
                expected_arr,
                got_arr.astype(expected_arr.dtype, copy=False),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                diff = np.abs(expected_arr - got_arr)
                max_diff = float(diff.max()) if diff.size else 0.0
                return (
                    False,
                    f"Output {idx} mismatch (max abs diff {max_diff}, rtol={rtol}, atol={atol})",
                )
        else:
            if not np.array_equal(
                expected_arr, got_arr.astype(expected_arr.dtype, copy=False)
            ):
                return (False, f"Output {idx} mismatch (non-floating tensors differ)")

    return True, "Outputs match within tolerance."


def _build_ort_inputs(
    session: "onnxruntime.InferenceSession",
    xs: List[Any],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    feed: Dict[str, np.ndarray] = {}
    positional_iter = iter(xs)

    for input_meta in session.get_inputs():
        name = input_meta.name
        if name in params:
            feed[name] = _to_numpy_input(params[name], input_meta)
        else:
            try:
                value = next(positional_iter)
            except StopIteration as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Not enough positional inputs provided for ORT (missing value for '{name}')"
                ) from exc
            feed[name] = _to_numpy_input(value, input_meta)

    try:
        next(positional_iter)
    except StopIteration:
        pass
    else:  # pragma: no cover - defensive
        raise ValueError("Too many positional inputs supplied for ORT session")

    return feed


def _to_numpy_input(value: Any, input_meta: Any) -> np.ndarray:
    arr = np.asarray(value)
    expected_dtype = getattr(input_meta, "type", None)

    if isinstance(expected_dtype, str) and expected_dtype.startswith("tensor("):
        dtype_name = expected_dtype[len("tensor(") : -1]
        dtype_map = {
            "float": np.float32,
            "double": np.float64,
            "float16": np.float16,
            "bf16": np.float16,
            "int64": np.int64,
            "int32": np.int32,
            "int16": np.int16,
            "int8": np.int8,
            "uint8": np.uint8,
            "bool": np.bool_,
        }
        target_dtype = dtype_map.get(dtype_name)
        if target_dtype is not None and arr.dtype != target_dtype:
            arr = arr.astype(target_dtype)

    return arr


def _to_jax_array(value: Any) -> jnp.ndarray:
    if isinstance(value, jnp.ndarray):
        return value
    return jnp.asarray(value)


def _to_jax_kwarg(value: Any) -> Any:
    if isinstance(value, (jnp.ndarray, np.ndarray)):
        return jnp.asarray(value)
    if isinstance(value, (list, tuple)):
        return jnp.asarray(value)
    return value


def _to_numpy_output(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, jnp.ndarray):
        return np.asarray(value)
    return np.asarray(value)


def _is_floating_dtype(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.floating)
