# jax2onnx/converter/conversion_api.py

# Add Tuple, Union to imports if not already present
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Union,
    Tuple,
    Set,
)
import onnx
from onnx import onnx_ml_pb2 as om
import logging
import numpy as np
import jax
from onnx import helper, mapping
from jax2onnx.converter.dynamic_utils import (
    _create_symbolic_input_avals,
)  # Import the helper
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from jax2onnx.converter.name_generator import UniqueNameGenerator
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.optimize_onnx_graph import improve_onnx_model
from jax import config as jax_config  # NEW
import jax.numpy as jnp

logger = logging.getLogger("jax2onnx.converter.conversion_api")


# Remove or comment out the old prepare_example_args if no longer needed
# def prepare_example_args(...): ...


# ------------------------------------------------------------------
# Promote items passed via *input_params* to proper graph inputs
# ------------------------------------------------------------------
def _elem_type_from_numpy(arr: np.ndarray) -> int:
    return mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]


def _promote_params_to_inputs(model: onnx.ModelProto, params: Dict[str, Any] | None):
    if not params:
        return

    for name, value in params.items():
        # ① drop initializer (if any)
        kept = [init for init in model.graph.initializer if init.name != name]
        model.graph.ClearField("initializer")
        model.graph.initializer.extend(kept)

        # ② drop stale value_info (INT32 in our case)
        kept = [vi for vi in model.graph.value_info if vi.name != name]
        model.graph.ClearField("value_info")
        model.graph.value_info.extend(kept)

        # ③ add graph input once
        if any(inp.name == name for inp in model.graph.input):
            continue
        dtype = _elem_type_from_numpy(np.asarray(value))
        vi = helper.make_tensor_value_info(name, dtype, [])  # scalar
        model.graph.input.append(vi)


# -----------------------------------------------------------------------------
# drop duplicate initializers for parameters promoted to real graph inputs
# -----------------------------------------------------------------------------
def _strip_param_initializers(model, input_params):
    if not input_params:
        return
    param_names = set(input_params)
    keep = [init for init in model.graph.initializer if init.name not in param_names]
    del model.graph.initializer[:]  # in‑place update
    model.graph.initializer.extend(keep)


# Type alias for input specs
InputSpec = Union[
    "jax.ShapeDtypeStruct",
    np.ndarray,
    jnp.ndarray,
    Tuple[Union[int, str], ...],
    Sequence[Union[int, str]],
]


def to_onnx(
    fn: Any,
    inputs: Sequence[InputSpec],
    input_params: Dict[str, Any] | None = None,
    model_name: str = "jax_model",
    opset: int = 21,
    *,
    enable_double_precision: bool = False,
    loosen_internal_shapes: bool = False,
    default_dtype: Any | None = None,
    record_primitive_calls_file: Optional[str] = None,
    # ... other parameters ...
) -> onnx.ModelProto:
    """
    Converts a JAX function into an ONNX model.
    Handles symbolic dimensions specified as strings in input shapes.

    Parameters:
    -----------
    fn : Any
        The JAX function to convert to ONNX.
    inputs : Sequence[Sequence[Union[int, str]]]
        Shapes of the input tensors. String values represent symbolic dimensions.
    input_params : Dict[str, Any] | None, optional
        Additional parameters to be passed to the function, by default None
    model_name : str, optional
        Name of the ONNX model, by default "jax_model"
    opset : int, optional
        ONNX opset version to target, by default 21
    enable_double_precision : bool, optional
        If **True**, the converter keeps every tensor in double
        precision (`tensor(double)`).  If **False** (default) the
        graph is down-cast to single precision.
    loosen_internal_shapes : bool, optional
        If True, allow plugins to relax internal subgraph
        value_info to rank-only (clear concrete dim_value/param) to keep ORT
        from re-tightening Loop/Scan body shapes, by default False
    default_dtype : Any | None, optional
        Default data type for inputs if not specified, by default None

    Returns:
    --------
    onnx.ModelProto
        The converted ONNX model
    """
    logger.info(f"Starting JAX to ONNX conversion for '{model_name}'")
    logger.debug(f"Received raw inputs (shapes): {inputs}")
    logger.debug(
        f"Received input_params: {input_params.keys() if input_params else 'None'}"
    )

    # ------------------------------------------------------------------
    # 1) Decide the working dtype, 2) flip JAX's global x64 switch
    #    before we trace the function.  Needs to happen **before**
    #    any array creation inside this call.
    # ------------------------------------------------------------------
    if enable_double_precision:
        jax_config.update("jax_enable_x64", True)
        # If enable_double_precision is True, working_dtype MUST be float64,
        # unless default_dtype is explicitly something else (which might be an edge case to clarify or restrict)
        working_dtype = (
            jnp.float64
        )  # Prioritize float64 if enable_double_precision is true
        if default_dtype is not None and default_dtype != jnp.float64:
            logger.warning(
                f"enable_double_precision is True, but default_dtype is {default_dtype}. Using jnp.float64."
            )
    else:
        jax_config.update("jax_enable_x64", False)
        working_dtype = jnp.float32 if default_dtype is None else default_dtype

    logger.debug(
        f"🔧 enable_double_precision = {enable_double_precision} → working dtype = {working_dtype}"
    )

    # --- Step 0: Format input_specs ---
    # build symbolic input avals — accept shapes, Array-like, or ShapeDtypeStruct
    from jax import ShapeDtypeStruct

    normalized_specs = []
    for spec in inputs:
        if isinstance(spec, ShapeDtypeStruct):
            # already has shape & dtype
            normalized_specs.append((spec.shape, spec.dtype))
        elif hasattr(spec, "shape") and hasattr(spec, "dtype"):
            # real JAX/NumPy array
            normalized_specs.append((tuple(spec.shape), spec.dtype))
        elif isinstance(spec, (tuple, list)):
            # plain shape tuple/list → assume working_dtype
            normalized_specs.append((tuple(spec), working_dtype))
        else:
            raise TypeError(
                f"Unsupported inputs element: {type(spec)}. "
                "Must be shape tuple, Array, or ShapeDtypeStruct."
            )

    logger.debug(f"Normalized input_specs: {normalized_specs}")

    # --- Step 1: Prepare Abstract Inputs with Symbolic Dimensions ---
    # (Assumes this part is now correct)
    symbolic_avals, var_to_symbol_map = _create_symbolic_input_avals(normalized_specs)

    # --- Setup Converter and Builder ---
    unique_name_generator = UniqueNameGenerator()

    # Initialize OnnxBuilder with the enable_double_precision flag
    builder = OnnxBuilder(
        unique_name_generator,
        opset=opset,
        converter=None,  # Will be set later
        enable_double_precision=enable_double_precision,  # Pass the flag
    )

    # Set the map as an attribute *after* initialization
    builder.var_to_symbol_map = var_to_symbol_map
    logger.debug(f"Set builder.var_to_symbol_map: {builder.var_to_symbol_map}")

    # Initialize Converter and link back (no change here)
    converter = Jaxpr2OnnxConverter(
        builder,
        record_primitive_calls_file=record_primitive_calls_file,
        function_context_for_recording=getattr(fn, "__name__", model_name),
    )
    builder.converter = converter

    # Propagate the knob so subgraph builders can read it (optional today; no-ops if unused).
    if hasattr(converter, "builder"):
        setattr(
            converter.builder, "loosen_internal_shapes", bool(loosen_internal_shapes)
        )

    converter.call_params = input_params or {}

    # --- Step 2: Trace the function using Symbolic Avals ---
    # Reminder: converter.trace_jaxpr needs modification next
    logger.info("Initiating JAX tracing with symbolic abstract values...")
    # *** NEXT STEP: Modify converter.trace_jaxpr to accept symbolic_avals ***
    converter.trace_jaxpr(fn, symbolic_avals, params=input_params)
    logger.info("JAX tracing finished.")

    # --- Step 3: Build and Optimize ONNX model ---
    logger.info("Building ONNX model...")
    builder.filter_unused_initializers()
    model = builder.create_onnx_model(model_name)

    # Replace with the new function for properly handling parameter inputs
    _promote_params_to_inputs(
        model, input_params
    )  # ← new instead of _strip_param_initializers

    logger.info("Optimizing ONNX model...")
    model = improve_onnx_model(model)

    # Ensure compatibility with the ONNX Runtime version being used for testing.
    # If the 'onnx' library (used for model creation) defaults to an IR version
    # higher than what the testing 'onnxruntime' supports (e.g., IRv11 vs max IRv10),
    # explicitly set the model's IR version to a compatible one.
    # For onnxruntime 1.18.0, the max supported IR version is 10.
    # Opset 21 (often used by jax2onnx) should correspond to IR version 10
    # according to ONNX specifications (see onnx.helper.VERSION_TABLE).
    # However, the `onnx` library might default to a newer IR version based on its own release.

    # Target IR version for compatibility with onnxruntime that supports up to IR version 10
    target_ir_version = 10
    if model.ir_version > target_ir_version:
        logger.info(
            f"Current model IR version is {model.ir_version}. "
            f"Setting IR version to {target_ir_version} for compatibility "
            f"with an ONNX Runtime that supports up to IR version {target_ir_version}."
        )
        model.ir_version = target_ir_version

    logger.info("ONNX model conversion complete.")
    logger.debug(onnx.helper.printable_graph(model.graph))

    # if primitive-call recording was enabled, flush the log to disk
    if record_primitive_calls_file and hasattr(converter, "recorded_calls_log"):
        from jax2onnx.utils.debug import save_primitive_calls_log

        getattr(fn, "__name__", model_name)
        # honor exactly the path the user passed in
        save_primitive_calls_log(
            converter.recorded_calls_log,
            record_primitive_calls_file,
        )

    # Post-optimization sanitizer: shape inference may have re-added concrete dims
    # inside Loop/Scan bodies. If requested, walk subgraphs and relax VIs again.
    if bool(loosen_internal_shapes):
        _relax_internal_value_infos_in_subgraphs(model)

    return model


def _relax_internal_value_infos_in_subgraphs(model: onnx.ModelProto) -> None:
    """
    Walk all subgraphs (Loop/Scan/If, recursively) and:
      1) Make *internal* value_info rank-only (keep rank; clear dim_value/dim_param).
      2) DO NOT touch subgraph inputs/outputs (preserves Loop control scalar shapes).
      3) Guard elementwise broadcasts by inserting Shape(ref) → Expand(other, shape) on:
         - binary ops: Mul, Add, Sub, Div, Pow
         - variadic:   Sum (expand all non-ref operands)
    """
    logger = logging.getLogger("jax2onnx.relax_and_align")

    # ---------- helpers ----------
    def _rank_only_type(tp: om.TypeProto) -> om.TypeProto:
        if not tp.HasField("tensor_type"):
            return tp
        new_tp = om.TypeProto()
        new_tp.tensor_type.elem_type = tp.tensor_type.elem_type
        if tp.tensor_type.HasField("shape"):
            shp = new_tp.tensor_type.shape
            for _ in tp.tensor_type.shape.dim:
                shp.dim.add()  # keep rank, drop concrete dims
        return new_tp

    def _collect_used_names(g: onnx.GraphProto):
        used_nodes: Set[str] = {n.name for n in g.node if n.name}
        used_tensors: Set[str] = set()
        for n in g.node:
            used_tensors.update(t for t in n.input if t)
            used_tensors.update(t for t in n.output if t)
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            used_tensors.add(vi.name)
        for init in g.initializer:
            used_tensors.add(init.name)
        for n in g.node:
            if n.op_type == "Constant" and n.output:
                used_tensors.update(n.output)
        return used_nodes, used_tensors

    def _fresh(base: str, used: set[str]) -> str:
        base = base or "n"
        name, k = base, 1
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        return name

    def _constant_like_sets(g: onnx.GraphProto):
        init_names = {i.name for i in g.initializer}
        const_outs = set()
        for n in g.node:
            if n.op_type == "Constant" and n.output:
                const_outs.update(n.output)
        return init_names, const_outs

    def _is_constant_like(
        name: str, init_names: set[str], const_outs: set[str]
    ) -> bool:
        return name in init_names or name in const_outs

    def _rank_lookup(g: onnx.GraphProto) -> dict[str, int]:
        ranks: dict[str, int] = {}
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField(
                "shape"
            ):
                ranks[vi.name] = len(vi.type.tensor_type.shape.dim)
        for init in g.initializer:
            ranks[init.name] = len(list(init.dims))
        for n in g.node:
            if n.op_type == "Constant" and n.output:
                for a in n.attribute:
                    if (
                        a.name == "value"
                        and a.type == onnx.AttributeProto.TENSOR
                        and a.t is not None
                    ):
                        ranks[n.output[0]] = len(list(a.t.dims))
        return ranks

    def _relax_internals(g: onnx.GraphProto) -> None:
        for vi in g.value_info:
            if vi.type.HasField("tensor_type"):
                vi.type.CopyFrom(_rank_only_type(vi.type))

    def _align_elementwise(g: onnx.GraphProto) -> tuple[int, int]:
        BIN = {"Mul", "Add", "Sub", "Div", "Pow"}
        VAR = {"Sum"}
        used_node_names, used_tensor_names = _collect_used_names(g)
        init_names, const_outs = _constant_like_sets(g)
        ranks = _rank_lookup(g)
        rewrites = 0
        expands = 0

        new_nodes = []
        for n in g.node:
            if (
                n.op_type not in BIN | VAR
                or not n.input
                or n.name.endswith("__expanded")
            ):
                new_nodes.append(n)
                continue

            ins = [i for i in n.input if i]
            if n.op_type in BIN and len(ins) != 2:
                new_nodes.append(n)
                continue
            if n.op_type in VAR and len(ins) < 2:
                new_nodes.append(n)
                continue

            # choose ref: highest known rank, prefer non-constants
            idxs = list(range(len(ins)))
            nonc = [
                i for i in idxs if not _is_constant_like(ins[i], init_names, const_outs)
            ]
            order = nonc if nonc else idxs
            ref_idx, best = order[0], -1
            for i in order:
                r = ranks.get(ins[i], -1)
                if r > best:
                    best = r
                    ref_idx = i
            ref = ins[ref_idx]

            shape_out = _fresh(f"{(n.name or n.op_type)}__shape", used_tensor_names)
            shape_node = helper.make_node(
                "Shape",
                inputs=[ref],
                outputs=[shape_out],
                name=_fresh(f"{(n.name or n.op_type)}__shape", used_node_names),
            )
            new_nodes.append(shape_node)

            new_inputs = list(ins)
            for i, t in enumerate(ins):
                if i == ref_idx:
                    continue
                exp_out = _fresh(f"{t}__exp", used_tensor_names)
                exp = helper.make_node(
                    "Expand",
                    inputs=[t, shape_out],
                    outputs=[exp_out],
                    name=_fresh(f"{(n.name or n.op_type)}__exp{i}", used_node_names),
                )
                new_nodes.append(exp)
                new_inputs[i] = exp_out
                expands += 1

            new_op = helper.make_node(
                n.op_type,
                inputs=new_inputs,
                outputs=list(n.output),
                name=(
                    (n.name + "__expanded")
                    if n.name
                    else _fresh(f"{n.op_type}__expanded", used_node_names)
                ),
            )
            for a in n.attribute:
                new_op.attribute.extend([a])
            new_nodes.append(new_op)
            rewrites += 1

        del g.node[:]
        g.node.extend(new_nodes)
        return rewrites, expands

    def _post_assertions(g: onnx.GraphProto) -> None:
        import os as _os

        if _os.environ.get("JAX2ONNX_STRICT_BROADCAST_ASSERT", "0") != "1":
            return
        BIN = {"Mul", "Add", "Sub", "Div", "Pow"}
        VAR = {"Sum"}
        prod = {}
        for n in g.node:
            for o in n.output:
                prod[o] = n.op_type
        ranks = _rank_lookup(g)
        for n in g.node:
            if n.op_type not in BIN | VAR:
                continue
            ins = [i for i in n.input if i]
            if len(ins) < 2:
                continue
            ref = max(ins, key=lambda t: ranks.get(t, -1))
            for t in ins:
                if t == ref:
                    continue
                rr, rt = ranks.get(ref, -1), ranks.get(t, -1)
                if rr >= 0 and rt >= 0 and rr != rt:
                    assert (
                        prod.get(t) == "Expand"
                    ), f"Elementwise '{n.name}' expects Expand on non-ref input '{t}'"

    def _walk_graph(g: onnx.GraphProto) -> tuple[int, int]:
        _relax_internals(g)  # <— keep internal VIs rank-only
        rw, ex = _align_elementwise(g)  # <— make broadcasting explicit
        for n in g.node:  # recurse into ALL subgraphs
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    sub = onnx.helper.get_attribute_value(a)
                    srw, sex = _walk_graph(sub)
                    rw += srw
                    ex += sex
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for sub in onnx.helper.get_attribute_value(a):
                        srw, sex = _walk_graph(sub)
                        rw += srw
                        ex += sex
        _post_assertions(g)
        return rw, ex

    total_rw, total_ex = _walk_graph(model.graph)
    logger.info(f"[relax_and_align] rewrites={total_rw}, expands_inserted={total_ex}")


def analyze_constants(model: onnx.ModelProto) -> None:
    logger.info("🔍 Constant Analysis Report (top-level graph)")
    g = model.graph
    graph_inputs = {i.name for i in g.input}
    initializers = {i.name for i in g.initializer}
    const_nodes = {n.output[0]: n for n in g.node if n.op_type == "Constant"}
    function_names = {f.name for f in model.functions}

    logger.info("📦 Inputs: %s", sorted(graph_inputs))
    logger.info("🧊 Initializers: %s", sorted(initializers))
    logger.info("🧱 Constant nodes: %s", sorted(const_nodes))

    for node in g.node:
        if node.op_type in function_names:
            logger.info("▶ Function call: %s", node.op_type)
            for inp in node.input:
                if inp in initializers:
                    style = "initializer"
                elif inp in graph_inputs:
                    style = "graph input"
                elif inp in const_nodes:
                    style = "constant node"
                else:
                    style = "intermediate"
                logger.info("   - %s → %s", inp, style)
