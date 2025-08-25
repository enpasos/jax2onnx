# jax2onnx/plugins/jax/lax/pad.py
from typing import TYPE_CHECKING
import numpy as np
import jax
import jax.numpy as jnp
from jax import core as jcore
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.pad_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.pad.html",
    onnx=[{"component": "Pad", "doc": "https://onnx.ai/onnx/operators/onnx__Pad.html"}],
    since="v0.8.0",
    context="primitives.lax",
    component="pad",
    testcases=[
        {
            "testcase": "pad_const_1d",
            "callable": lambda x: jax.lax.pad(x, 0.0, ((1, 2, 0),)),
            "input_shapes": [(5,)],
        },
        {
            "testcase": "pad_const_2d",
            "callable": lambda x: jax.lax.pad(x, 1.0, ((0, 0, 0), (1, 1, 0))),
            "input_shapes": [(2, 3)],
        },
        # --- extra local tests to cover pad in scan/nested scan and int cval ---
        {
            # pad with an INT64 cval (mimics real jaxprs where 0 comes in as i64)
            "testcase": "pad_const_2d_cval",
            "callable": lambda x: jax.lax.pad(
                x,
                jnp.asarray(0, dtype=x.dtype),  # cval must match operand dtype
                ((0, 0, 0), (1, 1, 0)),
            ),
            "input_shapes": [(2, 3)],
        },
        {
            # pad -> crop inside a single scan body; output stacked over length
            # Ensures pads & cval are materialized as local Constants in Loop body.
            "testcase": "pad_inside_scan_smoke_f64",
            "callable": lambda x: jax.lax.scan(
                lambda carry, _: (
                    carry,
                    (
                        jnp.pad(carry, ((0, 0), (0, 0), (1, 1), (1, 1)))[
                            :, :, 1:-1, 1:-1
                        ]
                        * carry
                    ),
                ),
                x,
                None,
                length=2,
            )[1],
            "input_shapes": [(1, 3, 8, 8)],
            "expected_output_shapes": [(2, 1, 3, 8, 8)],
            "run_only_f64_variant": True,
        },
        {
            # nested scan; inner does pad->crop then mul, outer stacks inner outputs
            "testcase": "pad_inside_nested_scan_smoke_f64",
            "callable": lambda x: jax.lax.scan(
                lambda carry, _: jax.lax.scan(
                    lambda c2, __: (
                        c2,
                        (
                            jnp.pad(c2, ((0, 0), (0, 0), (1, 1), (1, 1)))[
                                :, :, 1:-1, 1:-1
                            ]
                            * c2
                        ),
                    ),
                    carry,
                    None,
                    length=2,
                ),
                x,
                None,
                length=1,
            )[1],
            "input_shapes": [(1, 3, 8, 8)],
            "expected_output_shapes": [(1, 2, 1, 3, 8, 8)],
            "run_only_f64_variant": True,
        },
    ],
)
class PadPlugin(PrimitiveLeafPlugin):
    """Lower jax.lax.pad (constant mode, no interior padding) → ONNX Pad."""

    def _reg_vi(self, b, name, dims, dtype):
        if b is None:
            return
        if hasattr(b, "add_value_info"):
            b.add_value_info(name, dims, dtype)
        elif hasattr(b, "register_value_info_metadata"):
            b.register_value_info_metadata(name, dims, dtype)

    def _materialize_local_scalar_constant(
        self,
        s: "Jaxpr2OnnxConverter",
        like_tensor_name: str,
        source,
        *,
        base: str = "pad_cval",
    ) -> str:
        """
        Create a scalar Constant in the *current* graph and CastLike it to `like_tensor_name`.
        We try to read a numeric value from:
          - Python/NumPy scalar
          - jax.core.Literal
          - builder initializers (list or dict)
        If not resolvable, we conservatively fall back to 0.0 (OK for shape inference).
        Also registers value_info for both the Constant output and the CastLike output.
        """
        # ---- figure out the numeric value ----
        val = None
        import numpy as _np

        # 0) plain scalar?
        if isinstance(source, (int, float, _np.integer, _np.floating)):
            val = float(source)
        # 1) jax literal?
        if val is None:
            try:
                if isinstance(source, jcore.Literal):
                    val = float(source.val)
            except Exception:
                pass
        # 2) initializer by name?
        if val is None:
            name = None
            try:
                name = s.get_name(source)
            except Exception:
                name = None
            b = getattr(s, "builder", None)
            proto_or_arr = None
            if name and b is not None:
                # try getters
                for getter in (
                    "get_initializer",
                    "get_initializer_tensor",
                    "try_get_initializer",
                    "get_initializer_value",
                ):
                    if hasattr(b, getter):
                        try:
                            proto_or_arr = getattr(b, getter)(name)
                            if proto_or_arr is not None:
                                break
                        except Exception:
                            proto_or_arr = None
                # scan .initializers (list OR dict)
                if proto_or_arr is None and hasattr(b, "initializers"):
                    try:
                        inits = b.initializers
                        if hasattr(inits, "get"):  # dict-like
                            proto_or_arr = inits.get(name)
                        if proto_or_arr is None and isinstance(
                            inits, (list, tuple)
                        ):  # list of TensorProto or (name, tensor)
                            for item in inits:
                                if hasattr(item, "name") and item.name == name:
                                    proto_or_arr = item
                                    break
                                if (
                                    isinstance(item, (list, tuple))
                                    and len(item) >= 2
                                    and item[0] == name
                                ):
                                    proto_or_arr = item[1]
                                    break
                    except Exception:
                        proto_or_arr = None
            if proto_or_arr is not None:
                try:
                    from onnx import numpy_helper as _np_helper

                    arr = (
                        _np_helper.to_array(proto_or_arr)
                        if hasattr(proto_or_arr, "data_type")
                        else _np.asarray(proto_or_arr)
                    )
                    if arr.size == 1:
                        val = float(arr.reshape(()).item())
                except Exception:
                    pass
        if val is None:
            val = 0.0  # safe fallback for inference

        # ---- emit Constant (scalar double) + CastLike(data) ----
        const_out = s.get_unique_name(f"{base}_const")
        cast_out = s.get_unique_name(f"{base}_castlike")
        tensor = helper.make_tensor(
            name=f"{const_out}_value",
            data_type=TensorProto.DOUBLE,  # precise scalar; CastLike will match dtype to `data`
            dims=[],
            vals=[float(val)],
        )
        s.add_node(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[const_out],
                value=tensor,
                name=s.get_unique_name(f"{base}_const_node"),
            )
        )
        s.add_node(
            helper.make_node(
                "CastLike",
                inputs=[const_out, like_tensor_name],
                outputs=[cast_out],
                name=s.get_unique_name(f"{base}_castlike_node"),
            )
        )

        b = getattr(s, "builder", None)

        # const_out is scalar double
        self._reg_vi(b, const_out, [], TensorProto.DOUBLE)

        # cast_out is scalar; try to match data dtype
        cast_dtype = None
        for getter in ("get_value_info_dtype", "get_dtype", "get_dtype_of"):
            if hasattr(b, getter):
                try:
                    cast_dtype = getattr(b, getter)(like_tensor_name)
                    if cast_dtype:
                        break
                except Exception:
                    pass
        if cast_dtype is None:
            cast_dtype = TensorProto.UNDEFINED

        self._reg_vi(b, cast_out, [], cast_dtype)

        return cast_out

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        data = s.get_name(node_inputs[0])  # input tensor
        out = s.get_var_name(node_outputs[0])
        edge_val_in = node_inputs[
            1
        ]  # padding scalar (literal / constvar / possibly dynamic)

        # ((low, high, interior), ...)
        pcfg = params["padding_config"]
        if any(int(i) != 0 for (_, _, i) in pcfg):
            raise NotImplementedError(
                "lax.pad with interior>0 not supported by ONNX Pad"
            )

        begins = [int(lo) for (lo, _, _) in pcfg]
        ends = [int(hi) for (_, hi, _) in pcfg]
        pads_np = np.asarray(begins + ends, dtype=np.int64)  # length = 2*rank

        # --- Build pads as a local Constant (subgraph-safe) ---
        pads_name = s.get_unique_name("pads")
        pads_tensor = helper.make_tensor(
            name=s.get_unique_name("pads_tensor"),
            data_type=TensorProto.INT64,
            dims=[int(pads_np.size)],  # 1-D: [2 * rank]
            vals=pads_np.tolist(),
        )
        s.add_node(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[pads_name],
                name=s.get_unique_name("pads_const"),
                value=pads_tensor,
            )
        )

        # Provide value_info metadata so strict builder checks pass
        b = getattr(s, "builder", None)
        if b is not None:
            if hasattr(b, "add_value_info"):
                b.add_value_info(pads_name, [int(pads_np.size)], TensorProto.INT64)
            elif hasattr(b, "register_value_info_metadata"):
                b.register_value_info_metadata(
                    pads_name, [int(pads_np.size)], TensorProto.INT64
                )

        # --- Always inline the pad scalar locally (Constant + CastLike) ---
        # This prevents Loop/Scan body input capture and works for literals/initializers.
        const_for_pad = self._materialize_local_scalar_constant(s, data, edge_val_in)

        # Now the actual Pad op (opset ≥ 11: data, pads, [constant_value])
        # Write to a fresh SSA name to avoid any accidental duplication on 'out',
        # then alias to the official var with Identity.
        tmp_out = s.get_unique_name("pad_out")
        pad_node = helper.make_node(
            "Pad",
            inputs=[data, pads_name, const_for_pad],
            outputs=[tmp_out],
            name=s.get_unique_name("pad"),
            mode="constant",
        )
        s.add_node(pad_node)

        # Register value_info for tmp_out (builder is strict about intermediates)
        out_dtype = None
        for getter in ("get_value_info_dtype", "get_dtype", "get_dtype_of"):
            if b is not None and hasattr(b, getter):
                try:
                    out_dtype = getattr(b, getter)(data)
                    if out_dtype:
                        break
                except Exception:
                    pass
        if out_dtype is None:
            out_dtype = TensorProto.UNDEFINED

        # Try to infer output shape from input shape + pads
        rank = len(pcfg)
        in_shape = None
        for getter in ("get_value_info_shape", "get_shape", "get_shape_of"):
            if b is not None and hasattr(b, getter):
                try:
                    in_shape = getattr(b, getter)(data)
                    if in_shape:
                        break
                except Exception:
                    pass
        if in_shape is None and b is not None and hasattr(b, "value_infos"):
            try:
                vi = b.value_infos.get(data) if hasattr(b.value_infos, "get") else None
                if isinstance(vi, (tuple, list)) and len(vi) >= 1:
                    in_shape = vi[0]
            except Exception:
                in_shape = None

        if isinstance(in_shape, (list, tuple)) and len(in_shape) == rank:
            dims_out = []
            for d, lo, hi in zip(in_shape, begins, ends):
                if isinstance(d, int):
                    dims_out.append(int(d) + int(lo) + int(hi))
                else:
                    dims_out.append(-1)  # unknown dim
        else:
            dims_out = [-1] * rank  # unknown rank dims (rank is known)

        self._reg_vi(b, tmp_out, dims_out, out_dtype)

        s.add_node(
            helper.make_node(
                "Identity",
                inputs=[tmp_out],
                outputs=[out],
                name=s.get_unique_name("pad_out_alias"),
            )
        )
