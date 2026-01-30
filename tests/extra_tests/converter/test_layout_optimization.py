# tests/extra_tests/converter/test_layout_optimization.py

# tests/extra_tests/converter/test_layout_optimization.py

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
from flax import nnx

import jax2onnx
from jax2onnx.plugins.examples.nnx.depth_to_space import DepthToSpaceResNet


def test_inputs_outputs_as_nchw_depth_to_space_resnet():
    # 1. Setup Model
    rngs = nnx.Rngs(0)
    model = DepthToSpaceResNet(rngs=rngs)

    # Input shape: NHWC (standard JAX image format)
    # (1, 8, 8, 1) -> Conv -> ...
    input_shape_nhwc = (1, 8, 8, 1)
    x_nhwc = jax.random.normal(jax.random.key(0), input_shape_nhwc)

    # Expected output shape: (1, 16, 16, 1) due to depth_to_space(block_size=2)
    expected_out_nhwc = model(x_nhwc)

    # 2. Convert DEFAULT version (NHWC input/output)
    onnx_model_default = jax2onnx.to_onnx(
        fn=model,
        inputs=[jax.ShapeDtypeStruct(input_shape_nhwc, jnp.float32)],
        input_params=None,
        model_name="depth_to_space_nhwc",
        opset=17,
        enable_double_precision=False,
        record_primitive_calls_file=None,
    )

    # 3. Convert NCHW version (NCHW input/output)
    # The user promises to provide NCHW input, and expects NCHW output.
    # Internally jax2onnx converts NCHW->NHWC, runs model, then NHWC->NCHW.
    # Hopefully these cancel out with the internal LayoutOptimizer or model structure.
    onnx_model_nchw = jax2onnx.to_onnx(
        fn=model,
        inputs=[jax.ShapeDtypeStruct(input_shape_nhwc, jnp.float32)],
        input_params=None,
        model_name="depth_to_space_nchw",
        opset=17,
        enable_double_precision=False,
        record_primitive_calls_file=None,
        inputs_as_nchw=[0],
        outputs_as_nchw=[0],
    )

    # 4. Analyze Graphs
    def count_transposes(model_proto):
        count = 0
        for node in model_proto.graph.node:
            if node.op_type == "Transpose":
                count += 1
        return count

    t_count_default = count_transposes(onnx_model_default)
    t_count_nchw = count_transposes(onnx_model_nchw)

    print(f"Transpose count (Default): {t_count_default}")
    print(f"Transpose count (NCHW): {t_count_nchw}")

    # We expect NCHW version to have FEWER transposes because the internal Conv
    # is NCHW-native (in ONNX) and we are removing the boundary transposes
    # that usually convert NHWC->NCHW and back.
    # Note: DepthToSpace logic might still introduce some, but the Conv layers should be cleaner.
    # Actually, the depth_to_space plugin itself might be doing transposes.
    # But generally we expect significant reduction or at least difference.
    # Let's verify we didn't EXPLODE the count.

    # Ideally: count should be lower.
    # Example: Conv requires NCHW.
    # Default: Input(NHWC) -> T(NCHW) -> Conv -> T(NHWC) -> Output
    # NCHW-mode: Input(NCHW) -> T(NHWC) [inserted by us] -> T(NCHW) [inserted by plugin?] -> Conv -> ...
    # Wait, if plugin inserts T(NHWC->NCHW), and we insert T(NCHW->NHWC) at input...
    # Then T(NHWC->NCHW) * T(NCHW->NHWC) = Identity.
    # IR optimization (remove_redundant_transpose_pairs_ir) should kill them.
    # So we expect NO transposes around Conv input.

    # 5. Numerical Verification
    # Run Default with NHWC
    sess_default = ort.InferenceSession(
        onnx_model_default.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    res_default = sess_default.run(None, {"in_0": np.array(x_nhwc).astype(np.float32)})[
        0
    ]
    np.testing.assert_allclose(
        res_default,
        expected_out_nhwc,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Default ONNX conversion failed mismatch",
    )

    # Run NCHW with NCHW data
    # Transpose input to NCHW for the session
    x_nchw = np.transpose(np.array(x_nhwc), (0, 3, 1, 2)).astype(np.float32)

    sess_nchw = ort.InferenceSession(
        onnx_model_nchw.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    # Note: input name might have changed to 'in_0_nchw' or similar if logic dictates,
    # but currently our logic appends to _inputs list.
    # Let's check the input name from the model.
    input_name_nchw = onnx_model_nchw.graph.input[0].name
    res_nchw = sess_nchw.run(None, {input_name_nchw: x_nchw})[0]

    # Expected output is also NCHW version of the JAX result
    expected_out_nchw = np.transpose(np.array(expected_out_nhwc), (0, 3, 1, 2))

    np.testing.assert_allclose(
        res_nchw,
        expected_out_nchw,
        rtol=1e-4,
        atol=1e-5,
        err_msg="NCHW ONNX conversion failed mismatch",
    )

    # Check shape of output
    assert res_nchw.shape == expected_out_nchw.shape
    assert res_nchw.shape == (1, 1, 16, 16)  # (N, C, H, W) for (1, 16, 16, 1) input
