# tests/extra_tests/test_return_modes.py

from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx.user_interface import allclose, allclose_onnxruntime_web, to_onnx


def _simple(x):
    return jnp.add(x, 1.0)


def _shape_dims(
    model: onnx.ModelProto, value_name: str
) -> tuple[int | str | None, ...]:
    for value_info in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        if value_info.name != value_name:
            continue
        tensor_shape = value_info.type.tensor_type.shape
        dims: list[int | str | None] = []
        for dim in tensor_shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        return tuple(dims)
    raise AssertionError(f"Missing value_info for {value_name!r}")


def test_return_mode_proto():
    model = to_onnx(_simple, inputs=[(2,)], model_name="rt_proto")
    assert isinstance(model, onnx.ModelProto)
    assert any(output.name for output in model.graph.output)


def test_return_mode_ir():
    ir_model = to_onnx(
        _simple,
        inputs=[(2,)],
        model_name="rt_ir",
        return_mode="ir",
    )
    assert isinstance(ir_model, ir.Model)
    assert getattr(ir_model, "graph", None) is not None


def test_return_mode_file(tmp_path: Path):
    output = tmp_path / "model.onnx"
    result = to_onnx(
        _simple,
        inputs=[(2,)],
        model_name="rt_file",
        return_mode="file",
        output_path=output,
    )
    assert isinstance(result, str)
    assert Path(result) == output
    assert output.is_file()
    loaded = onnx.load_model(output)
    assert any(node.op_type == "Add" for node in loaded.graph.node)


def test_web_export_mode_writes_single_file(tmp_path: Path):
    output = tmp_path / "model.onnx"
    stale_sidecar = tmp_path / "model.onnx.data"
    stale_sidecar.write_bytes(b"stale")

    result = to_onnx(
        _simple,
        inputs=[(2,)],
        model_name="rt_file_web",
        return_mode="file",
        output_path=output,
        export_mode="web",
    )

    assert Path(result) == output
    assert output.is_file()
    assert not stale_sidecar.exists()
    loaded = onnx.load_model(output)
    assert not any(init.external_data for init in loaded.graph.initializer)


def test_invalid_export_mode_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported export_mode"):
        to_onnx(
            _simple,
            inputs=[(2,)],
            return_mode="file",
            output_path=tmp_path / "bad.onnx",
            export_mode="browser",  # type: ignore[arg-type]
        )


def test_onnxruntime_web_wasm_validation_smoke(tmp_path: Path):
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for onnxruntime-web validation")
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "node_modules" / "onnxruntime-web" / "package.json").is_file():
        pytest.skip("Run `npm install` to enable onnxruntime-web validation")

    output = tmp_path / "model.onnx"
    to_onnx(
        _simple,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        model_name="rt_file_web_smoke",
        return_mode="file",
        output_path=output,
        export_mode="web",
    )

    passed, message = allclose_onnxruntime_web(
        str(output),
        [np.array([1.0, 2.0], dtype=np.float32)],
        rtol=1e-6,
        atol=1e-6,
    )
    assert passed, message


def test_copysign_broadcast_web_export_keeps_operand_shapes(tmp_path: Path):
    def copysign(x, y):
        return jnp.copysign(x, y)

    output = tmp_path / "copysign_broadcast.onnx"
    to_onnx(
        copysign,
        inputs=[
            jax.ShapeDtypeStruct((2, 1), jnp.float32),
            jax.ShapeDtypeStruct((1, 3), jnp.float32),
        ],
        model_name="copysign_broadcast_web",
        return_mode="file",
        output_path=output,
        export_mode="web",
    )

    loaded = onnx.load_model(output)
    nodes = {node.op_type: node for node in loaded.graph.node}
    assert _shape_dims(loaded, nodes["Abs"].output[0]) == (2, 1)
    assert _shape_dims(loaded, nodes["Neg"].output[0]) == (2, 1)
    assert _shape_dims(loaded, nodes["Less"].output[0]) == (1, 3)
    assert _shape_dims(loaded, nodes["Where"].output[0]) == (2, 3)

    xs = [
        np.array([[-0.25], [0.5]], dtype=np.float32),
        np.array([[1.0, -1.0, 2.0]], dtype=np.float32),
    ]
    passed_cpu, cpu_msg = allclose(copysign, str(output), xs, rtol=1e-6, atol=1e-6)
    assert passed_cpu, cpu_msg

    if shutil.which("node") is None:
        pytest.skip("Node.js is required for onnxruntime-web validation")
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "node_modules" / "onnxruntime-web" / "package.json").is_file():
        pytest.skip("Run `npm install` to enable onnxruntime-web validation")

    passed_web, web_msg = allclose_onnxruntime_web(
        str(output),
        xs,
        rtol=1e-6,
        atol=1e-6,
    )
    assert passed_web, web_msg


def test_einsum_labeled_broadcast_web_export_expands_inputs(tmp_path: Path):
    def attention_logits(q, k):
        return jnp.einsum("...BTNH,BSNH->...BNTS", q, k)

    output = tmp_path / "einsum_attention_logits_batched.onnx"
    to_onnx(
        attention_logits,
        inputs=[
            jax.ShapeDtypeStruct((3, 1, 4, 8, 32), jnp.float32),
            jax.ShapeDtypeStruct((3, 4, 8, 32), jnp.float32),
        ],
        model_name="einsum_attention_logits_batched_web",
        return_mode="file",
        output_path=output,
        export_mode="web",
    )

    loaded = onnx.load_model(output)
    einsum_node = next(node for node in loaded.graph.node if node.op_type == "Einsum")
    assert _shape_dims(loaded, einsum_node.input[0]) == (3, 3, 4, 8, 32)
    assert _shape_dims(loaded, einsum_node.input[1]) == (3, 3, 4, 8, 32)
    assert _shape_dims(loaded, einsum_node.output[0]) == (3, 3, 8, 4, 4)

    xs = [
        np.linspace(-0.25, 0.25, 3 * 1 * 4 * 8 * 32, dtype=np.float32).reshape(
            3, 1, 4, 8, 32
        ),
        np.linspace(0.1, 0.4, 3 * 4 * 8 * 32, dtype=np.float32).reshape(3, 4, 8, 32),
    ]
    passed_cpu, cpu_msg = allclose(
        attention_logits,
        str(output),
        xs,
        rtol=1e-5,
        atol=1e-5,
    )
    assert passed_cpu, cpu_msg

    if shutil.which("node") is None:
        pytest.skip("Node.js is required for onnxruntime-web validation")
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "node_modules" / "onnxruntime-web" / "package.json").is_file():
        pytest.skip("Run `npm install` to enable onnxruntime-web validation")

    passed_web, web_msg = allclose_onnxruntime_web(
        str(output),
        xs,
        rtol=1e-5,
        atol=1e-5,
    )
    assert passed_web, web_msg


def test_irfft_web_export_keeps_real_channel_rank(tmp_path: Path):
    def irfft(x):
        return jnp.fft.irfft(x)

    output = tmp_path / "irfft_complex64.onnx"
    to_onnx(
        irfft,
        inputs=[jax.ShapeDtypeStruct((3,), jnp.complex64)],
        model_name="irfft_complex64_web",
        return_mode="file",
        output_path=output,
        export_mode="web",
    )

    loaded = onnx.load_model(output)
    nodes = {node.output[0]: node for node in loaded.graph.node}
    real_output = next(
        name for name, node in nodes.items() if name.startswith("irfft_real")
    )
    assert _shape_dims(loaded, real_output) == (1, 4)
    assert _shape_dims(loaded, loaded.graph.output[0].name) == (4,)

    xs = [np.array([1.0 + 0.0j, 2.0 - 0.5j, 3.0 + 0.0j], dtype=np.complex64)]
    passed_cpu, cpu_msg = allclose(irfft, str(output), xs, rtol=1e-5, atol=1e-5)
    assert passed_cpu, cpu_msg

    if shutil.which("node") is None:
        pytest.skip("Node.js is required for onnxruntime-web validation")
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "node_modules" / "onnxruntime-web" / "package.json").is_file():
        pytest.skip("Run `npm install` to enable onnxruntime-web validation")

    passed_web, web_msg = allclose_onnxruntime_web(
        str(output),
        xs,
        rtol=1e-5,
        atol=1e-5,
    )
    assert passed_web, web_msg


def test_nanmedian_web_export_sorts_nans_last(tmp_path: Path):
    def nanmedian(x):
        return jnp.nanmedian(x)

    output = tmp_path / "nanmedian.onnx"
    to_onnx(
        nanmedian,
        inputs=[jax.ShapeDtypeStruct((4,), jnp.float32)],
        model_name="nanmedian_web",
        return_mode="file",
        output_path=output,
        export_mode="web",
    )

    loaded = onnx.load_model(output)
    op_types = [node.op_type for node in loaded.graph.node]
    assert "TopK" in op_types
    assert "GatherElements" in op_types

    xs = [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)]
    passed_cpu, cpu_msg = allclose(nanmedian, str(output), xs, rtol=1e-6, atol=1e-6)
    assert passed_cpu, cpu_msg

    if shutil.which("node") is None:
        pytest.skip("Node.js is required for onnxruntime-web validation")
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "node_modules" / "onnxruntime-web" / "package.json").is_file():
        pytest.skip("Run `npm install` to enable onnxruntime-web validation")

    passed_web, web_msg = allclose_onnxruntime_web(
        str(output),
        xs,
        rtol=1e-6,
        atol=1e-6,
    )
    assert passed_web, web_msg


def test_sort_web_export_preserves_nan_after_sorting_last(tmp_path: Path):
    def sort_values(x):
        return jnp.sort(x)

    output = tmp_path / "sort_nan.onnx"
    to_onnx(
        sort_values,
        inputs=[jax.ShapeDtypeStruct((4,), jnp.float32)],
        model_name="sort_nan_web",
        return_mode="file",
        output_path=output,
        export_mode="web",
    )

    loaded = onnx.load_model(output)
    op_types = [node.op_type for node in loaded.graph.node]
    assert "IsNaN" in op_types
    assert "GatherElements" in op_types

    xs = [np.array([3.0, np.nan, 1.0, -2.0], dtype=np.float32)]
    passed_cpu, cpu_msg = allclose(sort_values, str(output), xs, rtol=1e-6, atol=1e-6)
    assert passed_cpu, cpu_msg

    if shutil.which("node") is None:
        pytest.skip("Node.js is required for onnxruntime-web validation")
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "node_modules" / "onnxruntime-web" / "package.json").is_file():
        pytest.skip("Run `npm install` to enable onnxruntime-web validation")

    passed_web, web_msg = allclose_onnxruntime_web(
        str(output),
        xs,
        rtol=1e-6,
        atol=1e-6,
    )
    assert passed_web, web_msg


def test_symmetric_product_web_export_stamps_transpose_shape(tmp_path: Path):
    def symmetric_product(a, c):
        return jax.lax.linalg.symmetric_product(a, c)

    output = tmp_path / "symmetric_product.onnx"
    to_onnx(
        symmetric_product,
        inputs=[
            jax.ShapeDtypeStruct((3, 2), jnp.float32),
            jax.ShapeDtypeStruct((3, 3), jnp.float32),
        ],
        model_name="symmetric_product_web",
        return_mode="file",
        output_path=output,
        export_mode="web",
    )

    loaded = onnx.load_model(output)
    transpose_node = next(
        node for node in loaded.graph.node if node.op_type == "Transpose"
    )
    matmul_node = next(node for node in loaded.graph.node if node.op_type == "MatMul")
    assert _shape_dims(loaded, transpose_node.output[0]) == (2, 3)
    assert _shape_dims(loaded, matmul_node.output[0]) == (3, 3)

    xs = [
        np.asarray([[1.0, 2.0], [0.5, -1.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray(
            [[2.0, 0.1, 0.2], [0.1, 3.0, -0.3], [0.2, -0.3, 1.5]],
            dtype=np.float32,
        ),
    ]
    passed_cpu, cpu_msg = allclose(
        symmetric_product,
        str(output),
        xs,
        rtol=1e-6,
        atol=1e-6,
    )
    assert passed_cpu, cpu_msg

    if shutil.which("node") is None:
        pytest.skip("Node.js is required for onnxruntime-web validation")
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "node_modules" / "onnxruntime-web" / "package.json").is_file():
        pytest.skip("Run `npm install` to enable onnxruntime-web validation")

    passed_web, web_msg = allclose_onnxruntime_web(
        str(output),
        xs,
        rtol=1e-6,
        atol=1e-6,
    )
    assert passed_web, web_msg


def test_zeta_broadcast_web_export_keeps_guard_shapes(tmp_path: Path):
    def zeta(s, q):
        return jax.lax.zeta(s, q)

    output = tmp_path / "zeta_broadcast.onnx"
    to_onnx(
        zeta,
        inputs=[
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.float32),
        ],
        model_name="zeta_broadcast_web",
        return_mode="file",
        output_path=output,
        export_mode="web",
    )

    loaded = onnx.load_model(output)
    invalid_or_pole = next(
        output_name
        for node in loaded.graph.node
        for output_name in node.output
        if output_name.startswith("zeta_invalid_or_pole")
    )
    assert _shape_dims(loaded, invalid_or_pole) == ()
    assert _shape_dims(loaded, loaded.graph.output[0].name) == (4,)

    xs = [
        np.asarray(2.0, dtype=np.float32),
        np.asarray([0.5, 1.0, 2.0, 5.0], dtype=np.float32),
    ]
    passed_cpu, cpu_msg = allclose(zeta, str(output), xs, rtol=1e-5, atol=1e-5)
    assert passed_cpu, cpu_msg

    if shutil.which("node") is None:
        pytest.skip("Node.js is required for onnxruntime-web validation")
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "node_modules" / "onnxruntime-web" / "package.json").is_file():
        pytest.skip("Run `npm install` to enable onnxruntime-web validation")

    passed_web, web_msg = allclose_onnxruntime_web(
        str(output),
        xs,
        rtol=1e-5,
        atol=1e-5,
    )
    assert passed_web, web_msg


def test_flip_sequences_web_export_expands_broadcast_index_grid(tmp_path: Path):
    from flax import nnx

    def flip_sequences(x, lengths):
        return nnx.nn.recurrent.flip_sequences(
            x,
            lengths,
            num_batch_dims=1,
            time_major=False,
        )

    output = tmp_path / "flip_sequences.onnx"
    to_onnx(
        flip_sequences,
        inputs=[
            jax.ShapeDtypeStruct((2, 5, 3), jnp.float32),
            jax.ShapeDtypeStruct((2,), jnp.int32),
        ],
        model_name="flip_sequences_web",
        return_mode="file",
        output_path=output,
        export_mode="web",
    )

    loaded = onnx.load_model(output)
    arange_output = next(
        node.output[0] for node in loaded.graph.node if node.op_type == "Range"
    )
    arange_reshape = next(
        node
        for node in loaded.graph.node
        if node.op_type == "Reshape" and node.input[0] == arange_output
    )
    assert _shape_dims(loaded, arange_reshape.output[0]) == (1, 5)
    assert any(
        node.op_type == "Expand" and node.input[0] == arange_reshape.output[0]
        for node in loaded.graph.node
    )

    xs = [
        np.linspace(-0.5, 0.5, 2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3),
        np.asarray([1, 4], dtype=np.int32),
    ]
    passed_cpu, cpu_msg = allclose(
        flip_sequences,
        str(output),
        xs,
        rtol=1e-6,
        atol=1e-6,
    )
    assert passed_cpu, cpu_msg

    if shutil.which("node") is None:
        pytest.skip("Node.js is required for onnxruntime-web validation")
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "node_modules" / "onnxruntime-web" / "package.json").is_file():
        pytest.skip("Run `npm install` to enable onnxruntime-web validation")

    passed_web, web_msg = allclose_onnxruntime_web(
        str(output),
        xs,
        rtol=1e-6,
        atol=1e-6,
    )
    assert passed_web, web_msg
