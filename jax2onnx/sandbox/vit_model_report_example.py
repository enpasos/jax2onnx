# jax2onnx/sandbox/vit_model_report_example.py

"""Export a small NNX ViT example and generate internal model reports.

Run from the repository root:

    PYTHONDONTWRITEBYTECODE=1 poetry run python -m jax2onnx.sandbox.vit_model_report_example
"""

from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.diagnostics import (
    RuntimeProfile,
    analyze_jax_export_profiles,
    format_profile_analysis_markdown,
    write_model_report,
    write_profile_analysis_report,
)
from jax2onnx.plugins.examples.nnx.vit import VisionTransformer
from jax2onnx.plugins.plugin_system import construct_and_call, with_rng_seed


def _make_vit() -> VisionTransformer:
    """Build a small deterministic ViT from the existing NNX ViT example."""

    return (
        construct_and_call(
            VisionTransformer,
            height=28,
            width=28,
            num_hiddens=32,
            num_layers=1,
            num_heads=4,
            mlp_dim=64,
            num_classes=10,
            embedding_type="patch",
            patch_size=4,
            embedding_dropout_rate=0.0,
            attention_dropout_rate=0.0,
            mlp_dropout_rate=0.0,
            rngs=with_rng_seed(0),
        )
        .with_dtype(jnp.float32)
        .instantiate()
    )


def _sample_image_batch(batch_size: int) -> np.ndarray:
    values = np.linspace(
        0.0,
        1.0,
        num=batch_size * 28 * 28,
        dtype=np.float32,
    )
    return values.reshape(batch_size, 28, 28, 1)


def main() -> None:
    vit = _make_vit()

    analysis = analyze_jax_export_profiles(
        vit,
        [jax.ShapeDtypeStruct(("B", 28, 28, 1), jnp.float32)],
        profiles=(
            RuntimeProfile(
                name="batch_1",
                sample_inputs=(_sample_image_batch(1),),
            ),
            RuntimeProfile(
                name="batch_3",
                sample_inputs=(_sample_image_batch(3),),
            ),
        ),
        model_name="sandbox_nnx_vit_report_demo",
        input_names=["image"],
        output_names=["logits"],
        targets=("ort-cpu", "ort-web", "ort-mobile"),
    )

    print(format_profile_analysis_markdown(analysis))

    output_dir = Path(gettempdir()) / "jax2onnx_vit_model_report_example"
    first_profile_report = analysis.profile_reports[0].report
    model_markdown_path = write_model_report(
        first_profile_report,
        output_dir / "model_report_batch_1.md",
    )
    model_json_path = write_model_report(
        first_profile_report,
        output_dir / "model_report_batch_1.json",
    )
    profile_markdown_path = write_profile_analysis_report(
        analysis,
        output_dir / "profile_report.md",
    )
    profile_json_path = write_profile_analysis_report(
        analysis,
        output_dir / "profile_report.json",
    )

    print(f"Model Markdown report written to: {model_markdown_path}")
    print(f"Model JSON report written to: {model_json_path}")
    print(f"Profile Markdown report written to: {profile_markdown_path}")
    print(f"Profile JSON report written to: {profile_json_path}")


if __name__ == "__main__":
    main()
