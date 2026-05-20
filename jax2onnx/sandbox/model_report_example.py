# jax2onnx/sandbox/model_report_example.py

from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.diagnostics import (
    RuntimeProfile,
    analyze_jax_export,
    analyze_jax_export_profiles,
    format_model_report_markdown,
    format_profile_analysis_markdown,
    write_model_report,
    write_profile_analysis_report,
)


def demo_model(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    return jnp.tanh(lhs + rhs)


def main() -> None:
    lhs = np.ones((2, 3), dtype=np.float32)
    rhs = np.full((2, 3), 0.25, dtype=np.float32)

    analysis = analyze_jax_export(
        demo_model,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        sample_inputs=(lhs, rhs),
        model_name="sandbox_model_report_demo",
        input_names=["lhs", "rhs"],
        output_names=["activation"],
        targets=("ort-cpu", "ort-web", "ort-mobile"),
    )
    report = analysis.report
    print(format_model_report_markdown(report))
    output_dir = Path(gettempdir()) / "jax2onnx_model_report_example"
    markdown_path = write_model_report(report, output_dir / "model_report.md")
    json_path = write_model_report(report, output_dir / "model_report.json")
    print(f"Markdown report written to: {markdown_path}")
    print(f"JSON report written to: {json_path}")

    profile_analysis = analyze_jax_export_profiles(
        demo_model,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        profiles=(
            RuntimeProfile(
                name="batch_2",
                sample_inputs=(lhs, rhs),
            ),
            RuntimeProfile(
                name="batch_5",
                sample_inputs=(
                    np.ones((5, 3), dtype=np.float32),
                    np.full((5, 3), 0.25, dtype=np.float32),
                ),
            ),
        ),
        model_name="sandbox_model_report_profiles_demo",
        input_names=["lhs", "rhs"],
        output_names=["activation"],
        targets=("ort-cpu",),
    )
    print("\n" + format_profile_analysis_markdown(profile_analysis))
    profile_markdown_path = write_profile_analysis_report(
        profile_analysis,
        output_dir / "profile_report.md",
    )
    profile_json_path = write_profile_analysis_report(
        profile_analysis,
        output_dir / "profile_report.json",
    )
    print(f"Profile Markdown report written to: {profile_markdown_path}")
    print(f"Profile JSON report written to: {profile_json_path}")


if __name__ == "__main__":
    main()
