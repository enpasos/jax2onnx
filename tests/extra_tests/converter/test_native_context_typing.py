# tests/extra_tests/converter/test_native_context_typing.py

"""Check the plugin typing contract through mypy, including invalid callers."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap


def test_native_context_builder_is_closed_without_an_extra_annotation(tmp_path) -> None:
    root = Path(__file__).resolve().parents[3]
    source = textwrap.dedent(
        """\
        from typing import assert_type

        import onnx_ir as ir

        from jax2onnx._compat.jax import JaxprEqn
        from jax2onnx.converter.typing_support import (
            CanonicalLowering,
            ConverterContext,
            LoweringContextProtocol,
            NativeLoweringContext,
            PrimitiveLowering,
        )


        def native_lower(ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
            value = ctx.require_value_for_var(eqn.invars[0])
            result = ctx.builder.op("Neg", [value])
            assert_type(result, ir.Value)
            ctx.bind_value_for_var(eqn.outvars[0], result)


        class NativePlugin:
            def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
                native_lower(ctx, eqn)


        native_callback: CanonicalLowering = native_lower
        native_plugin: PrimitiveLowering = NativePlugin()


        def legacy_lower(ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
            value = ctx.require_value_for_var(eqn.invars[0])
            ctx.bind_value_for_var(eqn.outvars[0], ctx.builder.Neg(value))


        legacy_callback: CanonicalLowering = legacy_lower


        def converter_operations(ctx: ConverterContext, value: ir.Value) -> None:
            native_context: NativeLoweringContext = ctx
            native_context.builder.op("Neg", [value])
            ctx.builder.Neg(value)
            ctx._lowering_record_owner = ctx


        def invalid_native_calls(ctx: NativeLoweringContext, value: ir.Value) -> None:
            ctx.builder.op_typo("Neg", [value])  # expect-error: attr-defined
            ctx.builder.Neg(value)  # expect-error: attr-defined
            ctx.builder.op("Neg", [1])  # expect-error: list-item
            ctx._lowering_record_owner  # expect-error: attr-defined
        """
    )
    sample = tmp_path / "native_context_contract.py"
    sample.write_text(source, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(root / "pyproject.toml"),
            "--cache-dir",
            "/dev/null",
            "--no-incremental",
            "--no-error-summary",
            "--no-pretty",
            "--no-color-output",
            # The project skips imported first-party modules. Include the
            # protocol explicitly so its builder cannot silently become Any.
            str(root / "jax2onnx/converter/typing_support.py"),
            str(sample),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    expected = [
        (line_number, line.split("# expect-error: ", 1)[1])
        for line_number, line in enumerate(source.splitlines(), start=1)
        if "# expect-error: " in line
    ]
    diagnostics = result.stdout.splitlines()
    assert result.returncode == 1, result.stdout + result.stderr
    assert not result.stderr
    assert len(diagnostics) == len(expected), result.stdout
    for diagnostic, (line_number, code) in zip(diagnostics, expected):
        assert diagnostic.startswith(f"{sample}:{line_number}: error:"), result.stdout
        assert diagnostic.endswith(f"[{code}]"), result.stdout
