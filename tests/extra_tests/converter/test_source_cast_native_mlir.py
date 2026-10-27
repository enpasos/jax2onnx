# tests/extra_tests/converter/test_source_cast_native_mlir.py

"""Whole native MLIR contracts, independent of source samples and emission."""

from collections.abc import Iterator
from dataclasses import FrozenInstanceError, fields
import hashlib
from typing import Any

import pytest

from jax2onnx._compat._cast_lowering import (
    NativeCastModule,
    validate_original_cast_module,
)


@pytest.fixture
def native_ir() -> Iterator[Any]:
    from jax._src.interpreters import mlir
    from jaxlib.mlir import ir

    with mlir.make_ir_context() as context:
        # Registered func/stablehlo operations retain their actual verifiers;
        # counterfeit-op tests additionally need to parse a foreign dialect.
        context.allow_unregistered_dialects = True
        yield ir


def _tensor(shape: tuple[Any, ...], element: str) -> str:
    dimensions = "".join(str(dimension) + "x" for dimension in shape)
    return "tensor<" + dimensions + element + ">"


def _function(
    shape: tuple[Any, ...] = (6,),
    *,
    source_dtype: str = "f64",
    result_dtype: str = "f16",
    metadata: str = "absent",
    body: str | None = None,
    arguments: str | None = None,
    input_types: str | None = None,
    output_types: str | None = None,
) -> str:
    source, result = _tensor(shape, source_dtype), _tensor(shape, result_dtype)
    attrs = [
        "function_type = ("
        + (source if input_types is None else input_types)
        + ") -> "
        + (result if output_types is None else output_types),
        'sym_name = "main"',
        'sym_visibility = "public"',
    ]
    if metadata in ("arg-only", "empty", "named"):
        attrs.append("arg_attrs = [{}]")
    if metadata in ("result-only", "empty", "named"):
        attrs.append(
            'res_attrs = [{jax.result_info = "independent_result"}]'
            if metadata == "named"
            else "res_attrs = [{}]"
        )
    if body is None:
        body = (
            '%cast = "stablehlo.convert"(%arg0) : ('
            + source
            + ") -> "
            + result
            + "\n"
            + '"func.return"(%cast) : ('
            + result
            + ") -> ()\n"
        )
    return (
        '"func.func"() ({\n^bb0('
        + ("%arg0: " + source if arguments is None else arguments)
        + "):\n"
        + body
        + "}) {"
        + ", ".join(attrs)
        + "} : () -> ()\n"
    )


def _program(
    shape: tuple[Any, ...] = (6,),
    *,
    metadata: str = "absent",
    symbol: str = "independent_source",
    function: str | None = None,
) -> str:
    return (
        "module @"
        + symbol
        + " attributes {\n"
        + "mhlo.num_replicas = 1 : i32,\n"
        + "mhlo.num_partitions = 1 : i32\n} {\n"
        + (_function(shape, metadata=metadata) if function is None else function)
        + "}\n"
    )


def _parsed(ir: Any, text: str) -> Any:
    module = ir.Module.parse(text)
    assert (
        module.operation.verify()
    ), "the negative must reach the checker as valid native IR"
    return module


def _assert_record(module: Any, shape: tuple[int, ...]) -> NativeCastModule:
    result = validate_original_cast_module(module, shape)
    assert type(result) is NativeCastModule
    assert result.shape == shape and type(result.shape) is tuple
    assert all(type(dimension) is int for dimension in result.shape)
    assert result.operations == ("stablehlo.convert", "func.return")
    assert type(result.text) is str and result.text == str(module)
    assert type(result.text_sha256) is str
    assert result.text_sha256 == hashlib.sha256(result.text.encode()).hexdigest()
    assert len(result.text.encode()) <= 65536
    assert [field.name for field in fields(result)] == [
        "text",
        "text_sha256",
        "shape",
        "operations",
    ]
    return result


@pytest.mark.parametrize(
    "shape",
    [(), (0,), (6,), (17,), (3, 7)],
    ids=["scalar", "empty", "vector", "odd-tail", "rank-two"],
)
@pytest.mark.parametrize(
    "metadata", ["absent", "arg-only", "result-only", "empty", "named"]
)
@pytest.mark.parametrize("symbol", ["independent_source", "not_a_fixture_whitelist"])
def test_actual_native_module_has_the_complete_original_typed_ssa_contract(
    native_ir: Any,
    shape: tuple[int, ...],
    metadata: str,
    symbol: str,
) -> None:
    module = _parsed(native_ir, _program(shape, metadata=metadata, symbol=symbol))
    _assert_record(module, shape)


@pytest.mark.parametrize(
    "symbol,raw_sha256",
    [
        (
            "jit_convert_element_type",
            "84f62f7aa15115ba2e14a7059842ab159950a45dd65a19ece5be890756b551e7",
        ),
        ("jit_fn", "897e1923c01c6df1d214d3be9d35770747f6528d0490b736afa92e6f43e9f326"),
    ],
    ids=["eager", "fresh-jit"],
)
def test_untouched_archived_native_program_text_is_a_supported_structural_witness(
    native_ir: Any,
    symbol: str,
    raw_sha256: str,
) -> None:
    # Literal original program text, independently read from CI545/311 report
    # 6c09d07d and CI546/314 report7a26020e. Both stacks stored these exact bytes.
    # Parsing this IR is structural validation, not source/compiled execution.
    text = (
        "module @"
        + symbol
        + " attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {\n"
        + '  func.func public @main(%arg0: tensor<6xf64>) -> (tensor<6xf16> {jax.result_info = "result"}) {\n'
        + "    %0 = stablehlo.convert %arg0 : (tensor<6xf64>) -> tensor<6xf16>\n"
        + "    return %0 : tensor<6xf16>\n  }\n}\n"
    )
    assert hashlib.sha256(text.encode()).hexdigest() == raw_sha256
    _assert_record(_parsed(native_ir, text), (6,))


@pytest.mark.parametrize(
    "source_dtype,result_dtype",
    [("f32", "f16"), ("i64", "f16"), ("f64", "f32"), ("f64", "i16")],
    ids=["narrow-source", "integer-source", "wide-result", "integer-result"],
)
def test_well_formed_other_conversions_cannot_borrow_the_original_typed_contract(
    native_ir: Any,
    source_dtype: str,
    result_dtype: str,
) -> None:
    _assert_record(_parsed(native_ir, _program()), (6,))
    function = _function(source_dtype=source_dtype, result_dtype=result_dtype)
    module = _parsed(native_ir, _program(function=function))
    with pytest.raises(ValueError):
        validate_original_cast_module(module, (6,))


@pytest.mark.parametrize(
    "actual_shape",
    [(), (0,), (7,), (2, 3), ("?",)],
    ids=["scalar", "empty", "changed-extent", "changed-rank", "dynamic"],
)
def test_well_formed_different_shapes_do_not_match_an_expected_six_element_vector(
    native_ir: Any,
    actual_shape: tuple[Any, ...],
) -> None:
    _assert_record(_parsed(native_ir, _program()), (6,))
    module = _parsed(native_ir, _program(actual_shape))
    with pytest.raises(ValueError):
        validate_original_cast_module(module, (6,))


def _well_formed_fault(fault: str) -> str:
    text = _program(metadata="named")
    if fault == "missing-replicas":
        return text.replace("mhlo.num_replicas = 1 : i32,\n", "")
    if fault == "missing-partitions":
        return text.replace(",\nmhlo.num_partitions = 1 : i32", "")
    if fault == "wide-replica-metadata":
        return text.replace(
            "mhlo.num_replicas = 1 : i32", "mhlo.num_replicas = 1 : i64"
        )
    if fault == "replicated":
        return text.replace(
            "mhlo.num_replicas = 1 : i32", "mhlo.num_replicas = 2 : i32"
        )
    if fault == "partitioned":
        return text.replace(
            "mhlo.num_partitions = 1 : i32", "mhlo.num_partitions = 2 : i32"
        )
    if fault == "extra-module-attribute":
        return text.replace("attributes {", "attributes {test.unreviewed = true,")
    if fault == "renamed-entry":
        return text.replace('sym_name = "main"', 'sym_name = "other"')
    if fault == "private-entry":
        return text.replace('sym_visibility = "public"', 'sym_visibility = "private"')
    if fault == "implicit-public":
        return text.replace(', sym_visibility = "public"', "")
    if fault == "target-feature-override":
        return text.replace(
            'sym_visibility = "public"',
            'sym_visibility = "public", llvm.target_features = "+avx512fp16"',
        )
    if fault == "argument-metadata":
        return text.replace(
            "arg_attrs = [{}]", "arg_attrs = [{test.unreviewed = true}]"
        )
    if fault == "result-metadata":
        return text.replace(
            'jax.result_info = "independent_result"', "test.unreviewed = true"
        )
    if fault == "convert-attribute":
        return text.replace(
            '"stablehlo.convert"(%arg0)',
            '"stablehlo.convert"(%arg0) {test.rounding = "rtz"}',
        )
    if fault == "return-attribute":
        return text.replace(
            '"func.return"(%cast)', '"func.return"(%cast) {test.unreviewed = true}'
        )
    if fault == "second-function":
        function = _function()
        return _program(
            function=function
            + function.replace('sym_name = "main"', 'sym_name = "helper"')
        )
    if fault == "top-level-extra-op":
        return _program(function=_function() + '"test.unreviewed"() : () -> ()\n')
    if fault == "nested-module-owner":
        return _program(function=_program())
    source, result = "tensor<6xf64>", "tensor<6xf16>"
    tail = '"func.return"(%cast) : (' + result + ") -> ()\n"
    convert = (
        '%cast = "stablehlo.convert"(%arg0) : (' + source + ") -> " + result + "\n"
    )
    if fault == "extra-input":
        function = _function(
            arguments="%arg0: " + source + ", %arg1: " + source,
            input_types=source + ", " + source,
        )
    elif fault == "extra-output":
        function = _function(
            body=convert
            + '"func.return"(%cast, %cast) : ('
            + result
            + ", "
            + result
            + ") -> ()\n",
            output_types="(" + result + ", " + result + ")",
        )
    elif fault == "no-output":
        function = _function(
            body=convert + '"func.return"() : () -> ()\n', output_types="()"
        )
    elif fault == "constant-result":
        function = _function(
            body='%cast = "stablehlo.constant"() {value = dense<0.0> : '
            + result
            + "} : () -> "
            + result
            + "\n"
            + tail
        )
    elif fault == "constant-source":
        constant = (
            '%constant = "stablehlo.constant"() {value = dense<0.0> : '
            + source
            + "} : () -> "
            + source
            + "\n"
        )
        function = _function(
            body=constant + convert.replace("(%arg0)", "(%constant)") + tail
        )
    elif fault == "intermediate-arithmetic":
        extra = (
            '%sum = "stablehlo.add"(%arg0, %arg0) : ('
            + source
            + ", "
            + source
            + ") -> "
            + source
            + "\n"
        )
        function = _function(body=extra + convert.replace("(%arg0)", "(%sum)") + tail)
    elif fault == "unused-extra-convert":
        function = _function(body=convert.replace("%cast", "%unused") + convert + tail)
    elif fault == "counterfeit-convert":
        function = _function(
            body=convert.replace("stablehlo.convert", "test.convert") + tail
        )
    elif fault == "recursive-call":
        function = _function(
            body=convert.replace(
                '"stablehlo.convert"(%arg0)', '"func.call"(%arg0) {callee = @main}'
            )
            + tail
        )
    elif fault in ("pure-custom-call", "effectful-custom-call"):
        effect = "true" if fault == "effectful-custom-call" else "false"
        call = (
            '%cast = "stablehlo.custom_call"(%arg0) {call_target_name = "independent_unknown", backend_config = "", api_version = 2 : i32, has_side_effect = '
            + effect
            + "} : ("
            + source
            + ") -> "
            + result
            + "\n"
        )
        function = _function(body=call + tail)
    elif fault == "nested-branch-regions":
        predicate = '%pred = "stablehlo.constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>\n'
        branch = (
            convert.replace("%cast", "%branch")
            + '"stablehlo.return"(%branch) : ('
            + result
            + ") -> ()\n"
        )
        conditional = (
            '%cast = "stablehlo.if"(%pred) ({\n'
            + branch
            + "}, {\n"
            + branch
            + "}) : (tensor<i1>) -> "
            + result
            + "\n"
        )
        function = _function(body=predicate + conditional + tail)
    else:
        raise AssertionError("unknown fixture fault")
    return _program(function=function)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-replicas",
        "missing-partitions",
        "wide-replica-metadata",
        "replicated",
        "partitioned",
        "extra-module-attribute",
        "renamed-entry",
        "private-entry",
        "implicit-public",
        "target-feature-override",
        "argument-metadata",
        "result-metadata",
        "convert-attribute",
        "return-attribute",
        "second-function",
        "top-level-extra-op",
        "extra-input",
        "extra-output",
        "no-output",
        "constant-result",
        "constant-source",
        "intermediate-arithmetic",
        "unused-extra-convert",
        "counterfeit-convert",
        "recursive-call",
        "pure-custom-call",
        "effectful-custom-call",
        "nested-module-owner",
        "nested-branch-regions",
    ],
)
def test_well_formed_semantic_or_coverage_changes_are_rejected(
    native_ir: Any,
    fault: str,
) -> None:
    _assert_record(_parsed(native_ir, _program(metadata="named")), (6,))
    changed = _well_formed_fault(fault)
    assert changed != _program(metadata="named")
    module = _parsed(native_ir, changed)
    with pytest.raises(ValueError):
        validate_original_cast_module(module, (6,))


@pytest.mark.parametrize(
    "fault",
    [
        "module-name-type",
        "replica-type",
        "partition-type",
        "missing-function-type",
        "function-type-kind",
        "argument-attribute-arity",
        "result-attribute-type",
        "return-wrong-ssa",
        "foreign-input-owner",
        "foreign-result-owner",
    ],
)
def test_malformed_native_metadata_and_ssa_fail_the_checker_boundary(
    native_ir: Any,
    fault: str,
) -> None:
    module = _parsed(native_ir, _program(metadata="named"))
    _assert_record(module, (6,))
    function = next(iter(module.body.operations)).operation
    if fault == "module-name-type":
        module.operation.attributes["sym_name"] = native_ir.IntegerAttr.get(
            native_ir.IntegerType.get_signless(32), 1
        )
    elif fault in ("replica-type", "partition-type"):
        key = "mhlo.num_replicas" if fault == "replica-type" else "mhlo.num_partitions"
        module.operation.attributes[key] = native_ir.StringAttr.get("1")
    elif fault == "missing-function-type":
        del function.attributes["function_type"]
    elif fault == "function-type-kind":
        function.attributes["function_type"] = native_ir.StringAttr.get(
            "(tensor<6xf64>) -> tensor<6xf16>"
        )
    elif fault == "argument-attribute-arity":
        function.attributes["arg_attrs"] = native_ir.ArrayAttr.get([])
    elif fault == "result-attribute-type":
        function.attributes["res_attrs"] = native_ir.ArrayAttr.get(
            [native_ir.StringAttr.get("result")]
        )
    elif fault == "return-wrong-ssa":
        block = function.regions[0].blocks[0]
        operations = list(block.operations)
        operations[1].operation.operands[0] = block.arguments[0]
    else:
        foreign_module = _parsed(native_ir, _program(symbol="foreign_owner"))
        foreign_function = next(iter(foreign_module.body.operations)).operation
        foreign_block = foreign_function.regions[0].blocks[0]
        block = function.regions[0].blocks[0]
        operations = list(block.operations)
        if fault == "foreign-input-owner":
            target = operations[0].operation
            foreign_value = foreign_block.arguments[0]
        else:
            foreign_convert = next(iter(foreign_block.operations)).operation
            target = operations[1].operation
            foreign_value = foreign_convert.results[0]
        original_operand = target.operands[0]
        original_text, foreign_text = str(module), str(foreign_module)
        # A native OpOperand records a use-list link, not Python ownership of
        # the donor module. Keep both trees alive until that link is restored.
        owners = (
            module,
            foreign_module,
            function,
            foreign_function,
            block,
            foreign_block,
        )
        try:
            target.operands[0] = foreign_value
            assert target.operands[0] == foreign_value
            assert target.operands[0] != original_operand
            with pytest.raises(ValueError):
                validate_original_cast_module(module, (6,))
        finally:
            target.operands[0] = original_operand
            assert target.operands[0] == original_operand
            assert owners[0] is module and owners[1] is foreign_module
            assert module.operation.verify() and foreign_module.operation.verify()
            assert str(module) == original_text and str(foreign_module) == foreign_text
        return
    with pytest.raises(ValueError):
        validate_original_cast_module(module, (6,))


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "none",
        "bool",
        "float",
        "negative",
        "symbolic",
        "nested",
        "tuple-subclass",
        "foreign",
    ],
)
def test_expected_shape_is_exact_passive_static_metadata(
    native_ir: Any, fault: str
) -> None:
    module = _parsed(native_ir, _program())
    _assert_record(module, (6,))
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("foreign-shape-protocol")
        raise AssertionError("foreign expected shape was inspected")

    class ForeignTuple(tuple):
        __iter__ = forbidden
        __len__ = forbidden
        __eq__ = forbidden

    class Foreign:
        __iter__ = forbidden
        __len__ = forbidden
        __ge__ = forbidden
        __eq__ = forbidden

    values: dict[str, Any] = {
        "list": [6],
        "none": None,
        "bool": (True,),
        "float": (6.0,),
        "negative": (-1,),
        "symbolic": ("B",),
        "nested": ((6,),),
        "tuple-subclass": ForeignTuple((6,)),
        "foreign": (Foreign(),),
    }
    with pytest.raises(ValueError):
        validate_original_cast_module(module, values[fault])
    assert calls == []


@pytest.mark.parametrize("fault", ["none", "text", "operation", "foreign"])
def test_module_authority_is_exact_native_ir_not_a_protocol(
    native_ir: Any, fault: str
) -> None:
    module = _parsed(native_ir, _program())
    calls: list[str] = []

    class Foreign:
        @property
        def operation(self) -> Any:
            calls.append("operation")
            raise AssertionError("foreign module was inspected")

        def __str__(self) -> str:
            calls.append("str")
            raise AssertionError("foreign module was formatted")

    values = {
        "none": None,
        "text": str(module),
        "operation": module.operation,
        "foreign": Foreign(),
    }
    with pytest.raises(ValueError):
        validate_original_cast_module(values[fault], (6,))
    assert calls == []


@pytest.mark.parametrize("field", ["module-symbol", "result-name"])
def test_oversized_nonsemantic_names_are_bounded(native_ir: Any, field: str) -> None:
    module = _parsed(native_ir, _program(metadata="named"))
    _assert_record(module, (6,))
    if field == "module-symbol":
        module.operation.attributes["sym_name"] = native_ir.StringAttr.get("m" * 65537)
    else:
        function = next(iter(module.body.operations)).operation
        name = native_ir.DictAttr.get(
            {"jax.result_info": native_ir.StringAttr.get("r" * 65537)}
        )
        function.attributes["res_attrs"] = native_ir.ArrayAttr.get([name])
    assert module.operation.verify()
    with pytest.raises(ValueError):
        validate_original_cast_module(module, (6,))


def test_result_retains_only_frozen_primitive_copy_after_the_native_module_changes(
    native_ir: Any,
) -> None:
    module = _parsed(native_ir, _program())
    result = _assert_record(module, (6,))
    before = (result.text, result.text_sha256, result.shape, result.operations)
    module.operation.attributes["test.unreviewed"] = native_ir.UnitAttr.get()
    assert module.operation.verify()
    with pytest.raises(ValueError):
        validate_original_cast_module(module, (6,))
    assert (result.text, result.text_sha256, result.shape, result.operations) == before
    with pytest.raises(FrozenInstanceError):
        result.text = "replacement"  # type: ignore[misc]


def test_checker_does_not_trace_lower_construct_options_or_acquire_a_backend(
    native_ir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax
    from jax._src import compiler, xla_bridge
    from jax._src.interpreters import mlir

    module = _parsed(native_ir, _program())
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("forbidden-source-or-compilation-entry")
        raise AssertionError("checker invoked a source or compilation entry")

    for name in ("make_jaxpr", "eval_shape", "jit"):
        monkeypatch.setattr(jax, name, forbidden)
    monkeypatch.setattr(compiler, "get_compile_options", forbidden)
    monkeypatch.setattr(mlir, "lower_jaxpr_to_module", forbidden)
    monkeypatch.setattr(xla_bridge, "get_backend", forbidden)
    _assert_record(module, (6,))
    assert calls == []
