# tests/extra_tests/converter/test_source_asin_repair.py

"""Finite SOURCE Asin domain and precision witnesses."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from tests.extra_tests.converter.test_source_elementary_reductions import (
    _accept,
    _export,
    _FINITE,
)


_ASIN_WORDS = {
    "float32": {
        "P_bits": [1042983541, 3173979066, 3155022699],
        "Q_bits": [1065353216, 3207914926],
        "pio2_hi_bits": 1070141402,
        "pio2_lo_bits": 866263400,
    },
    "float64": {
        "P_bits": [
            4595172819793696085,
            13822908529170411389,
            4596417465768494165,
            13809305468778614587,
            4560439845004096136,
            4540259411154564873,
        ],
        "Q_bits": [
            4607182418800017408,
            13835966419869248843,
            4611733184086379208,
            13827746767276147033,
            4590215604441354882,
        ],
        "pio2_hi_bits": 4609753056924675352,
        "pio2_lo_bits": 4364452196894661639,
    },
}
_REGION_WORDS = {
    "float32": [
        3212836864,
        3212836863,
        3211788288,
        3210739712,
        3208642560,
        3204448257,
        3204448256,
        3204448255,
        3196059648,
        3187671040,
        3112173568,
        3011510272,
        2155872257,
        2155872256,
        2155872255,
        2147483649,
        0,
        1,
        8388607,
        8388608,
        8388609,
        864026624,
        964689920,
        1040187392,
        1048576000,
        1056964607,
        1056964608,
        1056964609,
        1061158912,
        1063256064,
        1064304640,
        1065353215,
        1065353216,
    ],
    "float64": [
        13830554455654793216,
        13830554455654793215,
        13829991505701371904,
        13829428555747950592,
        13828302655841107968,
        13826050856027422721,
        13826050856027422720,
        13826050856027422719,
        13821547256400052224,
        13817043656772681728,
        13776511260126347264,
        13722468064597901312,
        9227875636482146305,
        9227875636482146304,
        9227875636482146303,
        9223372036854775809,
        0,
        1,
        4503599627370495,
        4503599627370496,
        4503599627370497,
        4499096027743125504,
        4553139223271571456,
        4593671619917905920,
        4598175219545276416,
        4602678819172646911,
        4602678819172646912,
        4602678819172646913,
        4604930618986332160,
        4606056518893174784,
        4606619468846596096,
        4607182418800017407,
        4607182418800017408,
    ],
}
_BLOCK_WORDS = {
    "float32": [3208642560, 3196059648, 2147483648, 0, 1048576000, 1061158912],
    "float64": [
        13828302655841107968,
        13821547256400052224,
        9223372036854775808,
        0,
        4598175219545276416,
        4604930618986332160,
    ],
}


def _assert_asin_decomposition(proto: Any, dtype: Any) -> None:
    """Bind same-width rational/half-angle arithmetic and exact selections."""
    dtype = np.dtype(dtype)
    word_dtype = np.uint32 if dtype == np.dtype(np.float32) else np.uint64
    expected_type = onnx.helper.np_dtype_to_tensor_dtype(dtype)
    graph = onnx.shape_inference.infer_shapes(proto, strict_mode=True).graph
    nodes = list(graph.node)
    producers = {name: node for node in nodes for name in node.output}
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*graph.input, *graph.value_info, *graph.output)
    }
    initializers = {
        value.name: onnx.numpy_helper.to_array(value) for value in graph.initializer
    }
    assert not any(node.op_type == "Asin" for node in nodes)
    # Output zero is the original Asin value; later outputs are source consumers.
    # Its dependency cone includes every owning arithmetic temporary, without
    # admitting unrelated reverse-index arithmetic by its dtype or node name.
    asin_values: set[str] = set()
    pending = [graph.output[0].name]
    while pending:
        name = pending.pop()
        if name in asin_values:
            continue
        asin_values.add(name)
        if name in producers:
            pending.extend(producers[name].input)
    for node in nodes:
        if any(name in asin_values for name in node.output) and node.op_type in {
            "Abs",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Sqrt",
            "Neg",
        }:
            assert all(types[name] == expected_type for name in node.output), (
                "Asin arithmetic width",
                node.name,
                list(node.output),
            )
    words = _ASIN_WORDS[dtype.name]

    def fixed(name: str, word: int) -> bool:
        value = initializers.get(name)
        return bool(
            value is not None
            and value.dtype == dtype
            and value.shape == ()
            and int(value.view(word_dtype)) == word
        )

    def single(op: str, inputs: list[str]) -> str:
        matches = [
            node for node in nodes if node.op_type == op and list(node.input) == inputs
        ]
        assert len(matches) == 1, (op, inputs)
        return matches[0].output[0]

    def horner(name: str, coefficients: list[int], t: str) -> None:
        if len(coefficients) == 1:
            assert fixed(name, coefficients[0])
            return
        node = producers[name]
        assert node.op_type == "Add" and fixed(node.input[0], coefficients[0])
        product = producers[node.input[1]]
        assert product.op_type == "Mul" and product.input[0] == t
        horner(product.input[1], coefficients[1:], t)

    def rational(t: str) -> str:
        matches = []
        for node in nodes:
            if node.op_type != "Div":
                continue
            product = producers.get(node.input[0])
            if (
                product is not None
                and product.op_type == "Mul"
                and product.input[0] == t
            ):
                horner(product.input[1], words["P_bits"], t)
                horner(node.input[1], words["Q_bits"], t)
                matches.append(node.output[0])
        assert len(matches) == 1
        return matches[0]

    def selected(condition: str, when_true: str, when_false: str) -> str:
        matches = []
        for node in nodes:
            if node.op_type != "GatherElements":
                continue
            choices, index = (producers[name] for name in node.input)
            if choices.op_type != "Concat" or index.op_type != "Unsqueeze":
                continue
            cast = producers[index.input[0]]
            branches = [producers[name] for name in choices.input]
            if (
                cast.op_type == "Cast"
                and cast.input[0] == condition
                and [branch.input[0] for branch in branches] == [when_false, when_true]
            ):
                assert all(branch.op_type == "Unsqueeze" for branch in branches)
                squeezes = [
                    tail
                    for tail in nodes
                    if tail.op_type == "Squeeze" and tail.input[0] == node.output[0]
                ]
                assert len(squeezes) == 1
                matches.append(squeezes[0].output[0])
        assert len(matches) == 1
        return matches[0]

    zero = next(name for name in initializers if fixed(name, 0))
    half_word = 0x3F000000 if dtype == np.dtype(np.float32) else 0x3FE0000000000000
    one_word = 0x3F800000 if dtype == np.dtype(np.float32) else 0x3FF0000000000000
    two_word = 0x40000000 if dtype == np.dtype(np.float32) else 0x4000000000000000
    half = next(name for name in initializers if fixed(name, half_word))
    one = next(name for name in initializers if fixed(name, one_word))
    two = next(name for name in initializers if fixed(name, two_word))
    x = graph.input[0].name
    a = single("Abs", [x])
    small_t = single("Mul", [a, a])
    small = single("Add", [a, single("Mul", [a, rational(small_t)])])
    large_t = single("Mul", [single("Sub", [one, a]), half])
    s = single("Sqrt", [large_t])
    upper = single(
        "Mul", [two, single("Add", [s, single("Mul", [s, rational(large_t)])])]
    )
    hi = next(name for name in initializers if fixed(name, words["pio2_hi_bits"]))
    lo = next(name for name in initializers if fixed(name, words["pio2_lo_bits"]))
    large = single("Sub", [hi, single("Sub", [upper, lo])])
    magnitude = selected(single("LessOrEqual", [a, half]), small, large)
    signed = selected(single("Less", [x, zero]), single("Neg", [magnitude]), magnitude)
    preserve = single("Or", [single("IsNaN", [x]), single("Equal", [x, zero])])
    result = selected(preserve, x, signed)
    assert single("Identity", [result]) == graph.output[0].name


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_asin_native_domains_signed_zeros_and_original_precision(dtype: Any) -> None:
    with jax.enable_x64(dtype is np.float64):
        outside = np.nextafter(np.asarray(1, dtype), np.asarray(np.inf, dtype))
        inside = np.nextafter(np.asarray(1, dtype), np.asarray(0, dtype))
        smallest = np.nextafter(np.asarray(0, dtype), np.asarray(1, dtype))
        values = (
            np.asarray(
                [
                    -np.inf,
                    -outside,
                    -1,
                    -inside,
                    -0.75,
                    -0.25,
                    -smallest,
                    -0.0,
                    0.0,
                    smallest,
                    0.25,
                    0.75,
                    inside,
                    1,
                    outside,
                    np.inf,
                    np.nan,
                ],
                dtype,
            ),
        )

        def model(x: Any) -> Any:
            value = jax.lax.asin(x)
            return value, np.asarray(1, dtype) / value, jax.lax.is_finite(value)

        proto, references = _export(model, (values,))
        expected_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        assert proto.graph.input[0].type.tensor_type.elem_type == expected_type
        assert proto.graph.output[0].type.tensor_type.elem_type == expected_type
        _assert_asin_decomposition(proto, dtype)
        _accept(proto, (values,), references, _FINITE[np.dtype(dtype).name])


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_asin_symbolic_regions_and_lane_consumers(dtype: Any) -> None:
    with jax.enable_x64(dtype is np.float64):
        name = np.dtype(dtype).name
        region = _REGION_WORDS[name]
        block = _BLOCK_WORDS[name]
        rows = [region]
        for offset in (0, 5, 11):
            row = [0] * 33
            row[offset : offset + 6] = block
            rows.append(row)
        rows.append(region[::-1])
        word_dtype = np.uint32 if dtype is np.float32 else np.uint64
        feeds = tuple(
            (np.asarray(rows[:batch], dtype=word_dtype).view(dtype),)
            for batch in (2, 5)
        )

        def model(x: Any) -> Any:
            y = jax.lax.asin(x)
            return (
                y,
                y.astype(jnp.float32),
                y + jax.lax.rev(y, dimensions=(1,)),
                y > np.asarray(0.125, dtype),
            )

        (batch,) = jax.export.symbolic_shape("B")
        proto, references = _export(
            model, feeds, specs=[jax.ShapeDtypeStruct((batch, 33), dtype)]
        )
        _assert_asin_decomposition(proto, dtype)
        for value in (proto.graph.input[0], *proto.graph.output):
            assert value.type.tensor_type.shape.dim[0].dim_param == "B"
            assert value.type.tensor_type.shape.dim[1].dim_value == 33
        assert (
            proto.graph.output[1].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        )
        assert proto.graph.output[3].type.tensor_type.elem_type == onnx.TensorProto.BOOL
        _accept(proto, feeds, references, _FINITE[name])
