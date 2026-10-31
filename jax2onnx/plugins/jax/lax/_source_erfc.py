# jax2onnx/plugins/jax/lax/_source_erfc.py

# Copyright 2020 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Modified by the jax2onnx contributors: typed standard-ONNX IR port, 2026.
"""StableHLO's source-typed Erfc decomposition, with unchanged expression order.

Pinned revisions (the Erfc blocks are identical):
https://github.com/openxla/stablehlo/blob/639335932274930617d9724fb25ec9f2f52c5e40/stablehlo/transforms/ChloLegalizeToStablehlo.cpp
https://github.com/openxla/stablehlo/blob/e6f81ebd06b3509f2c7fa6175430aadbd4d724ca/stablehlo/transforms/ChloLegalizeToStablehlo.cpp
Reviewed coefficient JSON SHA256:
6ca6b169fd53f1f8097dabd57f1a6af0b59e0eab66534ff005db58ca16e55c29
Upstream Erfc block SHA256:
02c5fd9de029079d91b99d53134da54e7813f57d98e16138ada5db260ec73d4a
Full upstream license/exception: jax2onnx/licenses/StableHLO-Erfc-LICENSE.txt.

Raw little-endian bytes avoid reparsing or rerounding the 66 coefficient literals.
This expression does not configure target arithmetic or flush subnormal values.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import NativeLoweringContext

_COEFFICIENTS: Final[dict[ir.DataType, dict[str, tuple[str, ...]]]] = {
    ir.DataType.FLOAT: {
        "ErfT": (
            "19b5a438",  # +7.853861353153693E-5
            "80fb51ba",  # -8.010193625184903E-4
            "d902aa3b",  # +5.188327685732524E-3
            "87fcdbbc",  # -2.685381193529856E-2
            "7c16e73d",  # +1.128358514861418E-1
            "9f93c0be",  # -3.761262582423300E-1
            "ba6e903f",  # +1.128379165726710E+0
        ),
        "ErfcP": (
            "f49cbe3c",  # +2.326819970068386E-2
            "68080ebe",  # -1.387039388740657E-1
            "d0cbbc3e",  # +3.687424674597105E-1
            "f81c15bf",  # -5.824733027278666E-1
            "e3f91e3f",  # +6.210004621745983E-1
            "c028fdbe",  # -4.944515323274145E-1
            "7154ae3e",  # +3.404879937665872E-1
            "80588cbe",  # -2.741127028184656E-1
            "e656103f",  # +5.638259427386472E-1
        ),
        "ErfcR": (
            "83a427c1",  # -1.047766399936249E+1
            "9ca24f41",  # +1.297719955372516E+1
            "4adbefc0",  # -7.495518717768503E+0
            "faf13a40",  # +2.921019019210786E+0
            "36f481bf",  # -1.015265279202700E+0
            "3efcd73e",  # +4.218463358204948E-1
            "5d6c90be",  # -2.820767439740514E-1
            "b96e103f",  # +5.641895067754075E-1
        ),
    },
    ir.DataType.DOUBLE: {
        "ErfT": (
            "885d371ebf352340",  # 9.60497373987051638749E0
            "7f069e4eaa815640",  # 9.00260197203842689217E1
            "b735b4bc0270a140",  # 2.23200534594684319226E3
            "90ef723c535bbb40",  # 7.00332514112805075473E3
            "dc1342a40925eb40",  # 5.55923013010394962768E4
        ),
        "ErfU": (
            "000000000000f03f",  # 1.00000000000000000000E0
            "baa6ef3fe6c74040",  # 3.35617141647503099647E1
            "ee3ac614dd4a8040",  # 5.21357949780152679795E2
            "12fd80e652f2b140",  # 4.59432382970980127987E3
            "0a7c01014019d640",  # 2.26290000613890934246E4
            "67f5c89d6c0ee840",  # 4.92673942608635921086E4
        ),
        "ErfcP": (
            "79644fa224ebf03d",  # 2.46196981473530512524E-10
            "88343646d70de23f",  # 5.64189564831068821977E-1
            "c46dc5de53da1d40",  # 7.46321056442269912687E0
            "66baadac8f514840",  # 4.86371970985681366614E1
            "f720e0a9aa906840",  # 1.96520832956077098242E2
            "58cf64c28f738040",  # 5.26445194995477358631E2
            "d834746c3a348d40",  # 9.34528527171957607540E2
            "2a97d621350e9040",  # 1.02755188689515710272E3
            "b3ffe85d486c8140",  # 5.57535335369399327526E2
        ),
        "ErfcQ": (
            "000000000000f03f",  # 1.00000000000000000000E0
            "cc237cfdd5742a40",  # 1.32281951154744992508E1
            "6573e1fe42ad5540",  # 8.67072140885989742329E1
            "0d616f24012f7640",  # 3.54937778887819891062E2
            "d041f602ab7d8e40",  # 9.75708501743205489753E2
            "5171a4fca27f9c40",  # 1.82390916687909736289E3
            "fff4fadaac8ca140",  # 2.24633760818710981792E3
            "e2ed9201a7e29940",  # 1.65666309194161350182E3
            "d642c460486c8140",  # 5.57535340817727675546E2
        ),
        "ErfcR": (
            "629b4250d70de23f",  # 5.64189583547755073984E-1
            "a6c5b8ebe667f43f",  # 1.27536670759978104416E0
            "1aa736f481131440",  # 5.01905042251180477414E0
            "0c0cdd580ea41840",  # 6.16021097993053585195E0
            "0e96189793a31d40",  # 7.40974269950448939160E0
            "6703470ab8d40740",  # 2.97886665372100240670E0
        ),
        "ErfcS": (
            "000000000000f03f",  # 1.00000000000000000000E0
            "21ce170990150240",  # 2.26052863220117276590E0
            "fd4cd821c5ca2240",  # 9.39603524938001434673E0
            "4d2c057f10192840",  # 1.20489539808096656605E1
            "f57f9c95d9143140",  # 1.70814450747565897222E1
            "c12663aaca372340",  # 9.60896809063285878198E0
            "902db65adef30a40",  # 3.36907645100081516050E0
        ),
    },
}

_SCALARS: Final[dict[ir.DataType, dict[str, str]]] = {
    ir.DataType.FLOAT: {
        "one": "0000803f",  # 1
        "zero": "00000000",  # 0
        "two": "00000040",  # 2
        "negative_maxlog": "1872b1c2",  # -88.72283905206835
    },
    ir.DataType.DOUBLE: {
        "one": "000000000000f03f",  # 1
        "zero": "0000000000000000",  # 0
        "two": "0000000000000040",  # 2
        "negative_maxlog": "ef39fafe422e86c0",  # -7.09782712893383996843E2
        "eight": "0000000000002040",  # 8
    },
}


class _ErfcExpression:
    def __init__(self, ctx: NativeLoweringContext, x: ir.Value) -> None:
        dtype = x.dtype
        if dtype is not ir.DataType.FLOAT and dtype is not ir.DataType.DOUBLE:
            raise ValueError("The Erfc expression requires FLOAT or DOUBLE input")
        self.ctx = ctx
        self.dtype = dtype
        self.shape = x.shape
        self.constants: dict[str, ir.Value] = {}

    def op(
        self, op: str, *inputs: ir.Value, output: ir.Value | None = None
    ) -> ir.Value:
        if output is None:
            output = ir.Value(
                name=self.ctx.fresh_name(f"erfc_{op.lower()}"),
                type=ir.TensorType(ir.DataType.BOOL if op == "Less" else self.dtype),
                shape=self.shape,
            )
        return self.ctx.builder.op(op, inputs, output=output)

    def constant(self, key: str, raw: str | None = None) -> ir.Value:
        if key not in self.constants:
            raw = _SCALARS[self.dtype][key] if raw is None else raw
            dtype = "<f4" if self.dtype is ir.DataType.FLOAT else "<f8"
            scalar: np.ndarray[Any, np.dtype[np.float32 | np.float64]] = np.frombuffer(
                bytes.fromhex(raw), dtype=dtype
            ).reshape(())
            self.constants[key] = self.ctx.builder.add_initializer_from_array(
                name=self.ctx.fresh_name(f"erfc_{key}"), array=scalar
            )
        return self.constants[key]

    def polynomial(self, x: ir.Value, group: str) -> ir.Value:
        coefficients = _COEFFICIENTS[self.dtype][group]
        result = self.constant(f"{group}_0", coefficients[0])
        for index, coefficient in enumerate(coefficients[1:], 1):
            result = self.op("Mul", result, x)
            result = self.op(
                "Add", result, self.constant(f"{group}_{index}", coefficient)
            )
        return result

    def emit(self, x: ir.Value, output: ir.Value) -> ir.Value:
        one, zero, two = (self.constant(key) for key in ("one", "zero", "two"))
        square = self.op("Mul", x, x)
        z = self.op("Neg", square)
        absolute = self.op("Abs", x)
        exponential = self.op("Exp", z)
        if self.dtype is ir.DataType.FLOAT:
            q = self.op("Div", one, square)
            reciprocal = self.op("Div", one, absolute)
            exponential_reciprocal = self.op("Mul", exponential, reciprocal)
            below_two = self.op("Less", absolute, two)
            p = self.polynomial(q, "ErfcP")
            r = self.polynomial(q, "ErfcR")
            selected = self.op("Where", below_two, p, r)
            tail = self.op("Mul", exponential_reciprocal, selected)
        else:
            p = self.polynomial(absolute, "ErfcP")
            ep = self.op("Mul", exponential, p)
            q = self.polynomial(absolute, "ErfcQ")
            pq = self.op("Div", ep, q)
            r = self.polynomial(absolute, "ErfcR")
            er = self.op("Mul", exponential, r)
            s = self.polynomial(absolute, "ErfcS")
            rs = self.op("Div", er, s)
            below_eight = self.op("Less", absolute, self.constant("eight"))
            tail = self.op("Where", below_eight, pq, rs)
        clamped = self.op("Less", z, self.constant("negative_maxlog"))
        tail = self.op("Where", clamped, zero, tail)
        negative = self.op("Less", x, zero)
        reflected = self.op("Sub", two, tail)
        large = self.op("Where", negative, reflected, tail)
        t = self.polynomial(square, "ErfT")
        small = self.op("Mul", x, t)
        if self.dtype is ir.DataType.DOUBLE:
            u = self.polynomial(square, "ErfU")
            small = self.op("Div", small, u)
        small = self.op("Sub", one, small)
        below_one = self.op("Less", absolute, one)
        return self.op("Where", below_one, small, large, output=output)


def source_erfc(ctx: NativeLoweringContext, x: ir.Value, output: ir.Value) -> ir.Value:
    """Emit the unchanged same-type body; the owning plugin adapts low precision."""
    return _ErfcExpression(ctx, x).emit(x, output)
