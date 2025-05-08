import unittest
import jax
from jax.core import ShapedArray
from onnx import helper, TensorProto
import numpy as np
import logging

import onnx


from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.name_generator import UniqueNameGenerator
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


class TestConverterDoesNotPromoteIntermediates(unittest.TestCase):
    def test_intermediate_not_promoted(self):
        ng = UniqueNameGenerator()
        builder = OnnxBuilder(ng, model_name="test")

        A = ng.get("A_int64")
        builder.add_scalar_input(A, TensorProto.INT64)
        B = ng.get("B_int32")
        builder.add_value_info(B, (), np.int32)
        builder.add_node(helper.make_node("Cast", [A], [B], to=TensorProto.INT32))

        def passthrough(x):
            return x + np.int32(0)  # forces new Var

        jaxpr = jax.make_jaxpr(passthrough)(np.int32(0)).jaxpr

        conv = Jaxpr2OnnxConverter(builder)
        conv.var_to_name[jaxpr.invars[0]] = B
        C = ng.get("C_int32")
        # tell the converter to *really* use a fresh output symbol
        conv.var_to_name[jaxpr.outvars[0]] = C

        conv._process_jaxpr(jaxpr, [])

        # builder itself should not have had to add Identity – converter must.
        producer = [n for n in builder.nodes if C in n.output]
        self.assertEqual(
            len(producer), 1, "converter must create an Identity producing C"
        )

        graph = builder.create_graph("subgraph", is_subgraph=True)

        self.assertEqual([vi.name for vi in graph.input], [A])  # only the true input

        model = builder.create_onnx_model("noPromote_intermediates")
        onnx.save_model(model, "docs/onnx/test_noPromote_intermediates.onnx")

        inputs = [vi.name for vi in graph.input]
        self.assertEqual(inputs, [A])  # only the real input
        self.assertNotIn(B, inputs)  # B stayed intermediate
