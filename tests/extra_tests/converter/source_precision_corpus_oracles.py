# tests/extra_tests/converter/source_precision_corpus_oracles.py

"""Fixed corpus structural counterparts and proposal-only numerical provenance.

These checks preserve raw callback failures. They neither admit a primitive nor
supply numerical criteria to a source scenario or execute runtime samples.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Final, TYPE_CHECKING

from flax import nnx
import numpy as np
import onnx

from jax2onnx._compat import jax as jax_compat
from jax2onnx.precision import CheckStatus
from scripts.source_precision_inventory import canonical_contract

if TYPE_CHECKING:
    from tests.extra_tests.converter.source_precision_support import PreparedSourceCase


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]
HISTORICAL_BASELINE_SHA256: Final = (
    "9ac003ba1e05b94cec26c16985892c506b617909534de320e223796c3cc8cd2b"
)
APPROVED_SELECTOR_SHA256: Final = (
    "1b4f2f47d8651d8ee4e7967e7654af520c11d40f96c4836178e6ecd9720d8d09"
)
DOT_SOURCE_PATH: Final = "jax2onnx/plugins/jax/lax/dot_general.py"
DOT_SOURCE_SHA256: Final = (
    "6433dfb3292cb8966e4b11d2f5f677f78a86dc50311298eb41cd236cabd4ec0b"
)
ORACLE_VERSION: Final = "static-dot-source-counterpart-v3"
LINEAR_SOURCE_PATH: Final = "jax2onnx/plugins/flax/nnx/linear.py"
LINEAR_SOURCE_SHA256: Final = (
    "3a3e17bd6d3ad1e259f0ebe293d8b303fec58e37569d5271515eef5b433660ba"
)
LINEAR_ORACLE_VERSION: Final = "leading-b-linear-source-counterpart-v2"
# Historical descriptors retain their own complete module/source provenance.
# They are evidence only; current checks below bind the new source and versions.
# The name-only emitter refresh keeps the same semantic oracle. Its predecessor
# source/module fences remain historical and never admit current artifacts.
HISTORICAL_DOT_SOURCE_SHA256_BEFORE_FRESH_NAMES: Final = (
    "96a458f4294fde2f3ec3c596d80a4188b36c422b4b19f22181f305ee59c8d4b3"
)
HISTORICAL_ORACLE_FILE_SHA256_BEFORE_FRESH_NAMES: Final = (
    "f9ed94b8417bc4ab5fa5442e108e87b01f138e38be56136806e8b10796b8f014"
)
HISTORICAL_DOT_SOURCE_PATH_V2: Final = "jax2onnx/plugins/jax/lax/dot_general.py"
HISTORICAL_DOT_SOURCE_SHA256_V2: Final = (
    "82098b7ecc569eef6bfbd54577b681dc151f026ea9fcbc11b35da00070fc5d45"
)
HISTORICAL_DOT_ORACLE_VERSION_V2: Final = "static-dot-source-counterpart-v2"
HISTORICAL_LINEAR_SOURCE_SHA256_V1: Final = (
    "3a3e17bd6d3ad1e259f0ebe293d8b303fec58e37569d5271515eef5b433660ba"
)
HISTORICAL_LINEAR_ORACLE_VERSION_V1: Final = "leading-b-linear-source-counterpart-v1"
HISTORICAL_ORACLE_FILE_SHA256_DOT_V2_LINEAR_V1: Final = (
    "d2d629683540ca5748a97c7ee6491eeee6f4e688a084772fbc8151a7fac58814"
)
HISTORICAL_DOT_DESCRIPTORS_V2: Final[tuple[dict[str, Any], ...]] = (
    {
        "identifier": "static-dot-source-counterpart-v2:plugin::dot_general:0",
        "sha256": "b3b92a7d119f7f588e6a84fd444a6c7c6bc15282c8a9afc5a6a4fb679e138dc1",
        "case": {
            "origin": "plugin::dot_general",
            "ordinal": 0,
            "testcase": "dot_contract_nm",
            "contract_sha256": "6957d2d4640c79d15f46c7e930fcbd67467e3b278929165cf8619e6f88c7f8d4",
        },
        "lhs_shape": (3, 4),
        "rhs_shape": (4, 2),
        "output_shape": (3, 2),
        "rhs_contract_axis": 0,
    },
    {
        "identifier": "static-dot-source-counterpart-v2:plugin::dot_general:1",
        "sha256": "3cad1249d237ca602511bfd27e986be814e3b74864946c3e83f1e8ddb71286d0",
        "case": {
            "origin": "plugin::dot_general",
            "ordinal": 1,
            "testcase": "dot_contract_min",
            "contract_sha256": "b3da3ae21dc080cb6acf7f35b9641292804557f024c77bc7bd7058d3487df0a7",
        },
        "lhs_shape": (3, 4),
        "rhs_shape": (2, 4),
        "output_shape": (3, 2),
        "rhs_contract_axis": 1,
    },
    {
        "identifier": "static-dot-source-counterpart-v2:plugin::dot_general:2",
        "sha256": "e058a9f9f349d6662e124f64b055d744a3ff30e8aa9ff5c90c8daf188ad6bd4a",
        "case": {
            "origin": "plugin::dot_general",
            "ordinal": 2,
            "testcase": "dot_general",
            "contract_sha256": "1f34685812f3e38d4c6f48c426bb9e4622802dc95dd0045e3c28b3b5fe1a4762",
        },
        "lhs_shape": (3, 3),
        "rhs_shape": (3, 3),
        "output_shape": (3, 3),
        "rhs_contract_axis": 0,
    },
    {
        "identifier": "static-dot-source-counterpart-v2:plugin::dot_general:3",
        "sha256": "c1c81ccf25c574d2fe3c758ee5d78bce8abc0b0f7355101f664746d1685e7037",
        "case": {
            "origin": "plugin::dot_general",
            "ordinal": 3,
            "testcase": "dot_general_lhs1_rhs1",
            "contract_sha256": "1f34685812f3e38d4c6f48c426bb9e4622802dc95dd0045e3c28b3b5fe1a4762",
        },
        "lhs_shape": (3, 3),
        "rhs_shape": (3, 3),
        "output_shape": (3, 3),
        "rhs_contract_axis": 1,
    },
)
HISTORICAL_LINEAR_DESCRIPTORS_V1: Final[tuple[dict[str, Any], ...]] = (
    {
        "identifier": "leading-b-linear-source-counterpart-v1:plugin::nnx.linear:0",
        "sha256": "b16e8e24ce038fb9d15ec4030f5c03095ebbd25adc15186049080be17f8906ed",
        "case": {
            "origin": "plugin::nnx.linear",
            "ordinal": 0,
            "testcase": "linear_symbolic_batch",
            "contract_sha256": "8c70628910c9d9bc30737e3bb959dafc3dea41c24eecfd7cc1853f076519a515",
        },
        "use_bias": True,
    },
    {
        "identifier": "leading-b-linear-source-counterpart-v1:plugin::nnx.linear:3",
        "sha256": "06632624eaa4086e92f2c9ea0e52d07ffa4541d9799b5e0f96c1d1ade47569d4",
        "case": {
            "origin": "plugin::nnx.linear",
            "ordinal": 3,
            "testcase": "linear_no_bias",
            "contract_sha256": "06c91c76e73a3daced2f70b65cb56d9e59797008354dababdcb3d4d2bebc77ef",
        },
        "use_bias": False,
    },
)

GATHER_SOURCE_PATH: Final = "jax2onnx/plugins/jax/lax/gather.py"
GATHER_SOURCE_SHA256: Final = (
    "b999da7239abf9dff0d1e19df611f3a79ad9434da5f2c7547091a00f2a68f907"
)
GATHER_ORACLE_VERSION: Final = "gather-trig-native-copy-source-counterpart-v2"
# Complete original native traces, including index normalization and Where JIT.
_GATHER_NATIVE_TRACE_SHA256: Final = (
    "9a14661837181e7170fd3d2e0a780d7fd927299c4c96349ad05cb185926f3cda",
    "44fef1e2b0e2260469c30e76bf4186b75b73a4c0f52d7de9153bbbf61e8fcac1",
)
# Fully concrete original trace measured on the native Mul schema without
# out_dtype. Its Boolean inline hint and all original payloads remain exact.
_GATHER_NATIVE_TRACE_NO_OUT_DTYPE_SHA256: Final = (
    "98f139d5ecf0b043c57b1605a3eb114245d42129ddb8a130839016d337823457"
)
_GATHER_NATIVE_FALSE_INLINE_HINT: Final = jax_compat.native_jit_boolean_inline_hints()[
    0
]
_GATHER_VALUE_SHA256: Final = (
    "924dee132820525df56cbdc03823b4622404d1d9bd32a741780427b73478efcb",
    "709f61b4a913b4cfd4c4bd8e61a12c894f315137de193462b74e1e2645bd5bfc",
    "61bae0409f845bbc23a7e636f8062ac26cdec8b5e538a328b0009d07f42457e9",
)
# Observed through the existing export path on both supported stacks. Port and
# producer roles replace generated names; constants retain every C-order bit.
# The native Cos branch is the existing FP64 1 - 2*sin(shared*0.5)^2 lowering.
# This fixed structure supplies no floating-point equivalence/runtime claim.
# ONNX types: DOUBLE=11, INT64=7, INT32=6, BOOL=9; INT attributes have type 2.
_GATHER_GRAPH_TEMPLATE_V1: Final[dict[str, Any]] = json.loads(
    """{
  "inputs": [[11,[4,3]],[7,[2]]],
  "constants": [
    {"dtype":7,"shape":[],"bytes":"0000000000000000"},
    {"dtype":7,"shape":[],"bytes":"0400000000000000"},
    {"dtype":7,"shape":[1],"bytes":"0200000000000000"},
    {"dtype":7,"shape":[1],"bytes":"0100000000000000"},
    {"dtype":7,"shape":[1],"bytes":"0200000000000000"},
    {"dtype":7,"shape":[1],"bytes":"0100000000000000"},
    {"dtype":7,"shape":[1],"bytes":"0200000000000000"},
    {"dtype":11,"shape":[],"bytes":"0000000000000040"},
    {"dtype":11,"shape":[],"bytes":"000000000000e03f"},
    {"dtype":11,"shape":[],"bytes":"0000000000000040"},
    {"dtype":11,"shape":[],"bytes":"000000000000f03f"},
    {"dtype":11,"shape":[],"bytes":"000000000000e03f"},
    {"dtype":11,"shape":[2,3],"bytes":"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"}
  ],
  "nodes": [
    {"op":"Less","inputs":["input1","constant0"],"outputs":[[9,[2]]],"attributes":{}},
    {"op":"Add","inputs":["input1","constant1"],"outputs":[[7,[2]]],"attributes":{}},
    {"op":"Where","inputs":["node0:0","node1:0","input1"],"outputs":[[7,[2]]],"attributes":{}},
    {"op":"Cast","inputs":["node2:0"],"outputs":[[6,[2]]],"attributes":{"to":[2,6]}},
    {"op":"Concat","inputs":["constant2","constant3"],"outputs":[[7,[2]]],"attributes":{"axis":[2,0]}},
    {"op":"Concat","inputs":["constant4","constant5"],"outputs":[[7,[2]]],"attributes":{"axis":[2,0]}},
    {"op":"Reshape","inputs":["node3:0","node5:0"],"outputs":[[6,[2,1]]],"attributes":{}},
    {"op":"Expand","inputs":["node6:0","node4:0"],"outputs":[[6,[2,1]]],"attributes":{}},
    {"op":"Reshape","inputs":["node7:0","constant6"],"outputs":[[6,[2]]],"attributes":{}},
    {"op":"Gather","inputs":["input0","node8:0"],"outputs":[[11,[2,3]]],"attributes":{"axis":[2,0]}},
    {"op":"Mul","inputs":["node9:0","constant7"],"outputs":[[11,[2,3]]],"attributes":{}},
    {"op":"Sin","inputs":["node10:0"],"outputs":[[11,[2,3]]],"attributes":{}},
    {"op":"Mul","inputs":["node10:0","constant8"],"outputs":[[11,[2,3]]],"attributes":{}},
    {"op":"Sin","inputs":["node12:0"],"outputs":[[11,[2,3]]],"attributes":{}},
    {"op":"Mul","inputs":["node13:0","node13:0"],"outputs":[[11,[2,3]]],"attributes":{}},
    {"op":"Mul","inputs":["node14:0","constant9"],"outputs":[[11,[2,3]]],"attributes":{}},
    {"op":"Sub","inputs":["constant10","node15:0"],"outputs":[[11,[2,3]]],"attributes":{}},
    {"op":"Add","inputs":["node11:0","node16:0"],"outputs":[[11,[2,3]]],"attributes":{}},
    {"op":"Greater","inputs":["node17:0","constant11"],"outputs":[[9,[2,3]]],"attributes":{}},
    {"op":"Where","inputs":["node18:0","node17:0","constant12"],"outputs":[[11,[2,3]]],"attributes":{}}
  ],
  "outputs": [["node19:0",[11,[2,3]]]]
} """
)


# Current v2 was captured before this oracle edit on both supported stacks.
# The false/true FP64 arms are stacked on axis 0 and selected by INT32 indices;
# GatherElements and Squeeze copy the chosen bits without floating arithmetic.
_GATHER_GRAPH_TEMPLATE_V2: Final[dict[str, Any]] = json.loads(
    """{
  "inputs": [[11,[4,3]],[7,[2]]],
  "constants": [
    {"bytes":"0000000000000000","dtype":7,"shape":[]},
    {"bytes":"0400000000000000","dtype":7,"shape":[]},
    {"bytes":"0200000000000000","dtype":7,"shape":[1]},
    {"bytes":"0100000000000000","dtype":7,"shape":[1]},
    {"bytes":"0200000000000000","dtype":7,"shape":[1]},
    {"bytes":"0100000000000000","dtype":7,"shape":[1]},
    {"bytes":"0200000000000000","dtype":7,"shape":[1]},
    {"bytes":"0000000000000040","dtype":11,"shape":[]},
    {"bytes":"000000000000e03f","dtype":11,"shape":[]},
    {"bytes":"0000000000000040","dtype":11,"shape":[]},
    {"bytes":"000000000000f03f","dtype":11,"shape":[]},
    {"bytes":"000000000000e03f","dtype":11,"shape":[]},
    {"bytes":"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000","dtype":11,"shape":[2,3]},
    {"bytes":"0000000000000000","dtype":7,"shape":[1]}
  ],
  "nodes": [
    {"attributes":{},"inputs":["input1","constant0"],"op":"Less","outputs":[[9,[2]]]},
    {"attributes":{},"inputs":["input1","constant1"],"op":"Add","outputs":[[7,[2]]]},
    {"attributes":{},"inputs":["node0:0","node1:0","input1"],"op":"Where","outputs":[[7,[2]]]},
    {"attributes":{"to":[2,6]},"inputs":["node2:0"],"op":"Cast","outputs":[[6,[2]]]},
    {"attributes":{"axis":[2,0]},"inputs":["constant2","constant3"],"op":"Concat","outputs":[[7,[2]]]},
    {"attributes":{"axis":[2,0]},"inputs":["constant4","constant5"],"op":"Concat","outputs":[[7,[2]]]},
    {"attributes":{},"inputs":["node3:0","node5:0"],"op":"Reshape","outputs":[[6,[2,1]]]},
    {"attributes":{},"inputs":["node6:0","node4:0"],"op":"Expand","outputs":[[6,[2,1]]]},
    {"attributes":{},"inputs":["node7:0","constant6"],"op":"Reshape","outputs":[[6,[2]]]},
    {"attributes":{"axis":[2,0]},"inputs":["input0","node8:0"],"op":"Gather","outputs":[[11,[2,3]]]},
    {"attributes":{},"inputs":["node9:0","constant7"],"op":"Mul","outputs":[[11,[2,3]]]},
    {"attributes":{},"inputs":["node10:0"],"op":"Sin","outputs":[[11,[2,3]]]},
    {"attributes":{},"inputs":["node10:0","constant8"],"op":"Mul","outputs":[[11,[2,3]]]},
    {"attributes":{},"inputs":["node12:0"],"op":"Sin","outputs":[[11,[2,3]]]},
    {"attributes":{},"inputs":["node13:0","node13:0"],"op":"Mul","outputs":[[11,[2,3]]]},
    {"attributes":{},"inputs":["node14:0","constant9"],"op":"Mul","outputs":[[11,[2,3]]]},
    {"attributes":{},"inputs":["constant10","node15:0"],"op":"Sub","outputs":[[11,[2,3]]]},
    {"attributes":{},"inputs":["node11:0","node16:0"],"op":"Add","outputs":[[11,[2,3]]]},
    {"attributes":{},"inputs":["node17:0","constant11"],"op":"Greater","outputs":[[9,[2,3]]]},
    {"attributes":{"to":[2,6]},"inputs":["node18:0"],"op":"Cast","outputs":[[6,[2,3]]]},
    {"attributes":{},"inputs":["constant12","constant13"],"op":"Unsqueeze","outputs":[[11,[1,2,3]]]},
    {"attributes":{},"inputs":["node17:0","constant13"],"op":"Unsqueeze","outputs":[[11,[1,2,3]]]},
    {"attributes":{"axis":[2,0]},"inputs":["node20:0","node21:0"],"op":"Concat","outputs":[[11,[2,2,3]]]},
    {"attributes":{},"inputs":["node19:0","constant13"],"op":"Unsqueeze","outputs":[[6,[1,2,3]]]},
    {"attributes":{"axis":[2,0]},"inputs":["node22:0","node23:0"],"op":"GatherElements","outputs":[[11,[1,2,3]]]},
    {"attributes":{},"inputs":["node24:0","constant13"],"op":"Squeeze","outputs":[[11,[2,3]]]}
  ],
  "outputs": [["node25:0",[11,[2,3]]]]
} """
)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CorpusCase:
    origin: str
    ordinal: int
    testcase: str
    contract_sha256: str


CORPUS_REPLAY_CASES: Final[tuple[CorpusCase, ...]] = (
    CorpusCase(
        "plugin::dot_general",
        0,
        "dot_contract_nm",
        "6957d2d4640c79d15f46c7e930fcbd67467e3b278929165cf8619e6f88c7f8d4",
    ),
    CorpusCase(
        "plugin::dot_general",
        1,
        "dot_contract_min",
        "b3da3ae21dc080cb6acf7f35b9641292804557f024c77bc7bd7058d3487df0a7",
    ),
    CorpusCase(
        "plugin::dot_general",
        2,
        "dot_general",
        "1f34685812f3e38d4c6f48c426bb9e4622802dc95dd0045e3c28b3b5fe1a4762",
    ),
    CorpusCase(
        "plugin::dot_general",
        3,
        "dot_general_lhs1_rhs1",
        "1f34685812f3e38d4c6f48c426bb9e4622802dc95dd0045e3c28b3b5fe1a4762",
    ),
    CorpusCase(
        "plugin::dot_general",
        6,
        "dot_highrank_batch",
        "317e5bfb607ff4263a14445fc375284b571518af5041a453260fd9c7d64601b5",
    ),
    CorpusCase(
        "plugin::gather",
        0,
        "gather_trig_where_pipeline_f64_indices_i64",
        "ab7d1021376f14588c8272d8bd53a961c18f4fcc9b1063fa31fa98b52ac39c39",
    ),
    CorpusCase(
        "plugin::jax.numpy.add",
        0,
        "add",
        "39f16c5258b9354829fd94c8825deb0e8a389215d0be1fcced026de8ae82a922",
    ),
    CorpusCase(
        "plugin::jax.numpy.add",
        1,
        "jnp_add_vector",
        "39f16c5258b9354829fd94c8825deb0e8a389215d0be1fcced026de8ae82a922",
    ),
    CorpusCase(
        "plugin::jax.numpy.add",
        2,
        "jnp_add_broadcast",
        "ba4d75c4de26c069de03f566d39f6808007f77df4ddf63b576e72deb2989e9c3",
    ),
    CorpusCase(
        "plugin::jax.numpy.add",
        3,
        "add_vmap_batching",
        "f08ffd2b45e50b4ce00b1ecff2e3f1622475f7ccf555a6a74d93136ddff6c240",
    ),
    CorpusCase(
        "plugin::jax.numpy.matmul",
        2,
        "matmul_2d",
        "e202a64ef5c979de1159983b35f9caded7b1a893e9b3b7492f844b0148aaa129",
    ),
    CorpusCase(
        "plugin::jax.numpy.matmul",
        4,
        "matmul_3d",
        "f48a5efbae62237d688e16e37a36f250963ce7552d28a0fdbe225328a276baf8",
    ),
    CorpusCase(
        "plugin::jax.numpy.matmul",
        8,
        "matmul_vmap_batching",
        "0de27237b463f38d80289cb2d3def996bd0e327cbefbb070c40c5d690d00c6f3",
    ),
)


def selector_manifest() -> dict[str, Any]:
    return {
        "version": "static-matrix-add-gather-v1",
        "approved_selector_sha256": APPROVED_SELECTOR_SHA256,
        "cases": [asdict(case) for case in CORPUS_REPLAY_CASES],
    }


@dataclass(frozen=True)
class CorpusStructureEvidence:
    counterpart_identifier: str
    counterpart_sha256: str
    raw_callback_status: str
    raw_callback_detail: str
    source_structure: CheckStatus
    source_detail: str

    @property
    def accepted(self) -> bool:
        return (
            self.source_structure is CheckStatus.VALIDATED
            and self.raw_callback_status in {"passed", "failed"}
        )


@dataclass(frozen=True)
class DotCounterpart:
    case: CorpusCase
    lhs_shape: tuple[int, int]
    rhs_shape: tuple[int, int]
    output_shape: tuple[int, int]
    rhs_contract_axis: int

    @property
    def identifier(self) -> str:
        return f"{ORACLE_VERSION}:{self.case.origin}:{self.case.ordinal}"

    @property
    def sha256(self) -> str:
        return fingerprint(
            {
                "identifier": self.identifier,
                "mapping": asdict(self),
                "source_sha256": DOT_SOURCE_SHA256,
                "oracle_file_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )

    def descriptor(self) -> dict[str, Any]:
        return {"identifier": self.identifier, "sha256": self.sha256, **asdict(self)}

    def _check_source(self, prepared: PreparedSourceCase) -> None:
        provenance = prepared.scenario.provenance
        if provenance is None or (provenance.origin, provenance.ordinal) != (
            self.case.origin,
            self.case.ordinal,
        ):
            raise ValueError("Counterpart provenance does not match the fixed raw row")
        if prepared.metadata.get("testcase") != self.case.testcase:
            raise ValueError("Counterpart testcase does not match the fixed raw row")
        unresolved: list[str] = []
        contract = canonical_contract(
            {
                key: value
                for key, value in prepared.metadata.items()
                if key not in {"testcase", "description", "context", "component"}
            },
            unresolved=unresolved,
        )
        if unresolved or fingerprint(contract) != self.case.contract_sha256:
            raise ValueError("Counterpart raw contract changed")
        source = Path(inspect.getsourcefile(prepared.fn) or "").resolve()
        if (
            source != REPOSITORY_ROOT / DOT_SOURCE_PATH
            or hashlib.sha256(source.read_bytes()).hexdigest() != DOT_SOURCE_SHA256
        ):
            raise ValueError("Counterpart original callable source bytes/path changed")
        inputs = [
            ["float32", list(shape), False]
            for shape in (self.lhs_shape, self.rhs_shape)
        ]
        output = {"var": 2, "aval": ["float32", list(self.output_shape), False]}
        expected = {
            "inputs": inputs,
            "consts": [],
            "const_payloads": [],
            "outputs": [output],
            "effects": [],
            "unresolved_parameters": [],
            "equations": [
                {
                    "primitive": "dot_general",
                    "inputs": [
                        {"var": index, "aval": aval}
                        for index, aval in enumerate(inputs)
                    ],
                    "outputs": [output],
                    "params": {
                        "dimension_numbers": [
                            [[1], [self.rhs_contract_axis]],
                            [[], []],
                        ],
                        "out_sharding": None,
                        "precision": None,
                        "preferred_element_type": None,
                    },
                    "effects": [],
                }
            ],
        }
        if json.loads(prepared.original_trace_json) != expected:
            raise ValueError(
                "Counterpart requires the exact reviewed typed dot topology/axes/parameters"
            )
        if prepared.settings.default_matmul_precision is not None:
            raise ValueError(
                "Counterpart requires the unchanged raw matmul configuration"
            )

    def _check_model(self, model: onnx.ModelProto) -> None:
        graph = model.graph
        if (
            len(graph.input) != 2
            or len(graph.output) != 1
            or model.functions
            or graph.initializer
            or graph.sparse_initializer
        ):
            raise ValueError("Counterpart forbids extra ports, constants and Functions")
        for value, shape in zip(
            (*graph.input, *graph.output),
            (self.lhs_shape, self.rhs_shape, self.output_shape),
            strict=True,
        ):
            tensor = value.type.tensor_type
            if (
                tensor.elem_type != onnx.TensorProto.FLOAT
                or any(not dim.HasField("dim_value") for dim in tensor.shape.dim)
                or tuple(dim.dim_value for dim in tensor.shape.dim) != shape
            ):
                raise ValueError(
                    "Counterpart boundary dtype/shape differs from the source"
                )
        nodes = list(graph.node)
        expected_ops = (
            ["Transpose", "MatMul"] if self.rhs_contract_axis == 1 else ["MatMul"]
        )
        if [node.op_type for node in nodes] != expected_ops or any(
            node.domain for node in nodes
        ):
            raise ValueError(
                "Counterpart requires only the exact standard-domain MatMul path"
            )
        rhs = graph.input[1].name
        if self.rhs_contract_axis == 1:
            transpose = nodes[0]
            if (
                list(transpose.input) != [rhs]
                or len(transpose.output) != 1
                or len(transpose.attribute) != 1
                or transpose.attribute[0].name != "perm"
                or list(transpose.attribute[0].ints) != [1, 0]
            ):
                raise ValueError("Counterpart RHS transpose/permutation is incorrect")
            rhs = transpose.output[0]
        matmul = nodes[-1]
        if (
            matmul.attribute
            or list(matmul.input) != [graph.input[0].name, rhs]
            or list(matmul.output) != [graph.output[0].name]
        ):
            raise ValueError(
                "Counterpart MatMul operands/output are disconnected, swapped or biased"
            )

    def check(
        self, prepared: PreparedSourceCase, model: onnx.ModelProto
    ) -> CorpusStructureEvidence:
        callback = prepared.metadata.get("post_check_onnx_graph")
        raw_status, raw_detail = "absent", "No original structural callback"
        if callback is not None:
            try:
                passed = bool(callback(model))
                raw_status = "passed" if passed else "failed"
                raw_detail = "Original raw callback returned " + str(passed)
            except Exception as exc:
                raw_status, raw_detail = "error", f"{type(exc).__name__}: {exc}"
        try:
            if self not in DOT_COUNTERPARTS:
                raise ValueError("Unreviewed counterpart mapping")
            self._check_source(prepared)
            self._check_model(model)
            status, detail = (
                CheckStatus.VALIDATED,
                "Exact source-bound MatMul structure checked; no lowering/numerical audit",
            )
        except Exception as exc:
            status, detail = CheckStatus.FAILED, str(exc)
        return CorpusStructureEvidence(
            self.identifier, self.sha256, raw_status, raw_detail, status, detail
        )


DOT_COUNTERPARTS: Final[tuple[DotCounterpart, ...]] = tuple(
    DotCounterpart(case, lhs, rhs, out, axis)
    for case, lhs, rhs, out, axis in (
        (CORPUS_REPLAY_CASES[0], (3, 4), (4, 2), (3, 2), 0),
        (CORPUS_REPLAY_CASES[1], (3, 4), (2, 4), (3, 2), 1),
        (CORPUS_REPLAY_CASES[2], (3, 3), (3, 3), (3, 3), 0),
        (CORPUS_REPLAY_CASES[3], (3, 3), (3, 3), (3, 3), 1),
    )
)


@dataclass(frozen=True)
class LinearCounterpart:
    """Exact original native Linear with its actual captured parameter bytes."""

    case: CorpusCase
    use_bias: bool

    @property
    def identifier(self) -> str:
        return f"{LINEAR_ORACLE_VERSION}:{self.case.origin}:{self.case.ordinal}"

    @property
    def sha256(self) -> str:
        return fingerprint(
            {
                "identifier": self.identifier,
                "mapping": asdict(self),
                "source_sha256": LINEAR_SOURCE_SHA256,
                "dot_source_sha256": DOT_SOURCE_SHA256,
                "oracle_file_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )

    def descriptor(self) -> dict[str, Any]:
        return {"identifier": self.identifier, "sha256": self.sha256, **asdict(self)}

    def _parameters(self, prepared: PreparedSourceCase) -> tuple[np.ndarray, ...]:
        if type(prepared.fn) is not nnx.Linear:
            raise ValueError("Linear counterpart requires the actual native module")
        kernel = np.asarray(prepared.fn.kernel[...])
        bias = prepared.fn.bias
        if kernel.dtype != np.dtype(np.float32) or kernel.shape != (128, 64):
            raise ValueError("Linear captured kernel dtype/shape changed")
        if self.use_bias:
            if bias is None:
                raise ValueError("Linear captured bias is missing")
            array = np.asarray(bias[...])
            if array.dtype != np.dtype(np.float32) or array.shape != (64,):
                raise ValueError("Linear captured bias dtype/shape changed")
            return kernel, array
        if bias is not None:
            raise ValueError("Bias-free Linear must keep its actual bias None")
        return (kernel,)

    def _check_source(self, prepared: PreparedSourceCase) -> None:
        from tests.extra_tests.converter.source_precision_support import (
            PRESERVE_FACTORY_DTYPE,
        )

        provenance = prepared.scenario.provenance
        if provenance is None or (provenance.origin, provenance.ordinal) != (
            self.case.origin,
            self.case.ordinal,
        ):
            raise ValueError(
                "Linear counterpart provenance differs from the fixed raw row"
            )
        unresolved: list[str] = []
        contract = canonical_contract(
            {
                key: value
                for key, value in prepared.metadata.items()
                if key not in {"testcase", "description", "context", "component"}
            },
            unresolved=unresolved,
        )
        if (
            prepared.metadata.get("testcase") != self.case.testcase
            or unresolved
            or fingerprint(contract) != self.case.contract_sha256
        ):
            raise ValueError("Linear counterpart raw contract changed")
        recipe = prepared.metadata["callable"]
        if (
            recipe._ctor is not nnx.Linear
            or recipe._args != (128, 64)
            or recipe._dtype is not None
            or recipe._instance is not None
            or prepared.scenario.model_dtype is not PRESERVE_FACTORY_DTYPE
        ):
            raise ValueError(
                "Linear counterpart requires the unchanged raw constructor recipe"
            )
        if (
            hashlib.sha256(
                (REPOSITORY_ROOT / LINEAR_SOURCE_PATH).read_bytes()
            ).hexdigest()
            != LINEAR_SOURCE_SHA256
            or hashlib.sha256(
                (REPOSITORY_ROOT / DOT_SOURCE_PATH).read_bytes()
            ).hexdigest()
            != DOT_SOURCE_SHA256
        ):
            raise ValueError("Linear counterpart source module bytes changed")
        arrays = self._parameters(prepared)
        count = len(arrays)

        def var(index: int, shape: list[int | str]) -> dict[str, Any]:
            return {"var": index, "aval": ["float32", shape, False]}

        product = var(count + 1, ["B", 64])
        output = var(count + 3, ["B", 64]) if self.use_bias else product
        equations = [
            {
                "primitive": "dot_general",
                "inputs": [var(count, ["B", 128]), var(0, [128, 64])],
                "outputs": [product],
                "params": {
                    "dimension_numbers": [[[1], [0]], [[], []]],
                    "out_sharding": None,
                    "precision": None,
                    "preferred_element_type": None,
                },
                "effects": [],
            }
        ]
        if self.use_bias:
            reshaped = var(count + 2, [1, 64])
            equations.extend(
                [
                    {
                        "primitive": "reshape",
                        "inputs": [var(1, [64])],
                        "outputs": [reshaped],
                        "params": {
                            "dimensions": None,
                            "new_sizes": [1, 64],
                            "sharding": None,
                        },
                        "effects": [],
                    },
                    {
                        "primitive": "add",
                        "inputs": [product, reshaped],
                        "outputs": [output],
                        "params": {},
                        "effects": [],
                    },
                ]
            )
        expected = {
            "inputs": [["float32", ["B", 128], False]],
            "consts": [["float32", list(a.shape), False] for a in arrays],
            "const_payloads": [
                {
                    "array_dtype": "float32",
                    "shape": list(a.shape),
                    "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest(),
                    "weak_type": False,
                }
                for a in arrays
            ],
            "outputs": [output],
            "equations": equations,
            "effects": [],
            "unresolved_parameters": [],
        }
        if json.loads(prepared.original_trace_json) != expected:
            raise ValueError(
                "Linear counterpart requires exact native typed topology and captured bytes"
            )
        if prepared.settings.default_matmul_precision is not None:
            raise ValueError(
                "Linear counterpart requires unchanged raw matmul settings"
            )

    def _check_model(
        self, prepared: PreparedSourceCase, model: onnx.ModelProto
    ) -> None:
        graph = model.graph
        if (
            len(graph.input) != 1
            or len(graph.output) != 1
            or model.functions
            or graph.sparse_initializer
        ):
            raise ValueError("Linear counterpart forbids extra ports or Functions")
        for value, shape in zip(
            (*graph.input, *graph.output), (("B", 128), ("B", 64)), strict=True
        ):
            tensor = value.type.tensor_type
            actual = tuple(
                dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
                for dim in tensor.shape.dim
            )
            if tensor.elem_type != onnx.TensorProto.FLOAT or actual != shape:
                raise ValueError("Linear counterpart boundary dtype/shape changed")
        nodes = list(graph.node)
        ops = [n.op_type for n in nodes]
        if any(n.domain for n in nodes) or (
            ops not in (["MatMul", "Reshape", "Add"], ["MatMul", "Add"])
            if self.use_bias
            else ops != ["MatMul"]
        ):
            raise ValueError(
                "Linear counterpart requires exact native MatMul/bias structure"
            )
        tensors = {t.name: onnx.numpy_helper.to_array(t) for t in graph.initializer}
        if len(tensors) != len(graph.initializer):
            raise ValueError("Linear counterpart has duplicate initializer names")
        arrays = self._parameters(prepared)
        matmul = nodes[0]
        if (
            matmul.attribute
            or len(matmul.input) != 2
            or len(matmul.output) != 1
            or matmul.input[0] != graph.input[0].name
        ):
            raise ValueError("Linear MatMul does not consume the actual runtime input")
        used = {matmul.input[1]}

        def exact(name: str, expected: np.ndarray) -> None:
            actual = tensors.get(name)
            if (
                actual is None
                or actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or actual.tobytes(order="C") != expected.tobytes(order="C")
            ):
                raise ValueError(
                    "Linear initializer differs from actual captured parameters"
                )

        exact(matmul.input[1], arrays[0])
        if self.use_bias:
            add = nodes[-1]
            if (
                add.attribute
                or len(add.output) != 1
                or len(add.input) != 2
                or add.input[0] != matmul.output[0]
            ):
                raise ValueError("Linear native bias Add is disconnected or reordered")
            if ops == ["MatMul", "Reshape", "Add"]:
                reshape = nodes[1]
                if (
                    len(reshape.input) != 2
                    or len(reshape.output) != 1
                    or reshape.attribute
                    or reshape.output[0] != add.input[1]
                ):
                    raise ValueError("Linear bias reshape is disconnected or permuted")
                exact(reshape.input[0], arrays[1])
                exact(reshape.input[1], np.asarray([1, 64], np.int64))
                used.update(reshape.input)
            else:
                exact(add.input[1], arrays[1].reshape(1, 64))
                used.add(add.input[1])
            result = add.output[0]
        else:
            result = matmul.output[0]
        if result != graph.output[0].name or set(tensors) != used:
            raise ValueError(
                "Linear counterpart has unused constants or disconnected output"
            )

    def check(
        self, prepared: PreparedSourceCase, model: onnx.ModelProto
    ) -> CorpusStructureEvidence:
        callback = prepared.metadata.get("post_check_onnx_graph")
        raw_status, raw_detail = "absent", "No original structural callback"
        if callback is not None:
            try:
                passed = bool(callback(model))
                raw_status = "passed" if passed else "failed"
                raw_detail = "Original raw callback returned " + str(passed)
            except Exception as exc:
                raw_status, raw_detail = "error", f"{type(exc).__name__}: {exc}"
        try:
            if self not in LINEAR_COUNTERPARTS or raw_status != "failed":
                raise ValueError(
                    "Unreviewed Linear mapping or changed raw Gemm callback outcome"
                )
            self._check_source(prepared)
            self._check_model(prepared, model)
            status, detail = (
                CheckStatus.VALIDATED,
                "Exact native Linear parameters and MatMul/bias structure checked; numerics unverified",
            )
        except Exception as exc:
            status, detail = CheckStatus.FAILED, str(exc)
        return CorpusStructureEvidence(
            self.identifier, self.sha256, raw_status, raw_detail, status, detail
        )


LINEAR_COUNTERPARTS: Final[tuple[LinearCounterpart, ...]] = (
    LinearCounterpart(
        CorpusCase(
            "plugin::nnx.linear",
            0,
            "linear_symbolic_batch",
            "8c70628910c9d9bc30737e3bb959dafc3dea41c24eecfd7cc1853f076519a515",
        ),
        True,
    ),
    LinearCounterpart(
        CorpusCase(
            "plugin::nnx.linear",
            3,
            "linear_no_bias",
            "06c91c76e73a3daced2f70b65cb56d9e59797008354dababdcb3d4d2bebc77ef",
        ),
        False,
    ),
)


@lru_cache(maxsize=1)
def _gather_original_callables() -> tuple[Any, Any]:
    from tests.extra_tests.converter.source_precision_support import raw_source_case

    raw = raw_source_case("plugin::gather", 0)
    return raw["callable"], raw["post_check_onnx_graph"]


def _gather_graph_signature(model: onnx.ModelProto) -> dict[str, Any]:
    """Record this bounded graph's ports, producer edges, types and payloads."""
    if (
        model.functions
        or model.training_info
        or model.graph.sparse_initializer
        or model.graph.quantization_annotation
        or [(item.domain, item.version) for item in model.opset_import] != [("", 23)]
    ):
        raise ValueError("Gather counterpart requires only the reviewed standard graph")
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    graph = inferred.graph
    names: dict[str, str] = {}

    def bind(name: str, role: str) -> None:
        if not name or name in names:
            raise ValueError("Gather counterpart has missing or duplicate value names")
        names[name] = role

    def tensor(value: onnx.TensorProto) -> dict[str, Any]:
        if value.external_data or value.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError("Gather counterpart forbids external constant payloads")
        array = onnx.numpy_helper.to_array(value)
        return {
            "dtype": value.data_type,
            "shape": list(array.shape),
            "bytes": array.tobytes(order="C").hex(),
        }

    def properties(value: onnx.ValueInfoProto) -> list[Any]:
        if not value.type.HasField("tensor_type"):
            raise ValueError("Gather counterpart requires tensor values")
        typ = value.type.tensor_type
        if any(not dim.HasField("dim_value") for dim in typ.shape.dim):
            raise ValueError("Gather counterpart requires exact static dimensions")
        return [typ.elem_type, [dim.dim_value for dim in typ.shape.dim]]

    types = {
        value.name: properties(value)
        for value in (*graph.input, *graph.value_info, *graph.output)
    }
    inputs = []
    for index, value in enumerate(graph.input):
        bind(value.name, f"input{index}")
        inputs.append(properties(value))
    constants = []
    for index, value in enumerate(graph.initializer):
        bind(value.name, f"constant{index}")
        constants.append(tensor(value))
        types[value.name] = [value.data_type, list(value.dims)]
    nodes = []
    for index, node in enumerate(graph.node):
        if node.domain or node.overload:
            raise ValueError("Gather counterpart forbids custom operator domains")
        if any(name not in names for name in node.input):
            raise ValueError("Gather counterpart has an unbound producer edge")
        attributes: dict[str, Any] = {}
        for attribute in node.attribute:
            if attribute.name in attributes or attribute.ref_attr_name:
                raise ValueError(
                    "Gather counterpart has duplicate or indirect attributes"
                )
            if attribute.type == onnx.AttributeProto.TENSOR:
                payload: Any = tensor(attribute.t)
            elif attribute.type == onnx.AttributeProto.INT:
                payload = attribute.i
            elif attribute.type == onnx.AttributeProto.INTS:
                payload = list(attribute.ints)
            else:
                raise ValueError("Gather counterpart has an unreviewed attribute kind")
            attributes[attribute.name] = [attribute.type, payload]
        edges = [names[name] for name in node.input]
        outputs = []
        for ordinal, name in enumerate(node.output):
            bind(name, f"node{index}:{ordinal}")
            if name not in types:
                raise ValueError("Gather counterpart has an untyped intermediate")
            outputs.append(types[name])
        nodes.append(
            {
                "op": node.op_type,
                "inputs": edges,
                "outputs": outputs,
                "attributes": attributes,
            }
        )
    return {
        "inputs": inputs,
        "constants": constants,
        "nodes": nodes,
        "outputs": [[names[value.name], properties(value)] for value in graph.output],
    }


def _gather_native_trace_sha256() -> str:
    fields = jax_compat.native_mul_parameter_names()
    hint = jax_compat.native_jit_boolean_inline_hints()[0]
    if type(fields) is not frozenset or any(type(name) is not str for name in fields):
        raise ValueError("Gather counterpart native trace schema is unreviewed")
    if hint is False:
        if fields == frozenset():
            return _GATHER_NATIVE_TRACE_NO_OUT_DTYPE_SHA256
        if fields == frozenset({"out_dtype"}):
            return _GATHER_NATIVE_TRACE_SHA256[0]
    elif (
        fields == frozenset({"out_dtype"})
        and hint is _GATHER_NATIVE_FALSE_INLINE_HINT
        and isinstance(hint, Enum)
    ):
        unresolved: list[str] = []
        descriptor = canonical_contract(hint, unresolved=unresolved)
        if not unresolved and descriptor == {
            "enum": "jax._src.api.Inline",
            "name": "AUTO",
        }:
            return _GATHER_NATIVE_TRACE_SHA256[1]
    raise ValueError("Gather counterpart native trace schema is unreviewed")


@dataclass(frozen=True)
class GatherCounterpart:
    case: CorpusCase

    @property
    def identifier(self) -> str:
        return f"{GATHER_ORACLE_VERSION}:{self.case.origin}:{self.case.ordinal}"

    @property
    def sha256(self) -> str:
        return fingerprint(
            {
                "identifier": self.identifier,
                "mapping": asdict(self),
                "source_sha256": GATHER_SOURCE_SHA256,
                "oracle_file_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )

    def descriptor(self) -> dict[str, Any]:
        return {"identifier": self.identifier, "sha256": self.sha256, **asdict(self)}

    def _check_source(self, prepared: PreparedSourceCase) -> None:
        from tests.extra_tests.converter.source_precision_diagnostics import (
            typed_trace_manifest,
        )
        from tests.extra_tests.converter.source_precision_support import (
            PRESERVE_FACTORY_DTYPE,
            SourceProvenance,
            _properties,
            value_evidence,
        )

        if type(self) is not GatherCounterpart or not any(
            self is item for item in GATHER_COUNTERPARTS
        ):
            raise ValueError("Unreviewed Gather singleton mapping")
        scenario = prepared.scenario
        if (
            type(scenario.provenance) is not SourceProvenance
            or (scenario.provenance.origin, scenario.provenance.ordinal)
            != (self.case.origin, self.case.ordinal)
            or prepared.identity != ("primitives.lax", "gather", self.case.testcase)
            or prepared.metadata.get("context") != "primitives.lax"
            or prepared.metadata.get("component") != "gather"
            or prepared.metadata.get("testcase") != self.case.testcase
            or prepared.metadata.get("opset_version", 23) != 23
            or prepared.metadata.get("run_only_f64_variant") is not True
        ):
            raise ValueError("Gather counterpart fixed raw row identity changed")
        unresolved: list[str] = []
        contract = canonical_contract(
            {
                key: value
                for key, value in prepared.metadata.items()
                if key not in {"testcase", "description", "context", "component"}
            },
            unresolved=unresolved,
        )
        if unresolved or fingerprint(contract) != self.case.contract_sha256:
            raise ValueError("Gather counterpart raw contract changed")
        original, callback = _gather_original_callables()
        source = Path(inspect.getsourcefile(prepared.fn) or "").resolve()
        if (
            prepared.fn is not original
            or prepared.metadata.get("callable") is not original
            or prepared.metadata.get("post_check_onnx_graph") is not callback
            or source != REPOSITORY_ROOT / GATHER_SOURCE_PATH
            or hashlib.sha256(source.read_bytes()).hexdigest() != GATHER_SOURCE_SHA256
        ):
            raise ValueError(
                "Gather counterpart original callable/callback/source changed"
            )
        settings = prepared.settings
        if (
            scenario.name != "raw-x64-on"
            or scenario.caller_x64_enabled is not True
            or scenario.model_dtype is not PRESERVE_FACTORY_DTYPE
            or scenario.default_matmul_precision is not None
            or np.dtype(scenario.default_input_dtype) != np.dtype(np.float32)
            or scenario.criteria is not None
            or scenario.require_reviewed_criteria is not True
            or prepared.criteria is not None
            or settings.x64_enabled is not True
            or settings.default_matmul_precision is not None
            or settings.numpy_dtype_promotion != "standard"
            or settings.numpy_rank_promotion != "allow"
            or settings.jax_platforms not in (None, "cpu")
        ):
            raise ValueError(
                "Gather counterpart raw configuration/None criteria changed"
            )
        expected = (("float64", (4, 3)), ("int64", (2,)), ("float64", (2, 3)))
        ports = (*prepared.traced_inputs, *prepared.traced_outputs)
        if (
            len(prepared.traced_inputs) != 2
            or len(prepared.traced_outputs) != 1
            or any(
                (port.dtype, port.shape, port.weak_type) != (dtype, shape, False)
                for port, (dtype, shape) in zip(ports, expected, strict=True)
            )
        ):
            raise ValueError("Gather counterpart typed source ports changed")
        if (
            len(prepared.input_specs) != 2
            or scenario.input_specs != prepared.input_specs
            or tuple(_properties(spec) for spec in prepared.input_specs)
            != prepared.traced_inputs
            or scenario.concrete_shapes is not None
            or len(prepared.arguments) != 2
            or len(prepared.reference_outputs) != 1
            or prepared.output_tree != "PyTreeDef(*)"
            or scenario.input_values is None
            or len(scenario.input_values) != 2
        ):
            raise ValueError("Gather counterpart input recipe/specification changed")
        values = (*prepared.arguments, *prepared.reference_outputs)
        evidence = tuple(
            value_evidence(value, port)
            for value, port in zip(values, ports, strict=True)
        )
        if (
            evidence != (*prepared.input_evidence, *prepared.reference_evidence)
            or tuple(item.logical_sha256 for item in evidence) != _GATHER_VALUE_SHA256
            or any(
                _properties(value) != port
                for value, port in zip(values, ports, strict=True)
            )
            or any(
                _properties(value) != port or value_evidence(value, port) != item
                for value, port, item in zip(
                    scenario.input_values, ports[:2], evidence[:2], strict=True
                )
            )
        ):
            raise ValueError(
                "Gather counterpart actual feed/reference payload or weak type changed"
            )
        actual = json.loads(json.dumps(typed_trace_manifest(prepared.original_jaxpr)))
        if (
            json.loads(prepared.original_trace_json) != actual
            or fingerprint(actual) != _gather_native_trace_sha256()
        ):
            raise ValueError(
                "Gather counterpart complete original native trace changed"
            )

    def _check_model(self, model: onnx.ModelProto) -> None:
        if _gather_graph_signature(model) != _GATHER_GRAPH_TEMPLATE_V2:
            raise ValueError(
                "Gather counterpart requires the exact reviewed native graph"
            )

    def check(
        self, prepared: PreparedSourceCase, model: onnx.ModelProto
    ) -> CorpusStructureEvidence:
        callback = prepared.metadata.get("post_check_onnx_graph")
        raw_status, raw_detail = "absent", "No original structural callback"
        if callback is not None:
            try:
                passed = bool(callback(model))
                raw_status = "passed" if passed else "failed"
                raw_detail = "Original raw callback returned " + str(passed)
            except Exception as exc:
                raw_status, raw_detail = "error", f"{type(exc).__name__}: {exc}"
        try:
            self._check_source(prepared)
            if raw_status != "failed":
                raise ValueError(
                    "Gather counterpart requires unchanged raw callback failure"
                )
            self._check_model(model)
            status, detail = (
                CheckStatus.VALIDATED,
                "Exact original Gather/index/trig/native-copy structure checked; runtime and numerics unverified",
            )
        except Exception as exc:
            status, detail = CheckStatus.FAILED, str(exc)
        return CorpusStructureEvidence(
            self.identifier, self.sha256, raw_status, raw_detail, status, detail
        )


GATHER_COUNTERPARTS: Final[tuple[GatherCounterpart, ...]] = (
    GatherCounterpart(CORPUS_REPLAY_CASES[5]),
)


def counterpart_for(
    origin: str, ordinal: int, contract_sha256: str
) -> (
    DotCounterpart
    | LinearCounterpart
    | GatherCounterpart
    | NormalizationCounterpart
    | None
):
    return next(
        (
            item
            for item in (
                *DOT_COUNTERPARTS,
                *LINEAR_COUNTERPARTS,
                *GATHER_COUNTERPARTS,
                *NORMALIZATION_COUNTERPARTS,
            )
            if (item.case.origin, item.case.ordinal, item.case.contract_sha256)
            == (origin, ordinal, contract_sha256)
        ),
        None,
    )


# Pinned semantic provenance, never executable SampleCriteria or runtime policy.
_CRITERION_PROVENANCE: Final[str] = (
    '{"gather_separate_pending_audit":{"contract_sha256":"ab7d1021376f14588c8272d8bd53a961c18f4fcc9b1063fa31fa98b52ac39c39","legacy_threshold_lookup_candidate_only":{"generated_true_variant_default_atol":1e-07,"generated_true_variant_default_rtol":1e-07},"metadata_variant":"run_only_f64_variant=True","ordinal":0,"origin":"plugin::gather","raw_inputs":"float64(4,3), int64(2)","raw_output":"float64(2,3)","required_audit":"Verify actual FP64 floating trace/results, exact I64 index preservation, true variant applicability, complete legacy input/output conversions, reference context float32 versus raw None, integer dispatch implications, and original comparator direction before proposing a criterion. Twelve-row FP32 proposal cannot activate here.","status":"PENDING_SEPARATE_FULL_TRACE_VARIANT_COMPARATOR_AUDIT_NO_CRITERION_ASSIGNED","testcase":"gather_trig_where_pipeline_f64_indices_i64","x64_off":"Preserve the typed FP64/I64 configuration conflict; never narrow inputs to make it run."},"legacy_comparator":{"arity":"Equal flattened output count required.","dtype":"No general exact dtype equality requirement in _run_allclose. ORT feed casts to declared input type. Output actual usually casts to reference dtype before comparison.","float_conversion":"bfloat16 on each side widens to float32. Cast actual to reference comparison dtype unless actual is complex while reference is noncomplex. Complex reference + real actual trailing-pair axis may unpack and cast actual to reference complex dtype.","float_dispatch":"Floating comparison if either dtype is floating/complex or dtype.name is bfloat16.","float_predicate":"np.allclose(reference_cmp, actual_cmp, rtol=rtol, atol=atol, equal_nan=True). For finite elements threshold scales abs(actual_cmp): abs(reference_cmp-actual_cmp) <= atol + rtol*abs(actual_cmp).","integer_boolean_predicate":"If neither side dispatches as floating, np.array_equal(reference, actual.astype(reference.dtype, copy=False)); actual dtype need not match and narrowing casts are permitted.","layout":"Selected 4D ORT NCHW outputs transpose back to NHWC before comparison; selected inputs transpose NHWC to NCHW for ORT.","nan_infinity_zero":"equal_nan=True; matching NaNs compare equal, same-sign same-location infinities compare equal. Signed zeros compare equal. At actual zero finite threshold is atol; at reference zero with nonzero actual legacy still includes rtol*abs(actual). No zero-specific exception or bitwise test.","shape":"Exact reference and actual shape after layout/complex unpacking; no broadcasting accepted by this wrapper."},"original_harness_policy":{"allclose_public_default_not_harness_default":"allclose API signature defaults rtol=1e-3, atol=1e-5, but tests pass _get_rtol_atol results explicitly.","reference_configuration":"allclose wraps _run_allclose in _temporary_x64(enable_double_precision) then jax.default_matmul_precision(\\"float32\\"). _temporary_x64 aliases _force_jax_x64. JAX eager fn executes inside both scopes; do not infer raw matmul_precision=None equivalence.","runtime_input_casts":"false variant narrows typed float64 to float32 and int64 to int32; true variant converts floating values to float64 and retains integer dtype. Shape-only numeric fallback uses float32/float64 by variant; generated numbers use testcase-specific seed including the variant.","target_configuration":"ORT CPUExecutionProvider; ORT_DISABLE_ALL; mem_pattern=False; intra/inter threads=1.","tolerance_precedence_false_variant":["rtol_f32/atol_f32 independently","generic rtol/atol independently","defaults rtol=1e-5, atol=1e-5"],"tolerance_precedence_true_variant":["rtol_f64/atol_f64 independently","generic rtol/atol independently","defaults rtol=1e-7, atol=1e-7"],"variant_semantics":"generate_test_params selects f64-only for run_only_f64_variant, enable_double_precision or testcase suffix _f64; otherwise generates base false variant and optional true _f64 variant. True variant widens all floating input recipes/values/declared result dtypes; false shape-only numerical branch defaults float32. _get_rtol_atol branches on the generated test flag, not observed JAX x64 alone."},"pinned_revision":"d1dd3162dfba92ebc014d9e3866e829855d9921c","pinned_source":{"jax2onnx/precision.py":{"file_sha256":"b9960f647b024a81bd315ad44eefa0782cd11e8a8da3343a0e913da89d957d7c","functions":{"SampleCriteria":"5801960e6a85e20e32d59f356c41264da77ca3a7c7c1fe6e37dba16c20d8cc4e","_compare":"5da0ec726864559f89f54b03402c8bbb37766f3ec19a4c5e136038de256647cf"}},"jax2onnx/user_interface.py":{"file_sha256":"d108c5eba5d3ebe16251bf87598ef3cbcaba00eb243f5ab9708dc539643164a2","functions":{"_build_ort_inputs":"99beea10ea3c92675b7c5fe6ae6301959b4319b15791167d9aaeaf8b7ea67463","_is_floating_dtype":"f614e828e2323181566382967229836ac8387c43c759cf8136a265b4be4da1dd","_run_allclose":"e2a8e1a95a4eda8cbb8391d8d82ba6f0a652772cc9ba74349e9bc12f8cff66e0","_to_jax_array":"d7155db5745fc17e12ce7f1b7ba72973e49d111901590e1953e361193a8f58d1","_to_jax_kwarg":"9315e2367299bbb782cc324554af8a557f38626f8060bcbab7ba2ef3b30b2280","_to_numpy_input":"01fdbe3c14362946584757be626e177331407127eaf13b439301bcadf12257c7","_to_numpy_output":"3eaaf7696e22455a4ea8d60c8cfb6c3d0aa813d516ae11b26a8257d41bdf5e27","_validation_inputs_to_arrays":"313fc63b751f03a3a656ab4c329c1fe0df0c52c61702861fe35bd4cb00638167","allclose":"e22889cfed6c53f7bfc8183b0a46994f584ecb857974a41a8d745b5d0439b3c4"}},"tests/t_generator.py":{"file_sha256":"60dabe3e454b390520c594e566b36e21fd5691f83ad7ba57b535cfb33a74a4c7","functions":{"_get_rtol_atol":"b732599d00bbb43cdc3ae9d13d0fecfa365192ef6957b504db5929b0938a2e47","_rand":"3c609b79d6efac6b31120c4cdffe5e2881bb084c3db3f4f119699f7a56bbcf2f","_runtime_input_for_current_precision":"a173d901f703fd0ce29d1537e50c68708b83ef4c6122a622f2cc5cdb2ce26c56","generate_test_params":"d1d344b404e4a87d835a1c51e652d78818e492d2b9c071cc302ac175e2cb5892","make_test_function":"4534d09958fda0939d2c33225c01d7e74a0d833b564fc42890a07b9a27dac8ac"}}},"source_comparator":{"bitwise":"Compare C-order bytes, distinguishing signed zeros and NaN payloads.","dtype_shape":"Exact dtype then exact shape before any numerical comparison; no casts.","exact_float":"np.array_equal(actual, reference, equal_nan=criteria.equal_nan); signed zeros equal.","integer_boolean":"Outside bitwise, dtype kinds b/i/u use np.array_equal exact values, independent of tolerant/exact mode.","raw_configuration":"Raw SourceScenario defaults default_matmul_precision=None; source_configuration explicitly enters jax.default_matmul_precision(None), not inherited float32. raw-x64-on changes only caller x64; strong FP32 input specs remain FP32. prepare_source_case computes original eager reference under this source scope.","raw_criteria":"_frozen_criteria refuses a contract unless both explicit generic rtol and atol exist; legacy variant/default tolerances are not adopted. All selected twelve rows lack all six tolerance keys.","tolerant_float":"np.allclose(actual, reference, rtol=criteria.rtol, atol=criteria.atol, equal_nan=criteria.equal_nan), scaling abs(reference). SampleCriteria.equal_nan defaults True.","zero":"At reference zero threshold is atol; default non-bitwise modes equate signed zero."}}'
)
_FP32_OUTPUT_SHAPES: Final = {
    ("plugin::dot_general", 0): [3, 2],
    ("plugin::dot_general", 1): [3, 2],
    ("plugin::dot_general", 2): [3, 3],
    ("plugin::dot_general", 3): [3, 3],
    ("plugin::dot_general", 6): [2, 2, 3, 5],
    ("plugin::jax.numpy.add", 0): [3],
    ("plugin::jax.numpy.add", 1): [3],
    ("plugin::jax.numpy.add", 2): [2, 3],
    ("plugin::jax.numpy.add", 3): [3, 4],
    ("plugin::jax.numpy.matmul", 2): [3, 5],
    ("plugin::jax.numpy.matmul", 4): [2, 3, 5],
    ("plugin::jax.numpy.matmul", 8): [3, 2, 5],
}


def criterion_proposal_ledger() -> dict[str, Any]:
    policy = json.loads(_CRITERION_PROVENANCE)
    proposals = []
    for case in CORPUS_REPLAY_CASES:
        if case.origin == "plugin::gather":
            continue
        proposals.append(
            {
                **asdict(case),
                "status": "PROPOSED_NOT_ADOPTED_NO_RUNTIME",
                "historical_criteria": None,
                "raw_scenario_criteria": None,
                "scenarios": ["raw-x64-off", "raw-x64-on"],
                "legacy_variant": "false double-precision variant; raw x64-on remains strong FP32",
                "typed_source_eligibility": "Before any future adoption require all relevant floating inputs/equation results/output strong FP32, original shapes, no casts and reviewed parameters; this ledger performs no source evaluation",
                "expected_output_shape": _FP32_OUTPUT_SHAPES[
                    (case.origin, case.ordinal)
                ],
                "original_thresholds": {"rtol": 1e-5, "atol": 1e-5, "equal_nan": True},
                "proposed_predicate": "exact dtype/shape with no casts AND np.allclose(reference, actual, rtol=1e-5, atol=1e-5, equal_nan=True)",
                "public_SOURCE_direction": "np.allclose(actual, reference, ...), unchanged; a conjunction would require separately reviewed adoption",
                "configuration_difference": "Legacy reference wrapper forces matmul_precision=float32; raw replay remains None. Recovering thresholds does not establish reference-context equivalence.",
                "review": {
                    "proposed_date": "2026-10-18",
                    "approval": None,
                    "approved_date": None,
                },
            }
        )
    return {
        "version": "corpus-criteria-proposal-v1",
        "scope": "Proposal evidence only, no criterion assignment or numerical execution",
        "policy": policy,
        "rows": proposals,
        "gather": policy["gather_separate_pending_audit"],
    }


NORMALIZATION_ORACLE_VERSION: Final = "shared-b-original-layer-norm-options-v2"
NORMALIZATION_CASES: Final[tuple[CorpusCase, ...]] = (
    CorpusCase(
        "plugin::eqx.nn.layer_norm",
        0,
        "layer_norm",
        "28ff0cee91358c0be1ae4ea4cf2386e42620abc4d2ca545c69644d0ea5f10243",
    ),
    CorpusCase(
        "plugin::eqx.nn.layer_norm",
        1,
        "layer_norm_multiaxis",
        "ce9d57750c2d9a7be56dac839b7d5d7b21c6c14766d0f20efe732a23612d3135",
    ),
    CorpusCase(
        "plugin::eqx.nn.layer_norm",
        2,
        "batched_layer_norm",
        "d3aeb83e25b9af77ec322414da31f61584914f3f36548f504cf85af53f54e769",
    ),
    CorpusCase(
        "plugin::eqx.nn.layer_norm",
        3,
        "layer_norm_no_bias_no_scale",
        "28ff0cee91358c0be1ae4ea4cf2386e42620abc4d2ca545c69644d0ea5f10243",
    ),
    CorpusCase(
        "plugin::linen.layer_norm",
        0,
        "layer_norm",
        "32cc21c8909a3dcaf93e212dd79de063e577df743ac2c2ab0409e2965d93ed0c",
    ),
    CorpusCase(
        "plugin::linen.layer_norm",
        1,
        "layer_norm_no_bias_no_scale",
        "b1c8735f9f737e4e69136b0857a2a4431e01f3cea8617b87053c35b67069796b",
    ),
    CorpusCase(
        "plugin::linen.layer_norm",
        2,
        "layer_norm_multiaxis",
        "376a8783f49a68fbb7cc0eba3ec53303bceb962cff0a1f79b245f3d8ee70539f",
    ),
    CorpusCase(
        "plugin::linen.layer_norm",
        3,
        "layer_norm_default_epsilon",
        "ff13993a7b3a5172ae874ab70056b6c46839631a9b2dbee0448bfca8f43a5f8b",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        0,
        "layer_norm",
        "87e43b4dbd8bbb031ff0e968e0735251980b7f73e4b88bf4ec61ae22512e6e5e",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        1,
        "layer_norm_no_bias_no_scale",
        "d50091c7c4db5714c72986d437af0e6ad1802ba33bc9f70177f74d7fbff1d7ec",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        2,
        "layer_norm_bias_no_scale",
        "bf026adcbbc5742d5c768ec44c0fa3e1cd0140e3136802599c38ddf007eb2a88",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        3,
        "layer_norm_no_bias_scale",
        "1e7329dac4c333452e0e94d7a959bec40a5a3f04e1cbe512fa3d2df17d4265b0",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        4,
        "layer_norm_bias_scale",
        "af0fc4f4485d1895af78b50057e55f54274b08867acd6c94d3fcd33e2af9cfd9",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        5,
        "layer_norm_multiaxis",
        "c43ed0f5e6be4ac911313597af571d7676564f57d1206f759432e85bd4a27333",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        7,
        "layer_norm_symbolic_batch_seq10_feat3",
        "15fd8a32e70652890df09b2f530d76b764c1a0525b8d71bc56fd4e0c87e83595",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        8,
        "layer_norm_symbolic_batch_seq10_feat3_2",
        "8fc138d204be5a200657ade547ea552d3a0843caf406140a162859b5e6561681",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        9,
        "layer_norm_negative_axis_no_div",
        "5803ae86668523245d20b3f70299cd4f8e6fb82ee112e2bf85911f6cb2f8647e",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        12,
        "layer_norm_slow_variance",
        "083a3b1f34988b909d14d2267e496648a18714f202f7125a93ed3c226c2d76eb",
    ),
    CorpusCase(
        "plugin::eqx.nn.layer_norm",
        4,
        "layer_norm_decomposed",
        "a07527a499f1abf95d9f1b56b92027a104114ce1f39d385ceba9a0248e483f63",
    ),
    CorpusCase(
        "plugin::eqx.nn.layer_norm",
        5,
        "layer_norm_multiaxis_decomposed",
        "b14f78c4bb2d86952c05640b57fc81183312a93bdbd163429c7076021db73490",
    ),
    CorpusCase(
        "plugin::eqx.nn.layer_norm",
        6,
        "batched_layer_norm_decomposed",
        "0c5954fef8e22b60e77f92e6ded9f109b70ccbe7653a91e5909a0d5eb13603e1",
    ),
    CorpusCase(
        "plugin::eqx.nn.layer_norm",
        7,
        "layer_norm_opset16",
        "30dc566db5aad41865ea1b7bc8150fc87eeb29d8c59c65f62992eae6db8d48fb",
    ),
    CorpusCase(
        "plugin::linen.layer_norm",
        4,
        "layer_norm_decomposed",
        "b33fe40a63fe66a7997d9ec6088908fa58a531d3a6857cc8eb65dd4d3190b3f5",
    ),
    CorpusCase(
        "plugin::linen.layer_norm",
        5,
        "layer_norm_multiaxis_decomposed",
        "0b66d4068a2ce766b392387bcdc51ea684757682182620c0816c1c50db09926f",
    ),
    CorpusCase(
        "plugin::linen.layer_norm",
        6,
        "layer_norm_opset16",
        "0894c287c2e9b96a3ecd20343ef698385245dc9e72f3bcf7cfd517d2d9055127",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        6,
        "layer_norm_symbolic_batch",
        "f1134007ba577dcd5eb18f6d36cb7b8cac45f7492e89ebac09bc9c14b8545c11",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        10,
        "layer_norm_decomposed",
        "4da28f77eea76c24629b71bc9acf38c5ad1aec7cc2280c6f21017d25346eb573",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        11,
        "layer_norm_slow_variance_decomposed",
        "f547213d45d32caa75dcfa0cbe55c27b84bce9bd9ae6eea66efde23ae0fdb2fc",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        13,
        "layer_norm_multiaxis_decomposed",
        "51a7863a574225600a8c0f435accfae27ed08870e7dbddde1ebd0a196b24eeaf",
    ),
    CorpusCase(
        "plugin::nnx.layer_norm",
        14,
        "layer_norm_opset16",
        "64b58523eb0e0fb14e5b095a7b40f91cf91ab69286c51326c29640b39d62c1c1",
    ),
)

_NORMALIZATION_MODULES: Final = {
    "plugin::nnx.layer_norm": (
        "jax2onnx/plugins/flax/nnx/layer_norm.py",
        "6219dd00d62677cac1c99739121ec6efb1f11167cc0101135b2a2e7a25a6bd09",
    ),
    "plugin::linen.layer_norm": (
        "jax2onnx/plugins/flax/linen/layer_norm.py",
        "6611cc7a7dbbf77669e6ffe0fbd964bc4cd378e106c8cc894440e5dd7993b59d",
    ),
    "plugin::eqx.nn.layer_norm": (
        "jax2onnx/plugins/equinox/eqx/nn/layer_norm.py",
        "2e6481dc8951d838355492a5c714f67d906be75ea419ce6844ff5027af1905cd",
    ),
}


# Original ordinal and exact registration source segment remain separate from
# opaque callable contracts: Eqx rows 0/3 deliberately share a contract hash.
_NORMALIZATION_REGISTRATIONS: Final = {
    ("plugin::eqx.nn.layer_norm", 0): (
        "prefer_native",
        "96fe6088f8f10dd69fcf69f8c2d09f18e7ca94e6f634c548c7dbc0de90b3a739",
    ),
    ("plugin::eqx.nn.layer_norm", 1): (
        "prefer_native",
        "104dcf05ec304ffaffa8a70101d7733eda0996eaac20a4fa0e7e82da4dd78876",
    ),
    ("plugin::eqx.nn.layer_norm", 2): (
        "prefer_native",
        "a45327c2e574bb95d0fd42a0069281cefd4ba5335f2ee42f608735b796927055",
    ),
    ("plugin::eqx.nn.layer_norm", 3): (
        "prefer_native",
        "f66bb25c604dbf5adb1ee872790574a8f9d65fc79be94ab259c8d59f02e8d0ce",
    ),
    ("plugin::eqx.nn.layer_norm", 4): (
        "auto",
        "b56dd5627742da81ea67b092915c74c8bed4097a110441eb2f0eb5817581b49f",
    ),
    ("plugin::eqx.nn.layer_norm", 5): (
        "auto",
        "454307187d5811af0299b6bd0d12ae845f3ec01b53e4695c6c1a9ac7bfc24e89",
    ),
    ("plugin::eqx.nn.layer_norm", 6): (
        "auto",
        "f98399e3db626768e407c77eb8ad2c73966b8f7e96e5d07d7c638b062252b68e",
    ),
    ("plugin::eqx.nn.layer_norm", 7): (
        "auto",
        "864a3c7b45af4131ea30f49afe81cde857995ff97b34c67c43bdeef7f0546d42",
    ),
    ("plugin::linen.layer_norm", 0): (
        "prefer_native",
        "71b6467a99af2a6f58e2b26b62e29dac42bbb550345aa7b3f72fa64e9e4e1eb8",
    ),
    ("plugin::linen.layer_norm", 1): (
        "prefer_native",
        "5b5089415168b6603e08d2075f5188b0c128a0aab6c2217ded3fa102a7f0ae02",
    ),
    ("plugin::linen.layer_norm", 2): (
        "prefer_native",
        "6c4decdd666ecf0d8dea63730f7a4723f9f0c22c5d21e62b0508db7034f09ab8",
    ),
    ("plugin::linen.layer_norm", 3): (
        "prefer_native",
        "db52bfd292e8cb4ff2c6240ee447ee87f8bca04cd02ecfc95f57539ce569ec55",
    ),
    ("plugin::linen.layer_norm", 4): (
        "auto",
        "3db21d33e5ec2f326b967b18f73754b0cbf5523c644cdfaee8d69b0caf681c1e",
    ),
    ("plugin::linen.layer_norm", 5): (
        "auto",
        "8df91168eeccacdf3e1540414d1a27b796b44fa73d1123aec5dace89087764fe",
    ),
    ("plugin::linen.layer_norm", 6): (
        "auto",
        "3ad6b8677696e7cd025d9b9680e5a00ad834566e3d3391ec22fe325cdd0a4c9c",
    ),
    ("plugin::nnx.layer_norm", 0): (
        "prefer_native",
        "254030c4ac34a087b4fbeade847306b23b5750b7115d44599662899ca0f2c93f",
    ),
    ("plugin::nnx.layer_norm", 1): (
        "prefer_native",
        "d5c611a729156ae8707b710f995b01ca4a8baed39aba583b6dce783e21e27e81",
    ),
    ("plugin::nnx.layer_norm", 2): (
        "prefer_native",
        "96c6e4eb08fb31f8be3e90ceba9b6f1978778bd004fdd06f1233c1062b29d444",
    ),
    ("plugin::nnx.layer_norm", 3): (
        "prefer_native",
        "4504abf253fc1d80fdbdbb1f00ecd98e0e20f548bffb1c230db64e560ae47305",
    ),
    ("plugin::nnx.layer_norm", 4): (
        "prefer_native",
        "22268adb8e430c0a05dc77ad0136c31a6dd8045be79e4a626d188f52a4613c8f",
    ),
    ("plugin::nnx.layer_norm", 5): (
        "prefer_native",
        "860b27aa25b88f2b7da429dd6380be19365c4de8be1b1654c3505329337ad59e",
    ),
    ("plugin::nnx.layer_norm", 6): (
        "auto",
        "fe7ae2f0c649e7d99ff80c2836a47675a0a71d559a1aac5ede682359f22b2cd6",
    ),
    ("plugin::nnx.layer_norm", 7): (
        "prefer_native",
        "2af0ee8d7e739bc1428c3c6023ae41843b94d32b975f4f3f6af3afd7e32a7ad2",
    ),
    ("plugin::nnx.layer_norm", 8): (
        "prefer_native",
        "327795d168c4b85d6ece4f4e9ef1de2ec195e4c6c2727fd88066531204757bd7",
    ),
    ("plugin::nnx.layer_norm", 9): (
        "prefer_native",
        "2a643d11b23c6ea11924d26d39621dc9fc64446ce877c748fffeb902d7269e40",
    ),
    ("plugin::nnx.layer_norm", 10): (
        "auto",
        "394b02592af084de76d4b70ff6dda49c01e360645c1bbee5da9f923bd889aace",
    ),
    ("plugin::nnx.layer_norm", 11): (
        "auto",
        "3f27280807aed5fabcd68b666f97b6c8b608028140b1449792e5287fb0c886c7",
    ),
    ("plugin::nnx.layer_norm", 12): (
        "prefer_native",
        "b152ddb350f81f72dddd2b59737fe723fccd73771489f5f4ad95af50069aad0b",
    ),
    ("plugin::nnx.layer_norm", 13): (
        "auto",
        "b82bb9d7b5ded195d42a5e7f9798bd634f4312f2d6b8285b668c5773d9e62563",
    ),
    ("plugin::nnx.layer_norm", 14): (
        "auto",
        "e0c44bacb403080e55d5f7b9fc14e6fe71099dcf67cd68140bde4bb640f7dfe1",
    ),
}


def _normalization_registration_source(origin: str, ordinal: int) -> str:
    path, _ = _NORMALIZATION_MODULES[origin]
    source = (REPOSITORY_ROOT / path).read_text()
    rows = [
        keyword.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "testcases" and isinstance(keyword.value, ast.List)
    ]
    if len(rows) != 1 or not 0 <= ordinal < len(rows[0].elts):
        raise ValueError("Original normalization registration ordinal changed")
    segment = ast.get_source_segment(source, rows[0].elts[ordinal])
    if segment is None:
        raise ValueError("Original normalization registration source unavailable")
    return hashlib.sha256(segment.encode()).hexdigest()


class _NormalizationExpressions:
    """Finite normalization DAG comparison, preserving every typed arithmetic edge.

    Shape construction and the audited dense selection are representation steps;
    arithmetic is never reassociated, fused or compared through sampled values.
    Unknown operations/attributes remain explicit counterpart failures.
    """

    @staticmethod
    def constant(value: Any, dtype: Any | None = None) -> tuple[Any, ...]:
        array = np.asarray(value)
        if dtype is not None:
            declared = np.dtype(dtype)
            if isinstance(value, (np.ndarray, np.generic)) and array.dtype != declared:
                raise ValueError(
                    "Native normalization constant payload/aval dtype differs"
                )
            array = np.asarray(value, dtype=declared)
        return (
            "constant",
            array.dtype.name,
            tuple(array.shape),
            array.tobytes(order="C").hex(),
        )

    @staticmethod
    def node(
        operation: str,
        dtype: str,
        shape: tuple[Any, ...],
        inputs: tuple[Any, ...],
        attributes: tuple[Any, ...] = (),
    ) -> tuple[Any, ...]:
        if operation in {
            "add",
            "sub",
            "mul",
            "div",
            "neg",
            "sqrt",
            "max",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "and",
            "or",
            "xor",
            "not",
        }:
            # ONNX implicit broadcasting and native explicit broadcasting carry
            # the same edge. No arithmetic is folded or reassociated here.
            inputs = tuple(
                _NormalizationExpressions.broadcast(value, shape) for value in inputs
            )
        return (operation, dtype, shape, inputs, attributes)

    @classmethod
    def array(cls, value: Any) -> Any:
        if value[0] != "constant":
            raise ValueError("Expected an actual normalization constant payload")
        return np.frombuffer(bytes.fromhex(value[3]), dtype=value[1]).reshape(value[2])

    @classmethod
    def cast(cls, value: Any, dtype: str, shape: tuple[Any, ...]) -> Any:
        if value[2] != shape:
            raise ValueError("Normalization Cast changes shape")
        if value[1] == dtype:
            return value
        if value[0] == "constant":
            # Existing broadcast lowering can materialize an explicit literal
            # Cast. Preserve its exact destination payload, not arithmetic folds.
            return cls.constant(np.asarray(cls.array(value), dtype=dtype))
        return cls.node("cast", dtype, shape, (value,))

    @classmethod
    def reshape(cls, value: Any, shape: tuple[Any, ...]) -> Any:
        if value[2] == shape:
            return value
        if value[0] == "reshape":
            value = value[3][0]
            if value[2] == shape:
                return value
        static = all(type(dim) is int and dim > 0 for dim in (*value[2], *shape))
        if static:
            # Match ordinary native C-order static reshape, including affine
            # parameter unpacking. Use Python integers rather than a fixed-width
            # NumPy reduction so a malformed large count cannot wrap around.
            from math import prod

            if prod(value[2]) != prod(shape):
                raise ValueError("Normalization static reshape element counts differ")
        elif tuple(dim for dim in value[2] if dim != 1) != tuple(
            dim for dim in shape if dim != 1
        ):
            raise ValueError("Normalization counterpart forbids shared-B regrouping")
        if value[0] == "constant":
            return cls.constant(cls.array(value).reshape(shape))
        return cls.node("reshape", value[1], shape, (value,))

    @classmethod
    def broadcast(
        cls,
        value: Any,
        shape: tuple[Any, ...],
        dimensions: tuple[int, ...] | None = None,
    ) -> Any:
        dimensions = (
            tuple(range(len(shape) - len(value[2]), len(shape)))
            if dimensions is None
            else dimensions
        )
        if (
            len(dimensions) != len(value[2])
            or tuple(sorted(set(dimensions))) != dimensions
        ):
            raise ValueError("Normalization broadcast mapping differs")
        inserted: list[Any] = [1] * len(shape)
        for source, axis in zip(value[2], dimensions, strict=True):
            if axis < 0 or axis >= len(shape) or source not in (1, shape[axis]):
                raise ValueError("Normalization broadcast changes an existing extent")
            inserted[axis] = source
        reshaped = cls.reshape(value, tuple(inserted))
        if reshaped[0] == "constant" and all(isinstance(dim, int) for dim in shape):
            return cls.constant(np.broadcast_to(cls.array(reshaped), shape))
        return (
            reshaped
            if tuple(inserted) == shape
            else cls.node("broadcast", value[1], shape, (reshaped,))
        )

    @classmethod
    def native(
        cls, closed: Any, arguments: tuple[Any, ...] | None = None
    ) -> tuple[Any, ...]:
        from jax2onnx._compat.jax import Literal
        from jax2onnx.converter.source_representation import normalized_shape_dimensions

        body = getattr(closed, "jaxpr", closed)

        def properties(var: Any) -> tuple[str, tuple[Any, ...]]:
            return np.dtype(var.aval.dtype).name, normalized_shape_dimensions(
                var.aval.shape
            )

        env = {}
        for var, value in zip(
            body.constvars, getattr(closed, "consts", ()), strict=True
        ):
            env[id(var)] = cls.constant(value, var.aval.dtype)
        supplied = (
            arguments
            if arguments is not None
            else tuple(
                ("input", *properties(var), index)
                for index, var in enumerate(body.invars)
            )
        )
        for var, value in zip(body.invars, supplied, strict=True):
            if (value[1], value[2]) != properties(var):
                raise ValueError("Native nested normalization port differs")
            env[id(var)] = value

        def use(var: Any) -> Any:
            return (
                cls.constant(var.val, var.aval.dtype)
                if isinstance(var, Literal)
                else env[id(var)]
            )

        for equation in body.eqns:
            name = equation.primitive.name
            args = tuple(use(var) for var in equation.invars)
            if name == "jit":
                child = equation.params.get("jaxpr", equation.params.get("call_jaxpr"))
                if child is None:
                    raise ValueError("Native normalization JIT body unavailable")
                child_body = getattr(child, "jaxpr", child)
                for caller, formal in zip(
                    (*equation.invars, *equation.outvars),
                    (*child_body.invars, *child_body.outvars),
                    strict=True,
                ):
                    if properties(caller) != properties(formal) or bool(
                        caller.aval.weak_type
                    ) != bool(formal.aval.weak_type):
                        raise ValueError(
                            "Native normalization JIT formal/result dtype/shape/weakness differs"
                        )
                outputs = cls.native(child, args)
                for var, value in zip(equation.outvars, outputs, strict=True):
                    if properties(var) != (value[1], value[2]):
                        raise ValueError("Native normalization JIT result differs")
                    env[id(var)] = value
                continue
            if len(equation.outvars) != 1:
                raise ValueError(
                    "Normalization counterpart requires single-result native primitives"
                )
            dtype, shape = properties(equation.outvars[0])
            attrs: tuple[Any, ...] = ()
            if name == "convert_element_type":
                result = cls.cast(args[0], dtype, shape)
            elif name == "reshape":
                if equation.params.get("dimensions") is not None:
                    raise ValueError("Normalization reshape permutation differs")
                result = cls.reshape(args[0], shape)
            elif name == "broadcast_in_dim":
                result = cls.broadcast(
                    args[0], shape, tuple(equation.params["broadcast_dimensions"])
                )
            elif name == "square":
                result = cls.node("mul", dtype, shape, (args[0], args[0]))
            elif name == "rsqrt":
                result = cls.node(
                    "div",
                    dtype,
                    shape,
                    (
                        cls.constant(np.asarray(1, dtype=dtype)),
                        cls.node("sqrt", dtype, shape, args),
                    ),
                )
            elif name == "select_n":
                result = cls.node(
                    "select", dtype, shape, (cls.broadcast(args[0], shape), *args[1:])
                )
            elif name in {
                "add",
                "sub",
                "mul",
                "div",
                "neg",
                "sqrt",
                "max",
                "eq",
                "ne",
                "lt",
                "le",
                "gt",
                "ge",
                "and",
                "or",
                "xor",
                "not",
                "reduce_sum",
            }:
                if name == "reduce_sum":
                    attrs = tuple(equation.params["axes"])
                    result = (
                        args[0]
                        if not attrs
                        else cls.node(name, dtype, shape, args, attrs)
                    )
                else:
                    result = cls.node(name, dtype, shape, args)
            else:
                raise ValueError(
                    f"Native normalization operation {name} lacks a counterpart"
                )
            env[id(equation.outvars[0])] = result
        return tuple(use(var) for var in body.outvars)

    def __init__(self, model: onnx.ModelProto):
        if model.functions:
            raise ValueError(
                "Original LayerNorm family counterpart forbids unexplained Functions"
            )
        self.model = model
        self.graph = model.graph
        self.nodes = {}
        self.types = {}
        self.declared_types = {}
        self.shape_types = {}
        self.resolving_shape_types: set[str] = set()
        self.constants = {
            tensor.name: onnx.numpy_helper.to_array(tensor)
            for tensor in self.graph.initializer
        }
        if len(self.constants) != len(self.graph.initializer):
            raise ValueError("Duplicate normalization initializer names")
        self.visited: set[str] = set()
        self.memo = {}
        for value in (*self.graph.input, *self.graph.value_info, *self.graph.output):
            if value.type.WhichOneof("value") not in {None, "tensor_type"}:
                raise ValueError("Normalization requires tensor value metadata")
            tensor = value.type.tensor_type
            dtype = (
                np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)).name
                if tensor.elem_type
                else None
            )
            shape = (
                tuple(
                    dim.dim_param if dim.HasField("dim_param") else dim.dim_value
                    for dim in tensor.shape.dim
                )
                if tensor.HasField("shape")
                else None
            )
            prior = self.declared_types.get(value.name, (None, None))
            if any(
                old is not None and new is not None and old != new
                for old, new in zip(prior, (dtype, shape), strict=True)
            ):
                raise ValueError("Conflicting normalization value metadata")
            declared = tuple(
                old if new is None else new
                for old, new in zip(prior, (dtype, shape), strict=True)
            )
            self.declared_types[value.name] = declared
            if None not in declared:
                self.types[value.name] = declared
        for name, array in self.constants.items():
            self.bind_type(name, (array.dtype.name, tuple(array.shape)))
        for node in self.graph.node:
            if (
                node.domain
                or len(node.output) != 1
                or node.output[0] in self.nodes
                or node.output[0] in self.constants
                or node.output[0] in {value.name for value in self.graph.input}
            ):
                raise ValueError(
                    "Normalization counterpart requires standard-domain single-definition nodes"
                )
            self.nodes[node.output[0]] = node
            if node.op_type == "Constant":
                if node.input:
                    raise ValueError("Normalization Constant must have zero inputs")
                attrs = self.attributes(node, {"value"})
                if set(attrs) != {"value"}:
                    raise ValueError(
                        "Normalization Constant requires its exact tensor payload"
                    )
                array = onnx.numpy_helper.to_array(attrs["value"])
                self.bind_type(node.output[0], (array.dtype.name, tuple(array.shape)))
        for value in (*self.graph.input, *self.graph.output):
            if value.name not in self.types:
                raise ValueError(
                    "Normalization boundary requires actual dtype and shape"
                )
        for index, value in enumerate(self.graph.input):
            self.memo[value.name] = ("input", *self.types[value.name], index)

    def bind_type(self, name: str, actual: tuple[str, tuple[Any, ...]]) -> None:
        declared = self.declared_types.get(name, (None, None))
        if any(
            expected is not None and expected != observed
            for expected, observed in zip(declared, actual, strict=True)
        ) or (name in self.types and self.types[name] != actual):
            raise ValueError("Normalization payload/derived type and metadata differ")
        self.types[name] = actual

    def shape_type(self, name: str) -> tuple[str, tuple[Any, ...]]:
        """Derive missing metadata only through the consumed INT64 shape grammar."""
        if name in self.shape_types:
            return self.shape_types[name]
        if name in self.resolving_shape_types:
            raise ValueError("Cyclic normalization shape metadata")
        self.resolving_shape_types.add(name)
        try:
            if name in self.constants:
                array = self.constants[name]
                actual = (array.dtype.name, tuple(array.shape))
            else:
                node = self.nodes.get(name)
                if node is None:
                    raise ValueError("Missing normalization shape producer")
                if node.op_type == "Constant":
                    actual = self.types[name]
                elif node.op_type == "Shape":
                    attrs = self.attributes(node, {"start", "end"})
                    if len(node.input) != 1 or node.input[0] not in self.types:
                        raise ValueError("Shape requires actual typed data input")
                    shape = self.types[node.input[0]][1]
                    actual = (
                        "int64",
                        (
                            len(
                                shape[
                                    attrs.get("start", 0) : attrs.get("end", len(shape))
                                ]
                            ),
                        ),
                    )
                elif node.op_type == "Gather":
                    if (
                        len(node.input) != 2
                        or self.attributes(node, {"axis"}).get("axis", 0) != 0
                    ):
                        raise ValueError("Shape Gather arity/axis differs")
                    data = self.shape_type(node.input[0])
                    indices = self.shape_type(node.input[1])
                    if (
                        data[0] != "int64"
                        or len(data[1]) != 1
                        or indices[0] != "int64"
                        or len(indices[1]) > 1
                    ):
                        raise ValueError(
                            "Shape Gather requires INT64 vector/scalar indices"
                        )
                    actual = ("int64", indices[1])
                elif node.op_type in {"Unsqueeze", "Squeeze"}:
                    if len(node.input) != 2 or self.attributes(node, set()):
                        raise ValueError("Shape scalar/vector promotion arity differs")
                    operand = self.shape_type(node.input[0])
                    self.shape_type(node.input[1])
                    if self.vector(node.input[1]) != (0,) or self.types[
                        node.input[1]
                    ] != ("int64", (1,)):
                        raise ValueError("Shape promotion axes must be INT64[1] zero")
                    expected = () if node.op_type == "Unsqueeze" else (1,)
                    if operand != ("int64", expected):
                        raise ValueError(
                            "Shape promotion requires INT64 scalar/one-vector"
                        )
                    actual = ("int64", (1,) if node.op_type == "Unsqueeze" else ())
                elif node.op_type == "Concat":
                    if not node.input or self.attributes(node, {"axis"}) != {"axis": 0}:
                        raise ValueError("Shape Concat requires axis zero and operands")
                    operands = tuple(self.shape_type(operand) for operand in node.input)
                    if any(
                        dtype != "int64"
                        or len(shape) != 1
                        or type(shape[0]) is not int
                        or shape[0] < 0
                        for dtype, shape in operands
                    ):
                        raise ValueError("Shape Concat requires resolved INT64 vectors")
                    actual = ("int64", (sum(shape[0] for _, shape in operands),))
                else:
                    raise ValueError(
                        f"Unaudited normalization shape producer {node.op_type}"
                    )
            if actual[0] != "int64" or len(actual[1]) > 1:
                raise ValueError("Normalization shape edge must be INT64 scalar/vector")
            self.bind_type(name, actual)
            self.shape_types[name] = actual
            return actual
        finally:
            self.resolving_shape_types.remove(name)

    def attributes(self, node: Any, allowed: set[str]) -> dict[str, Any]:
        if len(node.attribute) != len({item.name for item in node.attribute}) or any(
            item.name not in allowed for item in node.attribute
        ):
            raise ValueError("Unreviewed normalization node attributes")
        return {
            item.name: onnx.helper.get_attribute_value(item) for item in node.attribute
        }

    def take(self, name: str, operation: str, arity: int) -> Any:
        node = self.nodes.get(name)
        if node is None or node.op_type != operation or len(node.input) != arity:
            raise ValueError(f"Disconnected normalization {operation} producer")
        self.visited.add(name)
        return node

    def vector(self, name: str) -> tuple[Any, ...]:
        self.visited.add(name)
        self.shape_type(name)
        if name in self.constants:
            array = self.constants[name]
            if array.dtype != np.dtype(np.int64) or array.ndim > 1:
                raise ValueError(
                    "Shape/axis constant must retain exact INT64 scalar/vector storage"
                )
            return tuple(array.reshape(-1).tolist())
        node = self.nodes[name]
        if node.op_type == "Constant":
            return tuple(self.array(self.expression(name)).reshape(-1).tolist())
        if node.op_type == "Shape":
            attrs = self.attributes(node, {"start", "end"})
            if len(node.input) != 1:
                raise ValueError("Shape input arity differs")
            self.expression(node.input[0])
            shape = self.types[node.input[0]][1]
            return shape[attrs.get("start", 0) : attrs.get("end", len(shape))]
        if node.op_type == "Gather":
            if self.attributes(node, {"axis"}).get("axis", 0) != 0:
                raise ValueError("Shape Gather axis differs")
            vector = self.vector(node.input[0])
            return tuple(vector[index] for index in self.vector(node.input[1]))
        if node.op_type == "Concat":
            if self.attributes(node, {"axis"}) != {"axis": 0}:
                raise ValueError("Shape Concat axis differs")
            return tuple(dim for operand in node.input for dim in self.vector(operand))
        if node.op_type in {"Unsqueeze", "Squeeze"}:
            if self.attributes(node, set()) or self.vector(node.input[1]) != (0,):
                raise ValueError("Shape scalar/vector promotion differs")
            return self.vector(node.input[0])
        raise ValueError(f"Unaudited normalization shape producer {node.op_type}")

    def axis_zero(self, name: str) -> None:
        if self.vector(name) != (0,) or self.types[name] != ("int64", (1,)):
            raise ValueError("Dense selection axes must be exact INT64[1] zero")

    def selection(self, name: str) -> Any:
        squeeze = self.take(name, "Squeeze", 2)
        self.attributes(squeeze, set())
        self.axis_zero(squeeze.input[1])
        gather = self.take(squeeze.input[0], "GatherElements", 2)
        if self.attributes(gather, {"axis"}) != {"axis": 0}:
            raise ValueError("Dense selection GatherElements axis differs")
        choices = self.take(gather.input[0], "Concat", 2)
        if self.attributes(choices, {"axis"}) != {"axis": 0}:
            raise ValueError("Dense selection Concat axis differs")
        arms = []
        for operand in choices.input:
            arm = self.take(operand, "Unsqueeze", 2)
            self.attributes(arm, set())
            self.axis_zero(arm.input[1])
            arms.append(arm.input[0])
        indices = self.take(gather.input[1], "Unsqueeze", 2)
        self.attributes(indices, set())
        self.axis_zero(indices.input[1])
        dtype, shape = self.types[name]
        cast_name = indices.input[0]
        expansion = self.nodes.get(cast_name)
        if expansion is not None and expansion.op_type == "Expand":
            expansion = self.take(cast_name, "Expand", 2)
            self.attributes(expansion, set())
            if self.vector(expansion.input[1]) != shape:
                raise ValueError("Dense selection index Expand uses another shape")
            cast_name = expansion.input[0]
        else:
            expansion = None
        cast = self.take(cast_name, "Cast", 1)
        target = self.attributes(cast, {"to"}).get("to")
        if target not in {onnx.TensorProto.INT32, onnx.TensorProto.INT64}:
            raise ValueError("Dense selection index Cast is not INT32/INT64")
        selector_dtype, selector_shape = self.types[cast.input[0]]
        index_dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(target)).name
        if selector_dtype != "bool" or self.types[cast.output[0]] != (
            index_dtype,
            selector_shape,
        ):
            raise ValueError(
                "Dense selection Cast must preserve actual BOOL selector extents"
            )
        if expansion is not None and self.types[expansion.output[0]] != (
            index_dtype,
            shape,
        ):
            raise ValueError("Dense selection expanded indices dtype/shape differs")
        if expansion is None and selector_shape != shape:
            raise ValueError("Dense selection indices lack required broadcast")
        for output in (choices.output[0],):
            if self.types[output] != (dtype, (2, *shape)):
                raise ValueError("Dense selection choices type/shape differs")
        for output in (
            gather.output[0],
            *(self.nodes[operand].output[0] for operand in choices.input),
        ):
            if self.types[output] != (dtype, (1, *shape)):
                raise ValueError("Dense selection data type/shape differs")
        if self.types[indices.output[0]] != (index_dtype, (1, *shape)):
            raise ValueError("Dense selection index type/shape differs")
        first = self.nodes.get(arms[0])
        if first is not None and first.op_type == "Max":
            if target != onnx.TensorProto.INT64 or expansion is not None:
                raise ValueError(
                    "Max signed-zero route must retain its INT64 unexpanded indices"
                )
            base = self.take(arms[0], "Max", 2)
            self.attributes(base, set())
            correction = self.take(arms[1], "Gather", 2)
            if self.attributes(correction, {"axis"}) != {"axis": 0}:
                raise ValueError("Max zero-table Gather axis differs")
            if self.types[base.output[0]] != (dtype, shape) or self.types[
                correction.output[0]
            ] != (dtype, shape):
                raise ValueError(
                    "Max correction does not retain exact source/result dtype/shape"
                )
            if self.types[correction.input[0]] != (dtype, (2,)) or self.expression(
                correction.input[0]
            ) != self.constant(np.asarray([0.0, -0.0], dtype=dtype)):
                raise ValueError(
                    "Max zero table must retain exact original typed words"
                )
            sign_cast = self.take(correction.input[1], "Cast", 1)
            if self.attributes(sign_cast, {"to"}) != {"to": onnx.TensorProto.INT64}:
                raise ValueError("Max zero-table index Cast must be exact INT64")
            sign = self.take(sign_cast.input[0], "And", 2)
            self.attributes(sign, set())
            if self.types[sign_cast.output[0]] != ("int64", shape) or self.types[
                sign.output[0]
            ] != ("bool", shape):
                raise ValueError("Max zero sign/index dtype/shape differs")
            for condition, operand in zip(sign.input, base.input, strict=True):
                less = self.take(condition, "Less", 2)
                self.attributes(less, set())
                reciprocal = self.take(less.input[0], "Reciprocal", 1)
                self.attributes(reciprocal, set())
                operand_shape = self.types[operand][1]
                if (
                    reciprocal.input[0] != operand
                    or self.types[operand][0] != dtype
                    or self.types[reciprocal.output[0]] != (dtype, operand_shape)
                    or self.types[less.output[0]] != ("bool", operand_shape)
                ):
                    raise ValueError(
                        "Max zero correction operands/types differ from consumed Max"
                    )
                if self.expression(less.input[1]) != self.constant(
                    np.asarray(0, dtype=dtype)
                ):
                    raise ValueError(
                        "Max sign comparison requires original positive zero"
                    )
            tie = self.take(cast.input[0], "And", 2)
            self.attributes(tie, set())
            if self.types[tie.output[0]] != ("bool", shape):
                raise ValueError("Max tie predicate dtype/shape differs")
            for condition, operand in zip(tie.input, base.input, strict=True):
                equal = self.take(condition, "Equal", 2)
                self.attributes(equal, set())
                if self.types[equal.output[0]] != ("bool", self.types[operand][1]):
                    raise ValueError("Max Equal predicate dtype/shape differs")
                if equal.input[0] != operand:
                    raise ValueError("Max zero predicate consumes another operand")
                if self.expression(equal.input[1]) != self.constant(
                    np.asarray(0, dtype=dtype)
                ):
                    raise ValueError(
                        "Max tie constant is not original positive zero in result dtype"
                    )
            return self.node(
                "max",
                dtype,
                shape,
                tuple(self.expression(operand) for operand in base.input),
            )
        if target != onnx.TensorProto.INT32:
            raise ValueError(
                "Original native select_n route must retain its INT32 selector Cast"
            )
        selector = self.broadcast(self.expression(cast.input[0]), shape)
        return self.node(
            "select",
            dtype,
            shape,
            (selector, *(self.expression(operand) for operand in arms)),
        )

    def expression(self, name: str) -> Any:
        self.visited.add(name)
        if name in self.memo:
            return self.memo[name]
        if name in self.constants:
            result = self.constant(self.constants[name])
        else:
            node = self.nodes[name]
            if name not in self.types:
                raise ValueError(
                    "Normalization arithmetic requires actual dtype and shape metadata"
                )
            dtype, shape = self.types[name]
            operation = node.op_type
            if operation == "Constant":
                attrs = self.attributes(node, {"value"})
                if set(attrs) != {"value"}:
                    raise ValueError(
                        "Normalization Constant requires its actual tensor payload"
                    )
                result = self.constant(onnx.numpy_helper.to_array(attrs["value"]))
            elif operation == "Identity":
                self.attributes(node, set())
                result = self.expression(node.input[0])
                if (result[1], result[2]) != (dtype, shape):
                    raise ValueError("Identity alias changes normalization type/shape")
            elif operation == "Cast":
                attrs = self.attributes(node, {"to"})
                if attrs.get("to") != onnx.helper.np_dtype_to_tensor_dtype(
                    np.dtype(dtype)
                ):
                    raise ValueError("Normalization Cast attribute and result differ")
                operand = self.expression(node.input[0])
                result = self.cast(operand, dtype, shape)
            elif operation in {"Reshape", "Expand"}:
                attrs = self.attributes(
                    node, {"allowzero"} if operation == "Reshape" else set()
                )
                if (
                    attrs.get("allowzero", 0) != 0
                    or self.vector(node.input[1]) != shape
                ):
                    raise ValueError(
                        "Normalization shape input differs from the consumed result"
                    )
                operand = self.expression(node.input[0])
                result = (
                    self.reshape(operand, shape)
                    if operation == "Reshape"
                    else self.broadcast(operand, shape)
                )
            elif operation == "Squeeze":
                result = self.selection(name)
            elif operation == "Where":
                self.attributes(node, set())
                selector, true_value, false_value = (
                    self.expression(operand) for operand in node.input
                )
                result = self.node(
                    "select",
                    dtype,
                    shape,
                    (self.broadcast(selector, shape), false_value, true_value),
                )
            elif operation == "ReduceSum":
                attrs = self.attributes(node, {"keepdims", "noop_with_empty_axes"})
                axes = self.vector(node.input[1])
                operand = self.expression(node.input[0])
                if attrs.get("keepdims", 1) != 0 or attrs.get(
                    "noop_with_empty_axes", 0
                ) not in {0, 1}:
                    raise ValueError("Normalization reduction attributes differ")
                result = (
                    operand
                    if not axes and attrs.get("noop_with_empty_axes") == 1
                    else self.node("reduce_sum", dtype, shape, (operand,), axes)
                )
            else:
                names = {
                    "Add": "add",
                    "Sub": "sub",
                    "Mul": "mul",
                    "Div": "div",
                    "Neg": "neg",
                    "Sqrt": "sqrt",
                    "Equal": "eq",
                    "Less": "lt",
                    "LessOrEqual": "le",
                    "Greater": "gt",
                    "GreaterOrEqual": "ge",
                    "And": "and",
                    "Or": "or",
                    "Xor": "xor",
                    "Not": "not",
                }
                if operation not in names:
                    raise ValueError(
                        f"Unexplained serialized normalization operation {operation}"
                    )
                self.attributes(node, set())
                result = self.node(
                    names[operation],
                    dtype,
                    shape,
                    tuple(self.expression(operand) for operand in node.input),
                )
            if (result[1], result[2]) != (dtype, shape):
                raise ValueError("Normalization logical expression type/shape differs")
        self.memo[name] = result
        return result

    def outputs(self) -> tuple[Any, ...]:
        result = tuple(self.expression(value.name) for value in self.graph.output)
        disconnected = (
            set(self.nodes)
            | set(self.constants)
            | {value.name for value in self.graph.input}
        ) - self.visited
        if disconnected:
            details = [
                {
                    "name": name,
                    "op": (
                        self.nodes[name].op_type
                        if name in self.nodes
                        else "initializer"
                    ),
                    "inputs": (
                        list(self.nodes[name].input) if name in self.nodes else []
                    ),
                    "type": self.types.get(name),
                }
                for name in sorted(disconnected)
            ]
            raise ValueError(
                "Normalization graph contains disconnected producers/constants: "
                + repr(details)
            )
        return result


@dataclass(frozen=True)
class NormalizationCounterpart:
    case: CorpusCase

    @property
    def identifier(self) -> str:
        return f"{NORMALIZATION_ORACLE_VERSION}:{self.case.origin}:{self.case.ordinal}"

    @property
    def sha256(self) -> str:
        return fingerprint(
            {
                "identifier": self.identifier,
                "case": asdict(self.case),
                "original_modules": _NORMALIZATION_MODULES,
                "registration": _NORMALIZATION_REGISTRATIONS[
                    (self.case.origin, self.case.ordinal)
                ],
                "oracle_file_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "sha256": self.sha256,
            "case": asdict(self.case),
        }

    def check_metadata(self, metadata: dict[str, Any]) -> None:
        from tests.extra_tests.converter.source_precision_support import raw_source_case

        if type(self) is not NormalizationCounterpart or not any(
            self is item for item in NORMALIZATION_COUNTERPARTS
        ):
            raise ValueError("Unreviewed normalization counterpart mapping")
        raw = raw_source_case(self.case.origin, self.case.ordinal)
        if (
            metadata.get("testcase") != self.case.testcase
            or metadata.get("callable") is not raw.get("callable")
            or metadata.get("post_check_onnx_graph")
            is not raw.get("post_check_onnx_graph")
        ):
            raise ValueError(
                "Normalization original registered callable/callback identity changed"
            )
        unresolved: list[str] = []
        contract = canonical_contract(
            {
                key: value
                for key, value in metadata.items()
                if key not in {"testcase", "description", "context", "component"}
            },
            unresolved=unresolved,
        )
        requested_mode, registration_sha256 = _NORMALIZATION_REGISTRATIONS[
            (self.case.origin, self.case.ordinal)
        ]
        if (
            fingerprint(contract) != self.case.contract_sha256
            or metadata.get("normalization_mode", "auto") != requested_mode
            or raw.get("normalization_mode", "auto") != requested_mode
            or _normalization_registration_source(self.case.origin, self.case.ordinal)
            != registration_sha256
        ):
            raise ValueError(
                "Normalization complete raw contract/mode/registration changed"
            )
        if unresolved and self.case.origin != "plugin::eqx.nn.layer_norm":
            raise ValueError("Unexpected opaque normalization recipe")
        for path, wanted in _NORMALIZATION_MODULES.values():
            if (
                hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest()
                != wanted
            ):
                raise ValueError(
                    "Original normalization registration/source module changed"
                )
        if self.case.origin == "plugin::eqx.nn.layer_norm":
            import equinox as eqx

            target = raw["callable"]
            # Follow only the original fixed vmap fun capture, never invoke it.
            for _ in range(4):
                if type(target) is eqx.nn.LayerNorm:
                    break
                if (
                    not inspect.isfunction(target)
                    or "fun" not in target.__code__.co_freevars
                ):
                    raise ValueError(
                        "Original opaque Eqx vmap/LayerNorm binding changed"
                    )
                captures = dict(
                    zip(
                        target.__code__.co_freevars,
                        target.__closure__ or (),
                        strict=True,
                    )
                )
                target = captures["fun"].cell_contents
            expected_shape = (20, 32) if self.case.ordinal in (1, 5) else (32,)
            affine = self.case.ordinal != 3
            if (
                type(target) is not eqx.nn.LayerNorm
                or target.shape != expected_shape
                or target.eps != 1e-5
                or target.use_weight is not affine
                or target.use_bias is not affine
            ):
                raise ValueError("Original Eqx LayerNorm configuration changed")
            for value, expected in ((target.weight, 1), (target.bias, 0)):
                if not affine:
                    if value is not None:
                        raise ValueError(
                            "Original Eqx no-affine LayerNorm parameter must be absent"
                        )
                    continue
                array = np.asarray(value)
                if (
                    array.shape != expected_shape
                    or array.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
                    or array.tobytes(order="C")
                    != np.full(expected_shape, expected, dtype=array.dtype).tobytes(
                        order="C"
                    )
                ):
                    raise ValueError(
                        "Original Eqx LayerNorm affine parameter bytes changed"
                    )

    def check_source(self, prepared: PreparedSourceCase) -> None:
        from jax2onnx.converter.source_representation import normalized_shape_dimensions
        from tests.extra_tests.converter.source_precision_diagnostics import (
            typed_trace_manifest,
        )
        from tests.extra_tests.converter.source_precision_support import (
            PRESERVE_FACTORY_DTYPE,
        )

        self.check_metadata(dict(prepared.metadata))
        provenance = prepared.scenario.provenance
        if (
            provenance is None
            or (provenance.origin, provenance.ordinal)
            != (self.case.origin, self.case.ordinal)
            or prepared.scenario.model_dtype is not PRESERVE_FACTORY_DTYPE
        ):
            raise ValueError(
                "Normalization original raw provenance/factory policy changed"
            )
        if prepared.identity != (
            prepared.metadata["context"],
            prepared.metadata["component"],
            self.case.testcase,
        ):
            raise ValueError("Normalization original source display identity changed")
        # Eqx's raw recipe stays opaque in the inventory. Exact registered object,
        # configuration, affine bytes and retained native trace bind this finite
        # counterpart; this does not resolve other opaque metadata recipes.
        observed = json.loads(json.dumps(typed_trace_manifest(prepared.original_jaxpr)))
        if observed != json.loads(prepared.original_trace_json):
            raise ValueError("Retained original normalization trace/constants changed")
        if (
            len(prepared.traced_inputs) != 1
            or len(prepared.traced_outputs) != 1
            or prepared.traced_inputs[0].shape != prepared.traced_outputs[0].shape
        ):
            raise ValueError("Original LayerNorm family input/output contract changed")
        body = prepared.original_jaxpr.jaxpr
        for properties, var in zip(
            (*prepared.traced_inputs, *prepared.traced_outputs),
            (*body.invars, *body.outvars),
            strict=True,
        ):
            if (properties.dtype, properties.shape, properties.weak_type) != (
                np.dtype(var.aval.dtype).name,
                normalized_shape_dimensions(var.aval.shape),
                bool(var.aval.weak_type),
            ):
                raise ValueError(
                    "Normalization retained native port dtype/shape/weakness changed"
                )
        if prepared.settings.default_matmul_precision is not None:
            raise ValueError("Normalization raw matmul context changed")

    def _check_model(
        self, prepared: PreparedSourceCase, model: onnx.ModelProto
    ) -> None:
        from jax2onnx.precision import get_precision_report

        report = get_precision_report(model)
        opset = int(prepared.metadata.get("opset_version", 23))
        imports = {item.domain: item.version for item in model.opset_import}
        if (
            report is None
            or report.profile not in {"source-static-v1", "source-diagnostic-v1"}
            or report.opset != opset
            or imports.get("") != opset
            or set(imports) != {""}
            or len(imports) != len(model.opset_import)
            or opset not in (16, 23)
        ):
            raise ValueError("Normalization raw/report/actual graph opsets differ")
        if report.profile == "source-diagnostic-v1" and any(
            check.category == "lowering" and check.status is not CheckStatus.UNVERIFIED
            for check in report.checks
        ):
            raise ValueError(
                "Diagnostic normalization structure cannot certify lowering"
            )
        requested_mode = prepared.metadata.get("normalization_mode", "auto")
        decisions = tuple(
            item for item in report.adaptations if item.startswith("normalization:")
        )
        expected_decisions = (
            (
                f"normalization: requested={requested_mode}; effective=typed-native-jax-dag; "
                "reason=native ONNX normalization has no reviewed source-equivalence proof; "
                "original constants, casts, accumulation and rounding retained",
            )
            if requested_mode == "prefer_native"
            else ()
        )
        if decisions != expected_decisions:
            raise ValueError(
                "Normalization report loses the requested/effective representation decision"
            )
        if (
            model.graph.sparse_initializer
            or len(model.graph.input) != 1
            or len(model.graph.output) != 1
        ):
            raise ValueError(
                "Normalization actual boundary has extra ports or sparse constants"
            )
        if (
            report.caller_x64_enabled != prepared.settings.x64_enabled
            or report.caller_default_matmul_precision
            != prepared.settings.default_matmul_precision
            or report.caller_numpy_dtype_promotion
            != prepared.settings.numpy_dtype_promotion
            or report.caller_numpy_rank_promotion
            != prepared.settings.numpy_rank_promotion
        ):
            raise ValueError("Normalization report loses original caller settings")
        for source, value in zip(
            (*prepared.traced_inputs, *prepared.traced_outputs),
            (*model.graph.input, *model.graph.output),
            strict=True,
        ):
            tensor = value.type.tensor_type
            dtype = np.dtype(
                onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)
            ).name
            shape = tuple(
                dim.dim_param if dim.HasField("dim_param") else dim.dim_value
                for dim in tensor.shape.dim
            )
            if dtype != source.dtype or shape != source.shape:
                raise ValueError(
                    "Normalization actual boundary dtype/shape differs from native ports"
                )
        if (
            len(report.inputs) != 1
            or len(report.traced_outputs) != 1
            or len(report.exported_outputs) != 1
        ):
            raise ValueError("Normalization report port arity differs")
        for source, reported in (
            (prepared.traced_inputs[0], report.inputs[0].traced),
            (prepared.traced_outputs[0], report.traced_outputs[0]),
            (prepared.traced_outputs[0], report.exported_outputs[0]),
        ):
            if (reported.dtype, reported.shape) != (source.dtype, source.shape):
                raise ValueError("Normalization report boundary dtype/shape differs")
        if (
            report.inputs[0].traced.weak_type != prepared.traced_inputs[0].weak_type
            or report.traced_outputs[0].weak_type
            != prepared.traced_outputs[0].weak_type
        ):
            raise ValueError("Normalization report native port weakness differs")
        if any(
            node.op_type in {"LayerNormalization", "RMSNormalization"}
            for node in model.graph.node
        ):
            raise ValueError(
                "Normalization counterpart forbids a fused replacement of original arithmetic"
            )
        native = _NormalizationExpressions.native(prepared.original_jaxpr)
        serialized = _NormalizationExpressions(model).outputs()
        if native != serialized:
            raise ValueError(
                "Consumed normalization typed DAG/constants/axes/casts/ports differ from the actual original native computation"
            )

    def check(
        self, prepared: PreparedSourceCase, model: onnx.ModelProto
    ) -> CorpusStructureEvidence:
        callback = prepared.metadata.get("post_check_onnx_graph")
        raw_status, raw_detail = "absent", "No original structural callback"
        if callback is not None:
            try:
                passed = bool(callback(model))
                raw_status = "passed" if passed else "failed"
                raw_detail = "Original raw callback returned " + str(passed)
            except Exception as exc:
                raw_status, raw_detail = "error", f"{type(exc).__name__}: {exc}"
        try:
            self.check_source(prepared)
            self._check_model(prepared, model)
            status, detail = (
                CheckStatus.VALIDATED,
                "Original raw LayerNorm typed consumed DAG and exact constant/axis/opset correspondence; numerical evidence remains separate",
            )
        except Exception as exc:
            status, detail = CheckStatus.FAILED, str(exc)
        return CorpusStructureEvidence(
            self.identifier, self.sha256, raw_status, raw_detail, status, detail
        )


NORMALIZATION_COUNTERPARTS: Final[tuple[NormalizationCounterpart, ...]] = tuple(
    NormalizationCounterpart(case) for case in NORMALIZATION_CASES
)


def normalization_counterpart_for(
    origin: str, ordinal: int, contract_sha256: str
) -> NormalizationCounterpart | None:
    return next(
        (
            item
            for item in NORMALIZATION_COUNTERPARTS
            if (item.case.origin, item.case.ordinal, item.case.contract_sha256)
            == (origin, ordinal, contract_sha256)
        ),
        None,
    )
