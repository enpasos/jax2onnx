# tests/extra_tests/converter/source_precision_corpus_oracles.py

"""Fixed corpus structural counterparts and proposal-only numerical provenance.

These checks preserve raw callback failures. They neither admit a primitive nor
supply numerical criteria to a source scenario or execute runtime samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Final, TYPE_CHECKING

from flax import nnx
import numpy as np
import onnx

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
    "f3b4753d24f2e2ebb4da20919adadf2c5da203ee8af8fc9033a0bf89bd39447c"
)
ORACLE_VERSION: Final = "static-dot-source-counterpart-v2"
LINEAR_SOURCE_PATH: Final = "jax2onnx/plugins/flax/nnx/linear.py"
LINEAR_SOURCE_SHA256: Final = (
    "3a3e17bd6d3ad1e259f0ebe293d8b303fec58e37569d5271515eef5b433660ba"
)
LINEAR_ORACLE_VERSION: Final = "leading-b-linear-source-counterpart-v1"
GATHER_SOURCE_PATH: Final = "jax2onnx/plugins/jax/lax/gather.py"
GATHER_SOURCE_SHA256: Final = (
    "2d95f90f2ecc5d7bc1bf1f25e7b2a5f6978fbcc278ebb0b5aca181b22834b80c"
)
GATHER_ORACLE_VERSION: Final = "gather-trig-native-where-source-counterpart-v1"
# Complete original native traces, including index normalization and Where JIT.
_GATHER_NATIVE_TRACE_SHA256: Final = (
    "9a14661837181e7170fd3d2e0a780d7fd927299c4c96349ad05cb185926f3cda",
    "44fef1e2b0e2260469c30e76bf4186b75b73a4c0f52d7de9153bbbf61e8fcac1",
)
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
_GATHER_GRAPH_TEMPLATE: Final[dict[str, Any]] = json.loads(
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
        from jax2onnx._compat.jax import native_jit_boolean_inline_hints
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
        native_index = 0 if native_jit_boolean_inline_hints()[0] is False else 1
        if (
            json.loads(prepared.original_trace_json) != actual
            or fingerprint(actual) != _GATHER_NATIVE_TRACE_SHA256[native_index]
        ):
            raise ValueError(
                "Gather counterpart complete original native trace changed"
            )

    def _check_model(self, model: onnx.ModelProto) -> None:
        if _gather_graph_signature(model) != _GATHER_GRAPH_TEMPLATE:
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
                "Exact original Gather/index/trig/native-Where structure checked; runtime and numerics unverified",
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
) -> DotCounterpart | LinearCounterpart | GatherCounterpart | None:
    return next(
        (
            item
            for item in (*DOT_COUNTERPARTS, *LINEAR_COUNTERPARTS, *GATHER_COUNTERPARTS)
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
