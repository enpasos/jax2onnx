# jax2onnx/converter/source_deployment.py

"""Declarative private deployment requirements, separate from actual execution.

The initial closed capability set names reviewed ORT builds and one hardware
class. Unknown targets retain the existing precision rejection. Extending this
set requires additional target evidence; it is not a permanent platform limit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import re
from typing import Any
from urllib.parse import quote

from jax2onnx._compat.jax import is_native_precision_pair

DEPLOYMENT_METADATA_KEY: str = "jax2onnx.deployment"
_FP32_CONTRACT: str = "ort-cpu-x86-fp32-sgemm-v1"


def _deployment_graph_scope(name: str) -> str:
    return "graph:" + quote(name, safe="")


def _deployment_function_scope(domain: str, name: str, overload: str) -> str:
    scope = "function:" + quote(domain, safe="") + ":" + quote(name, safe="")
    return scope if not overload else scope + ":overload:" + quote(overload, safe="")


def _deployment_child_scope(parent: str, tag: str, attribute: str, index: int) -> str:
    if (
        re.fullmatch(r"jax2onnx_src_[0-9]{6}", tag) is None
        or type(index) is not int
        or index < 0
    ):
        raise ValueError("Invalid deployment lexical containment")
    return f"{parent}/node:{tag}/attribute:{quote(attribute, safe='')}/graph:{index}"


@dataclass(frozen=True)
class DeploymentTarget:
    runtime: str
    runtime_version: str
    runtime_build: str
    system: str
    architecture: str
    cpu_vendor: str
    cpu_model: str
    cpu_features: tuple[str, ...]
    libraries: tuple[tuple[str, str], ...]
    providers: tuple[str, ...]
    provider_options: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    session_config: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in (
            "runtime",
            "runtime_version",
            "runtime_build",
            "system",
            "architecture",
            "cpu_vendor",
            "cpu_model",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"Invalid deployment target {name}")
        for name in ("cpu_features", "providers"):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not str for item in value):
                raise ValueError(f"Invalid deployment target {name}")
            if len(set(value)) != len(value):
                raise ValueError(f"Duplicate deployment target {name}")
        object.__setattr__(self, "cpu_features", tuple(sorted(self.cpu_features)))
        for name in ("libraries", "session_config"):
            _string_pairs(getattr(self, name), name)
            object.__setattr__(self, name, tuple(sorted(getattr(self, name))))
        if any(
            "/" in name or "\\" in name or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for name, digest in self.libraries
        ):
            raise ValueError("Invalid deployment library identity")
        if type(self.provider_options) is not tuple:
            raise ValueError("Invalid deployment provider options")
        for item in self.provider_options:
            if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str:
                raise ValueError("Invalid deployment provider options")
            _string_pairs(item[1], "provider options")
        if len({name for name, _ in self.provider_options}) != len(
            self.provider_options
        ):
            raise ValueError("Duplicate deployment provider options")
        object.__setattr__(
            self,
            "provider_options",
            tuple(
                sorted(
                    (name, tuple(sorted(options)))
                    for name, options in self.provider_options
                )
            ),
        )


def _string_pairs(value: object, name: str) -> None:
    if type(value) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or any(type(part) is not str for part in item)
        for item in value
    ):
        raise ValueError(f"Invalid deployment {name}")
    if len({item[0] for item in value}) != len(value):
        raise ValueError(f"Duplicate deployment {name}")


@dataclass(frozen=True)
class MatrixPrecisionRequirement:
    source_scope: str
    source_equation: str
    output: str
    source_precision: tuple[str, str]
    dtypes: tuple[str, str, str]
    preferred_element_type: str | None
    kernel_contract: str
    matrix_output: str | None = None
    matrix_scope: str | None = None
    matrix_tag: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.source_scope) is not str
            or not self.source_scope
            or type(self.source_equation) is not str
            or not self.source_equation
            or type(self.output) is not str
            or not self.output
            or type(self.source_precision) is not tuple
            or self.source_precision != ("HIGHEST", "HIGHEST")
            or type(self.dtypes) is not tuple
            or self.dtypes != ("float32", "float32", "float32")
            or self.preferred_element_type not in (None, "float32")
            or self.kernel_contract != _FP32_CONTRACT
            or any(
                value is not None and (type(value) is not str or not value)
                for value in (self.matrix_output, self.matrix_scope, self.matrix_tag)
            )
            or self.matrix_tag is not None
            and re.fullmatch(r"jax2onnx_src_[0-9]{6}", self.matrix_tag) is None
        ):
            raise ValueError("Invalid matrix precision deployment requirement")


@dataclass(frozen=True)
class DeploymentRequirements:
    target: DeploymentTarget
    matrix_precision: tuple[MatrixPrecisionRequirement, ...]

    def __post_init__(self) -> None:
        if (
            type(self.target) is not DeploymentTarget
            or type(self.matrix_precision) is not tuple
        ):
            raise ValueError("Invalid deployment requirements")
        if any(
            type(item) is not MatrixPrecisionRequirement
            for item in self.matrix_precision
        ):
            raise ValueError("Invalid matrix deployment requirements")
        if self.matrix_precision and reviewed_fp32_contract(self.target) is None:
            raise ValueError(
                "Unreviewed target cannot claim a matrix precision contract"
            )


@dataclass(frozen=True)
class RuntimeTargetObservation:
    target: DeploymentTarget
    optimization: str
    execution_mode: str
    intra_op_threads: int
    inter_op_threads: int
    distribution_version: str
    optimized_graph_sha256: str | None
    boundary_contract: str | None
    boundary_error: str | None


# Exact package-library identities are normalized to basenames. These three
# reviewed points include the two current Python ABIs and the direct floor.
_REVIEWED_TARGETS: tuple[DeploymentTarget, ...] = (
    DeploymentTarget(
        runtime="onnxruntime",
        runtime_version="1.31.0",
        runtime_build="ORT Build Info: git-branch=HEAD, git-commit-id=402979a, fp8-kv-cache=1, build type=Release",
        system="Linux",
        architecture="x86_64",
        cpu_vendor="GenuineIntel",
        cpu_model="Intel(R) Xeon(R) E-2286M  CPU @ 2.40GHz",
        cpu_features=("avx", "avx2", "fma", "sse2"),
        libraries=(
            (
                "libonnxruntime.so.1.31.0",
                "21901b1b02233484b80888c16f4f5473742f796bf5528d5d0c4073dbe9802001",
            ),
            (
                "libonnxruntime_providers_shared.so",
                "086ec1d5388f64153d9c63470d126693db9a182c8ce236d3a1119068471b8a0d",
            ),
            (
                "onnxruntime_pybind11_state.cpython-312-x86_64-linux-gnu.so",
                "2377ce1c968f274da1e694680ea8dc19fdc619bd61c192d8b2dff3362710ec4e",
            ),
        ),
        providers=("CPUExecutionProvider",),
        provider_options=(("CPUExecutionProvider", ()),),
        session_config=(
            ("mlas.disable_kleidiai", "1"),
            ("mlas.enable_gemm_fastmath_arm64_bfloat16", "0"),
            ("session.set_denormal_as_zero", "0"),
        ),
    ),
    DeploymentTarget(
        runtime="onnxruntime",
        runtime_version="1.31.0",
        runtime_build="ORT Build Info: git-branch=HEAD, git-commit-id=402979a, fp8-kv-cache=1, build type=Release",
        system="Linux",
        architecture="x86_64",
        cpu_vendor="GenuineIntel",
        cpu_model="Intel(R) Xeon(R) E-2286M  CPU @ 2.40GHz",
        cpu_features=("avx", "avx2", "fma", "sse2"),
        libraries=(
            (
                "libonnxruntime.so.1.31.0",
                "21901b1b02233484b80888c16f4f5473742f796bf5528d5d0c4073dbe9802001",
            ),
            (
                "libonnxruntime_providers_shared.so",
                "086ec1d5388f64153d9c63470d126693db9a182c8ce236d3a1119068471b8a0d",
            ),
            (
                "onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so",
                "c8aaaf429bef152a634aa062024fc8024c773b9d03a024617d48201fb0e411ab",
            ),
        ),
        providers=("CPUExecutionProvider",),
        provider_options=(("CPUExecutionProvider", ()),),
        session_config=(
            ("mlas.disable_kleidiai", "1"),
            ("mlas.enable_gemm_fastmath_arm64_bfloat16", "0"),
            ("session.set_denormal_as_zero", "0"),
        ),
    ),
    DeploymentTarget(
        runtime="onnxruntime",
        runtime_version="1.24.1",
        runtime_build="ORT Build Info: git-branch=HEAD, git-commit-id=470ae16099, build type=Release",
        system="Linux",
        architecture="x86_64",
        cpu_vendor="GenuineIntel",
        cpu_model="Intel(R) Xeon(R) E-2286M  CPU @ 2.40GHz",
        cpu_features=("avx", "avx2", "fma", "sse2"),
        libraries=(
            (
                "libonnxruntime.so.1.24.1",
                "94ac52f650e195b784587e1cb7d25b4bf074048b50fa9a7127795b2ed34db22e",
            ),
            (
                "libonnxruntime_providers_shared.so",
                "086ec1d5388f64153d9c63470d126693db9a182c8ce236d3a1119068471b8a0d",
            ),
            (
                "onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so",
                "6cf7047a6586d5a85543975e35523d1adf09c8131bc40bd510e3880bd9f92307",
            ),
        ),
        providers=("CPUExecutionProvider",),
        provider_options=(("CPUExecutionProvider", ()),),
        session_config=(
            ("mlas.enable_gemm_fastmath_arm64_bfloat16", "0"),
            ("session.set_denormal_as_zero", "0"),
        ),
    ),
)


def reviewed_fp32_contract(target: DeploymentTarget | None) -> str | None:
    if type(target) is DeploymentTarget and target in _REVIEWED_TARGETS:
        return _FP32_CONTRACT
    return None


def ort_cpu_deployment_target(
    runtime_version: str, python_tag: str
) -> DeploymentTarget:
    """Explicitly select a reviewed declaration; this performs no observation."""
    for target in _REVIEWED_TARGETS:
        if target.runtime_version == runtime_version and any(
            name.startswith("onnxruntime_pybind11_state." + python_tag + "-")
            for name, _ in target.libraries
        ):
            return target
    raise ValueError("No reviewed deployment declaration for this runtime/ABI")


def matrix_precision_contract(
    target: DeploymentTarget | None,
    precision: Any,
    dtypes: tuple[str, ...],
    preferred_element_type: str | None,
) -> str | None:
    """Shared profile/lowerer proof; the target is an explicit requirement."""
    if (
        type(precision) is not tuple
        or not is_native_precision_pair(precision)
        or any(item.name != "HIGHEST" for item in precision)
        or dtypes != ("float32", "float32", "float32")
        or preferred_element_type not in (None, "float32")
    ):
        return None
    return reviewed_fp32_contract(target)


def deployment_requirements_to_json(requirements: DeploymentRequirements) -> str:
    return json.dumps(asdict(requirements), sort_keys=True, separators=(",", ":"))


def deployment_requirements_from_json(
    payload: str | None,
) -> DeploymentRequirements | None:
    if payload is None:
        return None
    data = json.loads(payload)
    if type(data) is not dict or set(data) != {"target", "matrix_precision"}:
        raise ValueError("Invalid deployment requirements schema")
    raw = data["target"]
    if (
        type(raw) is not dict
        or set(raw) != {item.name for item in fields(DeploymentTarget)}
        or any(
            type(raw[name]) is not list
            for name in (
                "cpu_features",
                "libraries",
                "providers",
                "provider_options",
                "session_config",
            )
        )
        or type(data["matrix_precision"]) is not list
    ):
        raise ValueError("Invalid deployment target schema")
    for item in data["matrix_precision"]:
        if (
            type(item) is not dict
            or set(item) != {field.name for field in fields(MatrixPrecisionRequirement)}
            or type(item["source_precision"]) is not list
            or type(item["dtypes"]) is not list
        ):
            raise ValueError("Invalid matrix precision requirement schema")
    target = DeploymentTarget(
        **{
            **raw,
            "cpu_features": tuple(raw["cpu_features"]),
            "libraries": tuple(tuple(item) for item in raw["libraries"]),
            "providers": tuple(raw["providers"]),
            "provider_options": tuple(
                (name, tuple(tuple(item) for item in options))
                for name, options in raw["provider_options"]
            ),
            "session_config": tuple(tuple(item) for item in raw["session_config"]),
        }
    )
    return DeploymentRequirements(
        target,
        tuple(
            MatrixPrecisionRequirement(
                **{
                    **item,
                    "source_precision": tuple(item["source_precision"]),
                    "dtypes": tuple(item["dtypes"]),
                }
            )
            for item in data["matrix_precision"]
        ),
    )


def require_runtime_target(
    required: DeploymentRequirements,
    actual: RuntimeTargetObservation,
) -> None:
    """Reject a deployment mismatch before model execution; grant no numerics."""
    if (
        reviewed_fp32_contract(required.target) is None
        or actual.target != required.target
        or actual.optimization not in ("ORT_DISABLE_ALL", "ORT_ENABLE_ALL")
        or actual.execution_mode != "ORT_SEQUENTIAL"
        or actual.intra_op_threads != 1
        or actual.inter_op_threads != 1
        or actual.distribution_version != actual.target.runtime_version
        or actual.optimized_graph_sha256 is None
        or re.fullmatch(r"[0-9a-f]{64}", actual.optimized_graph_sha256) is None
        or actual.boundary_error is not None
        or any(
            site.matrix_output is None
            or site.matrix_scope is None
            or site.matrix_tag is None
            for site in required.matrix_precision
        )
        or actual.boundary_contract
        not in {
            "matmul-transpose-and-original-casts-v1",
            "matmul-gemm-quickgelu-and-original-casts-v2",
        }
    ):
        raise ValueError(
            "Actual runtime does not satisfy the artifact deployment requirements"
            + ("; " + actual.boundary_error if actual.boundary_error else "")
        )
