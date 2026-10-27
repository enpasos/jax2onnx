# jax2onnx/_compat/_cast_activation_options.py

"""Unused, source-pinned semantic checks of untrusted current cast options.

Only FINAL ``NativeCastLowering.options`` belongs to the later conversion-owned
binding. This pure validator cannot establish that provenance, approve a build,
select an algorithm, or issue authority. Query/historical records and even a
supported result remain caller-provided semantic data.

Version labels select wire schemas only. Installed builds, Python ABI cohorts,
factory/provider identity and conversion-owned issuance remain separate gates.

The closed DebugOptions catalogs are from XLA 5a9e73cbd925 (JAX 0.10.2) and
d867969800d4 (JAX 0.12.0.1), reviewed in catalog d3e112ab9719944e and joined
contract 89a2b5941ff1cbc7. Catalog membership never uses a field-name prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any


_LIMIT = 65536
_ITEMS = 4096
_TEXT = 4096


@dataclass(frozen=True, slots=True)
class OptionFact:
    scope: str
    field: int
    present: bool
    value: object
    classification: str


@dataclass(frozen=True, slots=True)
class CurrentCastOptions:
    status: str
    schema: str
    reasons: tuple[str, ...]
    payload: bytes
    payload_sha256: str
    facts: tuple[OptionFact, ...]
    pending_bindings: tuple[str, ...] = (
        "final-original-native-lowering-and-module",
        "same-client-device-and-current-config",
        "propagation-and-detailed-logging",
        "reviewed-build-target-and-provider",
        "conversion-owned-lease-cleanup",
    )
    scope: str = "untrusted-current-options-semantics-not-kernel-authority"


@dataclass(frozen=True, slots=True)
class _Field:
    kind: str
    repeated: bool = False
    classification: str = "CURRENT_CONSTRUCTION"


@dataclass
class _Budget:
    items: int = 0

    def consume(self) -> None:
        self.items += 1
        if self.items > _ITEMS:
            raise _WireError("options-item-bound")


class _WireError(ValueError):
    pass


class _Unreviewed(ValueError):
    pass


_ENUMS_OLD = {
    "AutotuneCacheMode": (0, 1, 2),
    "CollectiveOpType": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "CollectivesMode": (0, 1, 2, 3),
    "CommandBufferCmdType": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
    "CommandBufferSchedulingMode": (0, 1, 2, 3),
    "CommandBufferUpdateMode": (0, 1, 2),
    "CpuOptPreset": (0, 1, 2),
    "CpuSchedulerType": (0, 1, 2),
    "DetectionMode": (0, 1, 2),
    "DevicelessCubMode": (0, 1, 2, 3, 4),
    "EffortLevel": (0, 9, 19, 29, 39),
    "LibNvJitLinkMode": (0, 1, 2),
    "LibraryFusionType": (0, 1, 2, 3, 4, 5),
    "PGLEStrictnessLevel": (0, 1, 2),
    "PartitioningAlgorithm": (0, 1, 2, 3),
    "PipelineParallelismOptLevel": (0, 1),
    "Precision": (0, 1, 2),
    "PrimitiveType": (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
    ),
    "ShapeChecks": (0, 1, 2),
    "StepMarkerLocation": (0, 1, 2, 3),
    "WhileLoopUnrolling": (0, 1, 2, 3, 4),
    "XnnGraphFusionMode": (0, 1, 2, 3),
    "autotuner.Backend": (0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16),
}


_DEBUG_OLD = (
    (
        "ThunkBufferDebugFilter",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (424,),
    ),
    ("bool", False, "BACKEND_IRRELEVANT", (174,)),
    ("bool", False, "BACKEND_OR_OPERATION_IRRELEVANT", (135, 136, 317)),
    ("bool", False, "DIAGNOSTIC_ONLY", (137,)),
    (
        "bool",
        False,
        "DIAGNOSTIC_OR_SERIALIZATION_ONLY",
        (
            2,
            33,
            92,
            112,
            113,
            114,
            115,
            116,
            118,
            131,
            144,
            149,
            151,
            153,
            164,
            182,
            185,
            253,
            290,
            315,
            381,
            405,
            436,
            447,
            466,
            502,
        ),
    ),
    (
        "bool",
        False,
        "FIRST_DEFAULT_ROUTE",
        (
            9,
            70,
            71,
            72,
            104,
            300,
            307,
            337,
            344,
            363,
            364,
            370,
            376,
            379,
            380,
            397,
            408,
            419,
            429,
            439,
            452,
            479,
            512,
        ),
    ),
    ("bool", False, "NAMING_ONLY", (372,)),
    ("bool", False, "OPERATION_IRRELEVANT", (35, 60, 106, 107, 175)),
    (
        "bool",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (
            62,
            100,
            101,
            103,
            105,
            125,
            138,
            146,
            148,
            155,
            165,
            166,
            181,
            186,
            188,
            189,
            190,
            203,
            209,
            215,
            216,
            217,
            219,
            227,
            231,
            232,
            235,
            236,
            239,
            240,
            244,
            245,
            246,
            247,
            248,
            250,
            254,
            255,
            257,
            259,
            262,
            267,
            268,
            269,
            272,
            280,
            284,
            287,
            291,
            296,
            297,
            301,
            304,
            305,
            312,
            321,
            322,
            329,
            334,
            338,
            340,
            342,
            345,
            347,
            348,
            349,
            350,
            353,
            356,
            360,
            362,
            366,
            368,
            375,
            382,
            384,
            387,
            388,
            389,
            390,
            393,
            401,
            406,
            409,
            410,
            411,
            413,
            414,
            415,
            420,
            421,
            427,
            431,
            435,
            437,
            440,
            441,
            444,
            445,
            450,
            453,
            454,
            455,
            456,
            460,
            462,
            470,
            471,
            475,
            478,
            480,
            481,
            484,
            488,
            490,
            492,
            501,
        ),
    ),
    (
        "bool",
        False,
        "PRIOR_EXACT_OPTION_TABLE",
        (73, 97, 99, 120, 121, 122, 126, 129, 140, 252, 335, 359, 412, 425, 489, 493),
    ),
    ("bool", False, "TEST_DRIVER_ONLY", (90, 91, 373)),
    (
        "enum:AutotuneCacheMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (324,),
    ),
    (
        "enum:CollectiveOpType",
        True,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (289,),
    ),
    (
        "enum:CollectivesMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (473, 482, 485),
    ),
    (
        "enum:CommandBufferCmdType",
        True,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (258,),
    ),
    (
        "enum:CommandBufferSchedulingMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (404,),
    ),
    (
        "enum:CommandBufferUpdateMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (469,),
    ),
    ("enum:CpuOptPreset", False, "PRIOR_EXACT_OPTION_TABLE", (467,)),
    ("enum:CpuSchedulerType", False, "FIRST_DEFAULT_ROUTE", (448,)),
    ("enum:DetectionMode", False, "OPERATION_IRRELEVANT", (403, 432)),
    (
        "enum:DetectionMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (426, 428),
    ),
    (
        "enum:DevicelessCubMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (483,),
    ),
    (
        "enum:LibNvJitLinkMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (343,),
    ),
    ("enum:LibraryFusionType", True, "PRIOR_EXACT_OPTION_TABLE", (399, 400, 422)),
    (
        "enum:PGLEStrictnessLevel",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (341,),
    ),
    ("enum:PartitioningAlgorithm", False, "FIRST_DEFAULT_ROUTE", (187,)),
    (
        "enum:PipelineParallelismOptLevel",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (351,),
    ),
    ("enum:ShapeChecks", False, "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE", (170,)),
    ("enum:StepMarkerLocation", False, "FIRST_DEFAULT_ROUTE", (108,)),
    (
        "enum:WhileLoopUnrolling",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (294,),
    ),
    ("enum:XnnGraphFusionMode", False, "PRIOR_EXACT_OPTION_TABLE", (365,)),
    (
        "enum:autotuner.Backend",
        True,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (442,),
    ),
    ("float", False, "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE", (225, 316)),
    ("int32", False, "ACTUAL_CLIENT_BOUND", (102,)),
    ("int32", False, "DIAGNOSTIC_ONLY", (395,)),
    ("int32", False, "DIAGNOSTIC_OR_SERIALIZATION_ONLY", (132,)),
    ("int32", False, "FIRST_DEFAULT_ROUTE", (142, 323)),
    ("int32", False, "OPERATION_IRRELEVANT", (417, 418, 438, 463)),
    (
        "int32",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (
            123,
            147,
            159,
            205,
            208,
            224,
            256,
            260,
            285,
            318,
            327,
            328,
            331,
            336,
            391,
            392,
            457,
            459,
            465,
            472,
            476,
            486,
            487,
        ),
    ),
    ("int32", False, "PRIOR_EXACT_OPTION_TABLE", (31, 308)),
    ("int64", False, "BACKEND_OR_OPERATION_IRRELEVANT", (311,)),
    ("int64", False, "DIAGNOSTIC_OR_SERIALIZATION_ONLY", (251,)),
    ("int64", False, "OPERATION_IRRELEVANT", (293,)),
    (
        "int64",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (
            157,
            163,
            212,
            213,
            228,
            237,
            265,
            273,
            274,
            277,
            283,
            288,
            339,
            374,
            378,
            416,
            449,
            461,
        ),
    ),
    ("string", False, "DIAGNOSTIC_OR_SERIALIZATION_ONLY", (109, 110, 111, 154, 433)),
    (
        "string",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (61, 128, 210, 222, 223, 261, 292, 295, 306, 310, 377, 383, 407, 434, 451, 468),
    ),
    ("string", False, "PRIOR_EXACT_OPTION_TABLE", (333,)),
    ("string", True, "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE", (127, 150, 491)),
    ("string", True, "PRIOR_EXACT_OPTION_TABLE", (30, 124)),
    ("string_map", False, "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE", (357, 474)),
    ("string_map", False, "PRIOR_EXACT_OPTION_TABLE", (500,)),
)


_ENUMS_MODERN = {
    "AutotuneCacheMode": (0, 1, 2, 3),
    "CollectiveKernelType": (0, 1, 2, 3, 4, 8),
    "CollectiveOpType": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "CollectivePipeliningMode": (0, 1, 2, 3),
    "CollectivesMode": (0, 1, 2, 3),
    "CommandBufferCmdType": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13),
    "CommandBufferSchedulingMode": (0, 1, 2, 3),
    "CommandBufferUpdateMode": (0, 1, 2),
    "CpuOptPreset": (0, 1, 2),
    "CpuSchedulerType": (0, 1, 2),
    "CudnnDevicelessCompilationMode": (0, 1, 2, 3),
    "DetectionMode": (0, 1, 2, 3),
    "DeterministicOpsTritonMode": (0, 1, 2),
    "DevicelessCubMode": (0, 1, 2, 3, 4),
    "EarlyExitPoint": (0, 1),
    "EffortLevel": (0, 9, 19, 29, 39),
    "LibNvJitLinkMode": (0, 1, 2),
    "LibraryFusionType": (0, 1, 2, 3, 4, 5),
    "PGLEStrictnessLevel": (0, 1, 2),
    "PartitioningAlgorithm": (0, 1, 2, 3),
    "PipelineParallelismOptLevel": (0, 1),
    "Precision": (0, 1, 2),
    "PrimitiveType": (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
    ),
    "ShapeChecks": (0, 1, 2),
    "StepMarkerLocation": (0, 1, 2, 3),
    "WhileLoopUnrolling": (0, 1, 2, 3, 4),
    "XnnGraphFusionMode": (0, 1, 2, 3),
    "autotuner.Backend": (0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19),
}


_DEBUG_MODERN = (
    ("CollectiveFilter", True, "BACKEND_OR_OPERATION_IRRELEVANT", (503,)),
    (
        "ThunkBufferDebugFilter",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (424,),
    ),
    ("bool", False, "BACKEND_IRRELEVANT", (174,)),
    ("bool", False, "BACKEND_OR_OPERATION_IRRELEVANT", (135, 136, 317, 520)),
    ("bool", False, "DIAGNOSTIC_ONLY", (137,)),
    (
        "bool",
        False,
        "DIAGNOSTIC_OR_SERIALIZATION_ONLY",
        (
            2,
            33,
            92,
            112,
            113,
            114,
            115,
            116,
            118,
            131,
            144,
            149,
            151,
            153,
            164,
            182,
            185,
            253,
            290,
            315,
            381,
            405,
            436,
            447,
            466,
            495,
            498,
            502,
            535,
        ),
    ),
    (
        "bool",
        False,
        "FIRST_DEFAULT_ROUTE",
        (
            9,
            70,
            71,
            72,
            104,
            307,
            337,
            344,
            363,
            364,
            370,
            379,
            380,
            397,
            408,
            419,
            429,
            439,
            452,
            479,
            512,
            529,
            539,
        ),
    ),
    ("bool", False, "NAMING_ONLY", (372,)),
    ("bool", False, "OPERATION_IRRELEVANT", (35, 60, 106, 107, 175, 548)),
    (
        "bool",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (
            62,
            100,
            101,
            103,
            105,
            125,
            138,
            146,
            148,
            155,
            166,
            181,
            186,
            188,
            189,
            190,
            203,
            209,
            215,
            216,
            219,
            232,
            236,
            240,
            244,
            245,
            246,
            247,
            248,
            250,
            254,
            255,
            257,
            259,
            262,
            267,
            269,
            272,
            280,
            284,
            287,
            291,
            296,
            297,
            301,
            304,
            305,
            312,
            321,
            322,
            329,
            334,
            338,
            340,
            342,
            345,
            347,
            348,
            349,
            350,
            353,
            356,
            360,
            362,
            366,
            368,
            375,
            382,
            384,
            387,
            390,
            393,
            401,
            406,
            409,
            410,
            411,
            414,
            415,
            421,
            427,
            431,
            437,
            440,
            441,
            444,
            445,
            453,
            454,
            455,
            456,
            460,
            462,
            470,
            471,
            475,
            478,
            481,
            484,
            488,
            490,
            492,
            494,
            501,
            504,
            506,
            513,
            515,
            521,
            524,
            525,
            531,
            532,
            533,
            534,
            545,
            546,
            550,
            551,
            552,
            555,
            557,
        ),
    ),
    (
        "bool",
        False,
        "PRIOR_EXACT_OPTION_TABLE",
        (73, 97, 99, 120, 121, 122, 126, 129, 140, 252, 335, 359, 412, 425, 489, 493),
    ),
    ("bool", False, "TEST_DRIVER_ONLY", (90, 91, 373)),
    (
        "enum:AutotuneCacheMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (324,),
    ),
    (
        "enum:CollectiveKernelType",
        True,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (516,),
    ),
    (
        "enum:CollectiveOpType",
        True,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (289, 497, 541),
    ),
    (
        "enum:CollectivePipeliningMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (517, 518, 519),
    ),
    (
        "enum:CollectivesMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (473, 482, 485),
    ),
    (
        "enum:CommandBufferCmdType",
        True,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (258,),
    ),
    (
        "enum:CommandBufferSchedulingMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (404,),
    ),
    (
        "enum:CommandBufferUpdateMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (469,),
    ),
    ("enum:CpuOptPreset", False, "PRIOR_EXACT_OPTION_TABLE", (467,)),
    ("enum:CpuSchedulerType", False, "FIRST_DEFAULT_ROUTE", (448,)),
    (
        "enum:CudnnDevicelessCompilationMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (505,),
    ),
    ("enum:DetectionMode", False, "OPERATION_IRRELEVANT", (403, 432)),
    (
        "enum:DetectionMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (426, 428),
    ),
    (
        "enum:DeterministicOpsTritonMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (556,),
    ),
    (
        "enum:DevicelessCubMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (483,),
    ),
    (
        "enum:EarlyExitPoint",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (549,),
    ),
    (
        "enum:LibNvJitLinkMode",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (343,),
    ),
    ("enum:LibraryFusionType", True, "PRIOR_EXACT_OPTION_TABLE", (399, 400, 422)),
    (
        "enum:PGLEStrictnessLevel",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (341,),
    ),
    ("enum:PartitioningAlgorithm", False, "FIRST_DEFAULT_ROUTE", (187,)),
    (
        "enum:PipelineParallelismOptLevel",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (351,),
    ),
    ("enum:ShapeChecks", False, "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE", (170,)),
    ("enum:StepMarkerLocation", False, "FIRST_DEFAULT_ROUTE", (108,)),
    (
        "enum:WhileLoopUnrolling",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (294,),
    ),
    ("enum:XnnGraphFusionMode", False, "PRIOR_EXACT_OPTION_TABLE", (365,)),
    ("enum:autotuner.Backend", False, "FIRST_DEFAULT_ROUTE", (547,)),
    (
        "enum:autotuner.Backend",
        True,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (442,),
    ),
    ("float", False, "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE", (225, 316)),
    ("int32", False, "ACTUAL_CLIENT_BOUND", (102,)),
    ("int32", False, "DIAGNOSTIC_ONLY", (395,)),
    ("int32", False, "DIAGNOSTIC_OR_SERIALIZATION_ONLY", (132,)),
    ("int32", False, "FIRST_DEFAULT_ROUTE", (142, 323)),
    ("int32", False, "OPERATION_IRRELEVANT", (417, 418, 438, 463)),
    (
        "int32",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (
            123,
            147,
            159,
            205,
            208,
            224,
            256,
            260,
            285,
            318,
            327,
            328,
            331,
            336,
            391,
            392,
            457,
            459,
            465,
            472,
            476,
            486,
            487,
            527,
            530,
            538,
        ),
    ),
    ("int32", False, "PRIOR_EXACT_OPTION_TABLE", (31, 308)),
    ("int64", False, "BACKEND_OR_OPERATION_IRRELEVANT", (311,)),
    ("int64", False, "DIAGNOSTIC_OR_SERIALIZATION_ONLY", (251, 542)),
    ("int64", False, "OPERATION_IRRELEVANT", (293, 528)),
    (
        "int64",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (
            157,
            163,
            212,
            213,
            228,
            237,
            265,
            273,
            274,
            277,
            283,
            288,
            339,
            374,
            378,
            416,
            449,
            461,
            514,
            522,
            523,
            526,
        ),
    ),
    ("string", False, "DIAGNOSTIC_OR_SERIALIZATION_ONLY", (109, 110, 111, 154, 433)),
    ("string", False, "FIRST_DEFAULT_ROUTE", (496, 543, 544)),
    (
        "string",
        False,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (
            61,
            128,
            210,
            222,
            223,
            261,
            292,
            295,
            306,
            310,
            377,
            383,
            407,
            434,
            451,
            468,
            536,
            540,
            553,
            554,
        ),
    ),
    ("string", False, "PRIOR_EXACT_OPTION_TABLE", (333,)),
    (
        "string",
        True,
        "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE",
        (127, 150, 491, 537),
    ),
    ("string", True, "PRIOR_EXACT_OPTION_TABLE", (30, 124)),
    ("string_map", False, "PINNED_GPU_FIELD_NOT_CONSUMED_BY_CPU_ROUTE", (357, 474)),
    ("string_map", False, "PRIOR_EXACT_OPTION_TABLE", (500,)),
)


_REQUIRED_DEBUG = {
    9: False,
    31: 3,
    70: True,
    71: True,
    72: True,
    73: False,
    97: False,
    99: False,
    104: False,
    108: 0,
    142: -1,
    187: 0,
    307: False,
    308: 256,
    323: 32,
    333: "",
    335: False,
    337: False,
    344: False,
    359: True,
    363: False,
    364: False,
    365: 0,
    370: False,
    379: False,
    380: False,
    397: False,
    399: (),
    400: (),
    408: False,
    412: False,
    419: False,
    422: (4, 5, 3),
    425: True,
    429: False,
    439: False,
    448: 0,
    467: 0,
    479: True,
    493: True,
}


_REQUIRED_OLD = {300: True, 376: True, 452: False, 489: False, 512: False}


_REQUIRED_MODERN = {
    452: True,
    489: True,
    496: "",
    512: False,
    529: True,
    539: False,
    543: "",
    544: "",
    547: 0,
}


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    start, result = offset, 0
    for index in range(10):
        if offset >= len(data):
            raise _WireError("truncated-varint")
        byte = data[offset]
        offset += 1
        if index == 9 and byte > 1:
            raise _WireError("varint-overflow")
        result |= (byte & 127) << (7 * index)
        if byte < 128:
            if offset - start > 1 and byte == 0:
                raise _WireError("noncanonical-varint")
            return result, offset
    raise _WireError("unterminated-varint")


def _wire(data: bytes, budget: _Budget) -> tuple[tuple[int, int, int | bytes], ...]:
    result = []
    offset = 0
    while offset < len(data):
        budget.consume()
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if not 0 < number < 2**29:
            raise _WireError("field-number-bound")
        value: int | bytes
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, offset = _varint(data, offset)
            else:
                size = 8 if wire == 1 else 4
            if size > len(data) - offset:
                raise _WireError("truncated-field")
            value = data[offset : offset + size]
            offset += size
        else:
            raise _WireError("unsupported-wire-type")
        result.append((number, wire, value))
    return tuple(result)


def _signed(value: int, width: int) -> int:
    if value < 1 << (width - 1):
        return value
    if value >= (1 << 64) - (1 << (width - 1)):
        return value - (1 << 64)
    raise _WireError("signed-integer-bound")


def _text(data: bytes) -> str:
    if len(data) > _TEXT:
        raise _WireError("string-bound")
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _WireError("invalid-utf8") from exc


def _default(field: _Field) -> Any:
    if field.repeated or field.kind in ("string_map", "override_map"):
        return ()
    if field.kind == "bool":
        return False
    if field.kind == "string":
        return ""
    if field.kind == "bytes" or field.kind.startswith("opaque:"):
        return b""
    if field.kind in ("float", "double"):
        return 0.0
    if field.kind in ("int32", "int64") or field.kind.startswith("enum:"):
        return 0
    return None


def _schemas(
    version: str,
) -> tuple[dict[str, dict[int, _Field]], dict[str, tuple[int, ...]]]:
    modern = version == "0.12.0.1"
    enums: dict[str, tuple[int, ...]] = dict(_ENUMS_MODERN if modern else _ENUMS_OLD)
    debug: dict[int, _Field] = {}
    for kind, repeated, classification, numbers in (
        _DEBUG_MODERN if modern else _DEBUG_OLD
    ):
        for number in numbers:
            debug[number] = _Field(kind, repeated, classification)
    build: dict[int, _Field] = {
        1: _Field("int64"),
        2: _Field("opaque:ShapeProto"),
        3: _Field("DebugOptions"),
        4: _Field("int64"),
        5: _Field("int64"),
        6: _Field("bool"),
        7: _Field("bool"),
        8: _Field("bool"),
        9: _Field("DeviceAssignment"),
        10: _Field("bool"),
        11: _Field("bool"),
        12: _Field("bool", True),
        13: _Field("opaque:CompilationEnvironmentsProto"),
        14: _Field("bytes"),
        15: _Field("int64"),
        16: _Field("int64", True),
        17: _Field("int64", True),
        18: _Field("bool", True),
        19: _Field("bool"),
        22: _Field("int64"),
        23: _Field("int64"),
        24: _Field("enum:EffortLevel"),
        25: _Field("enum:EffortLevel"),
    }
    if modern:
        build[27] = _Field("opaque:GpuTopologyProto")
    else:
        build.update({20: _Field("float"), 21: _Field("float"), 26: _Field("int64")})
    compile_options: dict[int, _Field] = {
        1: _Field("opaque:ShapeProto", True),
        2: _Field("bool"),
        3: _Field("ExecutableBuildOptions"),
        4: _Field("bool"),
        5: _Field("int64"),
        6: _Field("bytes"),
        7: _Field("override_map"),
        8: _Field("opaque:GpuTargetConfigProto"),
        9: _Field("bool"),
        10: _Field("enum:Precision"),
        11: _Field("string"),
    }
    if modern:
        compile_options[12] = _Field("int32", True)
    return {
        "CompileOptions": compile_options,
        "ExecutableBuildOptions": build,
        "DebugOptions": debug,
        "DeviceAssignment": {
            1: _Field("int32"),
            2: _Field("int32"),
            3: _Field("ComputationDevice", True),
        },
        "ComputationDevice": {1: _Field("int64", True)},
        "ThunkBufferDebugFilter": {
            1: _Field("IntRangeInclusive", True),
            2: _Field("string", True),
        },
        "IntRangeInclusive": {1: _Field("int64"), 2: _Field("int64")},
        "CollectiveFilter": {
            1: _Field("enum:CollectiveOpType"),
            2: _Field("int64"),
            3: _Field("enum:PrimitiveType"),
        },
        "StringEntry": {1: _Field("string"), 2: _Field("string")},
        "OverrideEntry": {1: _Field("string"), 2: _Field("OptionOverride")},
        "OptionOverride": {
            1: _Field("string"),
            2: _Field("bool"),
            3: _Field("int64"),
            4: _Field("double"),
        },
    }, enums


def _decode(
    kind: str,
    wire: int,
    raw: int | bytes,
    schemas: dict[str, dict[int, _Field]],
    enums: dict[str, tuple[int, ...]],
    budget: _Budget,
    depth: int,
) -> Any:
    if kind in ("bool", "int32", "int64") or kind.startswith("enum:"):
        if wire != 0 or type(raw) is not int:
            raise _WireError("scalar-wire-type")
        if kind == "bool":
            if raw not in (0, 1):
                raise _WireError("boolean-domain")
            return bool(raw)
        value = _signed(raw, 64 if kind == "int64" else 32)
        if kind.startswith("enum:") and value not in enums[kind[5:]]:
            raise _WireError("unknown-enum-value:" + kind[5:])
        return value
    if kind in ("float", "double"):
        if wire != (5 if kind == "float" else 1) or type(raw) is not bytes:
            raise _WireError("floating-wire-type")
        value = struct.unpack("<f" if kind == "float" else "<d", raw)[0]
        if not math.isfinite(value):
            raise _WireError("nonfinite-option")
        return value
    if wire != 2 or type(raw) is not bytes:
        raise _WireError("message-or-string-wire-type")
    if kind == "string":
        return _text(raw)
    if kind == "bytes":
        return raw
    if kind.startswith("opaque:"):
        _wire(raw, budget)
        return raw  # Retained but unreviewed; its presence cannot earn support.
    message = _message(raw, kind, schemas, enums, budget, depth + 1)
    return tuple(sorted(message.items()))


def _message(
    payload: bytes,
    scope: str,
    schemas: dict[str, dict[int, _Field]],
    enums: dict[str, tuple[int, ...]],
    budget: _Budget,
    depth: int = 0,
) -> dict[int, Any]:
    if depth > 8:
        raise _WireError("message-depth-bound")
    result: dict[int, Any] = {}
    repeated: dict[int, list[object]] = {}
    map_keys: dict[int, set[object]] = {}
    for number, wire, raw in _wire(payload, budget):
        field = schemas[scope].get(number)
        if field is None:
            raise _Unreviewed("unknown-or-reserved-field:" + scope + "." + str(number))
        if field.kind in ("string_map", "override_map"):
            entry_kind = (
                "StringEntry" if field.kind == "string_map" else "OverrideEntry"
            )
            entry = _decode(entry_kind, wire, raw, schemas, enums, budget, depth)
            values = dict(entry)
            key = values.get(1, "")
            if key in map_keys.setdefault(number, set()):
                raise _WireError("duplicate-map-key")
            map_keys[number].add(key)
            repeated.setdefault(number, []).append(entry)
        elif field.repeated:
            numeric = field.kind in ("bool", "int32", "int64") or field.kind.startswith(
                "enum:"
            )
            if numeric and wire == 2 and type(raw) is bytes:
                pos = 0
                while pos < len(raw):
                    budget.consume()
                    integer, pos = _varint(raw, pos)
                    repeated.setdefault(number, []).append(
                        _decode(field.kind, 0, integer, schemas, enums, budget, depth)
                    )
                repeated.setdefault(number, [])
            else:
                repeated.setdefault(number, []).append(
                    _decode(field.kind, wire, raw, schemas, enums, budget, depth)
                )
        else:
            if number in result:
                raise _WireError(
                    "duplicate-singular-field:" + scope + "." + str(number)
                )
            result[number] = _decode(
                field.kind, wire, raw, schemas, enums, budget, depth
            )
    if scope == "OptionOverride" and len(result) > 1:
        raise _WireError("duplicate-oneof-value")
    result.update((number, tuple(values)) for number, values in repeated.items())
    return result


def validate_current_cast_options(payload: bytes, schema: str) -> CurrentCastOptions:
    """Check pinned semantic/default routes; return no provenance or authority.

    Presence is protobuf presence. Absent fields use protobuf zero values,
    never C++ factory defaults. Device/propagation/logging/config facts require
    later binding to the actual conversion-owned native result and client.
    """
    if type(payload) is not bytes or len(payload) > _LIMIT:
        return CurrentCastOptions(
            "FAILED", "", ("options-type-or-size-bound",), b"", "", ()
        )
    digest = hashlib.sha256(payload).hexdigest()
    if type(schema) is not str or schema not in ("0.10.2", "0.12.0.1"):
        return CurrentCastOptions(
            "UNVERIFIED", "", ("unreviewed-schema",), payload, digest, ()
        )
    schemas, enums = _schemas(schema)
    facts: list[OptionFact] = []
    reasons: list[str] = []
    try:
        options = _message(payload, "CompileOptions", schemas, enums, _Budget())
        build = dict(options.get(3) or ())
        debug = dict(build.get(3) or ())
        messages = {
            "CompileOptions": options,
            "ExecutableBuildOptions": build,
            "DebugOptions": debug,
        }
        for scope, fields in schemas.items():
            if scope not in messages:
                continue
            for number, field in sorted(fields.items()):
                facts.append(
                    OptionFact(
                        scope,
                        number,
                        number in messages[scope],
                        messages[scope].get(number, _default(field)),
                        field.classification,
                    )
                )
        if 3 not in options or 3 not in build:
            reasons.append("current-build-or-debug-message-absent")
        required = dict(_REQUIRED_DEBUG)
        required.update(_REQUIRED_MODERN if schema == "0.12.0.1" else _REQUIRED_OLD)
        requirements = {
            "CompileOptions": {2: False, 4: False, 5: -1, 6: b"", 9: False, 10: 0},
            "ExecutableBuildOptions": {
                1: -1,
                4: 1,
                5: 1,
                6: True,
                7: False,
                8: False,
                10: False,
                11: False,
                14: b"",
                15: 0,
                16: (),
                17: (),
                22: 0,
                23: 1,
                24: 0,
                25: 29,
            },
            "DebugOptions": required,
        }
        if schema == "0.10.2":
            requirements["ExecutableBuildOptions"].update({20: 0.0, 21: 0.0, 26: 0})
        else:
            requirements["CompileOptions"][12] = ()
        for scope, values in requirements.items():
            for number, expected in values.items():
                actual = messages[scope].get(number, _default(schemas[scope][number]))
                if type(actual) is not type(expected) or actual != expected:
                    reasons.append(
                        "unreviewed-current-option:" + scope + "." + str(number)
                    )
        for scope, numbers in {
            "CompileOptions": (1, 7, 8, 11),
            "ExecutableBuildOptions": (2, 13, 27) if schema == "0.12.0.1" else (2, 13),
            "DebugOptions": (30, 124, 500),
        }.items():
            for number in numbers:
                if number in messages[scope]:
                    reasons.append(
                        "unreviewed-present-option:" + scope + "." + str(number)
                    )
        if debug.get(395, 0) not in (0, 1):
            reasons.append("unreviewed-emitter-verification-level")
        for number in (12, 18):
            values = build.get(number, ())
            if type(values) is not tuple or len(values) != 1:
                reasons.append("unbound-single-port-propagation:" + str(number))
        assignment = dict(build.get(9) or ())
        devices = assignment.get(3, ())
        if assignment.get(1, 0) != 1 or assignment.get(2, 0) != 1 or len(devices) != 1:
            reasons.append("unbound-single-device-assignment")
        elif len(dict(devices[0]).get(1, ())) != 1 or dict(devices[0])[1][0] < 0:
            reasons.append("unbound-single-device-assignment")
    except _WireError as exc:
        return CurrentCastOptions(
            "FAILED", schema, (str(exc),), payload, digest, tuple(facts)
        )
    except _Unreviewed as exc:
        return CurrentCastOptions(
            "UNVERIFIED", schema, (str(exc),), payload, digest, tuple(facts)
        )
    return CurrentCastOptions(
        "UNVERIFIED" if reasons else "SUPPORTED_CURRENT_OPTIONS",
        schema,
        tuple(reasons),
        payload,
        digest,
        tuple(facts),
    )
