# jax2onnx/_compat/_cast_build_profile.py

"""Closed installed-build catalog and already-loaded Linux library fence.

These private rows bind reviewed source/provider contracts to exact wheel bytes.
A matching version, adjacent file, caller digest or parsed maps fixture cannot
create authority. Unknown loaders, mappings and mixed ABI rows fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.machinery import ExtensionFileLoader, ModuleSpec
import os
from pathlib import Path
import re
import stat
import sys
import sysconfig
from types import ModuleType
from typing import Any, cast

_MAP_BYTES = 2 * 1024 * 1024
_MAP_ROWS = 32768
_JOINED_PROOF = "89a2b5941ff1cbc7c5e9c532f81a02a66fd5b3fc082fb899b2b0c345f42538ed"
_OPTIONS_PROOF = "d3e112ab9719944eb25278c13a8528817c792a66c9a20956c1c7852f6d52f4d8"


@dataclass(frozen=True, slots=True)
class CastBuildProfile:
    cohort: str
    cache_tag: str
    version: str
    shim_sha256: str
    common_bytes: int
    common_sha256: str
    jax_commit: str
    xla_commit: str
    llvm_commit: str
    schema_protos: tuple[str, str]
    factory_sources: tuple[str, str]
    provider_proof: str
    joined_proof: str = _JOINED_PROOF
    options_proof: str = _OPTIONS_PROOF
    platform: str = "linux-x86_64"
    source_stage_proof: str = (
        "28b1e378439018e48fae326001875f88221de4f6cc914bd9a52c82d08ad228a5"
    )
    llvm_target_proof: str = (
        "10431d31b20dabd8aa6e2f20a294beb559058eca439223cc939b17df0e3b55b3"
    )
    default_factory_proof: str = (
        "7ea7ffadb25be58330d22ee856fa6cab618b678757840c73f320ded38c292f6b"
    )


_OLD = (
    "990e6a0b84138346e6a38785412f36356e0e5dc3",
    "5a9e73cbd92530cac2ac36f4736a774b2412afe2",
    "35901313800ea6e6cbeb9226e51c7c4b29bfc40e",
    (
        "1bde4e04acb8dffc7cf7ee747c11564ff5370676cda67c0c53da7a125ac601a4",
        "a93c984938c5ab0ed442be9ae6995e2aad01667a6c8ee8c0ac7a81389d20fddd",
    ),
    (
        "44d7913d2fb8a2c9d703cf483825e47d64dea9dba375ec73dc8a3dccb929c8c0",
        "c955c14607d47de641c0d4d15587a47c242194b8e019dfe58b6a19293febdb5e",
    ),
)
_MODERN = (
    "dc4af684446e65ee98eee8090e1ff6f8211473fb",
    "d867969800d4a8fd045505eabc65e1e66f30794f",
    "2b35fb9283a44aae92cb72a41d077ee8de2be7ad",
    (
        "16dd4a448ff97cc155624404d2a99994dcdad25b4fc273ab9e7dc5a95eb7fc43",
        "2d6bb017503b74b207f23d7c9de2c3793aa19a7e31f43b507b6d6b87f2c196e1",
    ),
    (
        "1247ccda4a16eec80a1820b8a70bd2293cc1a6f7553c63a7c09693cccb4371af",
        "167bfa76b266f477f3ea9af9afcb2fa8ec595b8ddd21d2e037e610bd1d6f658d",
    ),
)
# Provider closure identifiers are source-reviewed catalog data. Runtime never
# reads external proof files or accepts registration of additional build rows.
_PROFILES = (
    CastBuildProfile(
        "python311",
        "cpython-311",
        "0.10.2",
        "1b8ba599253bddc480606ab30d6a47731ac9fe3be4e6082dec7a85e82778857f",
        349673696,
        "2e0ff3956f5d8e24368171568cc9f8a2465bd875f33ff427ce2b418fbb735f85",
        *_OLD,
        "2a8c3c182845d280d00934de06a6fe668f30ee1c76af2598f78278be3db4472c",
    ),
    CastBuildProfile(
        "python312",
        "cpython-312",
        "0.10.2",
        "1b8ba599253bddc480606ab30d6a47731ac9fe3be4e6082dec7a85e82778857f",
        349823440,
        "e33404479113ee4ea958abb922d86705e8caec6c224867b0ab477b62ba61d2cc",
        *_OLD,
        "2a8c3c182845d280d00934de06a6fe668f30ee1c76af2598f78278be3db4472c",
    ),
    CastBuildProfile(
        "python313",
        "cpython-313",
        "0.12.0.1",
        "6a28f034febe24e313cd12b45a97edeed3e226fa23383effa9d1fe7937beaa83",
        328518520,
        "8df619276e027fd2449f2f8865d4daef00942bf3cff79462846d02864cafbedb",
        *_MODERN,
        "7cf41b589727320f9a18cc233c9ff58c9e5760fa2a846b917c080615252faeb4",
    ),
    CastBuildProfile(
        "python314",
        "cpython-314",
        "0.12.0.1",
        "6a28f034febe24e313cd12b45a97edeed3e226fa23383effa9d1fe7937beaa83",
        328513552,
        "9fac43fccaa169228d89ae49cf128ec31cfe28a9544208ff40f092cd045cb26e",
        *_MODERN,
        "4c16d9d23835a6ef7850bbbcbc7a1d57afa33db89ae4406c7ba241559be3a53d",
    ),
)


def _match_profile(
    cache_tag: str,
    jax_version: str,
    jaxlib_version: str,
    shim: tuple[tuple[int, ...], str],
    common: tuple[tuple[int, ...], str],
) -> CastBuildProfile:
    """Pure row comparison; not an installed-build or authority constructor."""
    matches = tuple(
        row
        for row in _PROFILES
        if row.cache_tag == cache_tag
        and row.version == jax_version == jaxlib_version
        and shim[0][3] == 3576
        and shim[1] == row.shim_sha256
        and common[0][3] == row.common_bytes
        and common[1] == row.common_sha256
    )
    if len(matches) != 1:
        raise ValueError("unreviewed or mixed CPU cast build")
    return matches[0]


def _selected_mappings(
    payload: bytes,
    shim: tuple[str, int, int],
    common: tuple[str, int, int],
) -> tuple[tuple[Any, ...], ...]:
    """Bounded maps grammar and selected path/device/inode evidence only."""
    if (
        type(payload) is not bytes
        or len(payload) > _MAP_BYTES
        or not payload.endswith(b"\n")
    ):
        raise ValueError("loaded CPU maps size or termination differs")
    lines = payload.splitlines()
    if not lines or len(lines) > _MAP_ROWS:
        raise ValueError("loaded CPU maps extent differs")
    selected: list[tuple[Any, ...]] = []
    common_files: set[tuple[str, int, int]] = set()
    previous = 0
    for line in lines:
        parts = line.split(None, 5)
        if len(parts) < 5:
            raise ValueError("malformed loaded CPU map")
        address, permissions, offset, device, inode = parts[:5]
        if (
            re.fullmatch(rb"[0-9a-f]+-[0-9a-f]+", address) is None
            or re.fullmatch(rb"[r-][w-][x-][ps]", permissions) is None
            or re.fullmatch(rb"[0-9a-f]+", offset) is None
            or re.fullmatch(rb"[0-9a-f]+:[0-9a-f]+", device) is None
            or re.fullmatch(rb"[0-9]+", inode) is None
        ):
            raise ValueError("malformed loaded CPU map fields")
        start, end = (int(value, 16) for value in address.split(b"-"))
        if start < previous or start >= end:
            raise ValueError("ambiguous loaded CPU map ranges")
        previous = end
        if len(parts) == 5:
            continue
        path = parts[5].decode("utf-8", errors="strict")
        is_common = Path(path.removesuffix(" (deleted)")).name == "libjax_common.so"
        if (
            path.removesuffix(" (deleted)") not in (shim[0], common[0])
            and not is_common
        ):
            continue
        if not path.startswith("/") or "\\" in path or path.endswith(" (deleted)"):
            raise ValueError("unsupported loaded CPU pathname")
        if str(Path(path).resolve(strict=True)) != path:
            raise ValueError("noncanonical loaded CPU pathname")
        major, minor = (int(value, 16) for value in device.split(b":"))
        identity = (path, os.makedev(major, minor), int(inode))
        if is_common:
            common_files.add(identity)
        expected = common if is_common else shim
        if identity != expected or identity[2] <= 0:
            raise ValueError("loaded CPU path/device/inode differs")
        selected.append((start, end, permissions, int(offset, 16), *identity))
    if (
        common_files != {common}
        or not any(row[4:] == shim and b"x" in row[2] for row in selected)
        or not any(row[4:] == common and b"x" in row[2] for row in selected)
    ):
        raise ValueError("executable loaded CPU shim/common evidence unavailable")
    return tuple(selected)


def _read_maps(
    shim: tuple[str, int, int], common: tuple[str, int, int]
) -> tuple[tuple[Any, ...], ...]:
    with Path("/proc/self/maps").open("rb") as stream:
        payload = stream.read(_MAP_BYTES + 1)
    return _selected_mappings(payload, shim, common)


class LoadedCastBuildPins:
    """Actual imported extension, full wheel bytes and mapped-object identity."""

    def __init__(
        self, native: ModuleType, jax_version: str, jaxlib_version: str
    ) -> None:
        from .jax import _CAST_CPU_IMPORT_ANCHORS
        from ._cast_metadata import _file, _require

        _require(
            sys.platform == "linux" and os.uname().machine == "x86_64",
            "unsupported CPU cast platform",
        )
        _require(
            not any(os.environ.get(name) for name in ("LD_PRELOAD", "LD_AUDIT")),
            "unreviewed CPU cast loader interposition",
        )
        anchored = _CAST_CPU_IMPORT_ANCHORS
        _require(
            type(native) is ModuleType and native is anchored[0],
            "native CPU module identity differs",
        )
        self._native: Any = native
        self._anchors = anchored
        self._namespace: Any = vars(native)
        self._check_module()
        filename = self._namespace["__file__"]
        self._shim = Path(filename).resolve(strict=True)
        self._common = self._shim.parent / "libjax_common.so"
        self._files: tuple[Any, ...] = (_file(self._shim), _file(self._common))
        self.profile = _match_profile(
            sys.implementation.cache_tag, jax_version, jaxlib_version, *self._files
        )
        _require(
            sys.implementation.name == "cpython"
            and sys.byteorder == "little"
            and sysconfig.get_config_var("SOABI")
            == self.profile.cache_tag + "-x86_64-linux-gnu",
            "unreviewed CPU cast Python ABI",
        )
        _require(
            bool(self.profile.provider_proof),
            "installed CPU cast provider proof pending",
        )
        self._mapped_files = tuple(
            (str(path), identity[0][0], identity[0][1])
            for path, identity in zip(
                (self._shim, self._common), self._files, strict=True
            )
        )
        self._maps = _read_maps(*self._mapped_files)
        self._closed = False

    def _check_module(self) -> None:
        from ._cast_metadata import _require

        native, anchors = self._native, self._anchors
        namespace = self._namespace
        _require(
            vars(native) is namespace and sys.modules.get("jaxlib._jax") is native,
            "native CPU module namespace changed",
        )
        spec = namespace.get("__spec__")
        _require(
            namespace.get("__file__") == anchors[11]
            and spec is anchors[12]
            and type(spec) is ModuleSpec
            and spec.origin == anchors[13]
            and spec.loader is anchors[14]
            and type(spec.loader) is ExtensionFileLoader
            and spec.name == "jaxlib._jax"
            and type(namespace.get("__file__")) is str
            and Path(namespace["__file__"]).resolve(strict=True)
            == Path(cast(str, spec.origin)).resolve(strict=True)
            and namespace.get("get_tfrt_cpu_client") is anchors[8],
            "native CPU loader/origin/factory differs",
        )

    def assert_current(self, *, full: bool = False) -> None:
        from ._cast_metadata import _file, _require

        _require(not self._closed, "loaded CPU cast pins closed")
        self._check_module()
        for path, expected in zip((self._shim, self._common), self._files, strict=True):
            value = path.stat()
            identity = tuple(
                getattr(value, key)
                for key in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            )
            _require(
                stat.S_ISREG(value.st_mode) and identity == expected[0],
                "loaded CPU cast file identity changed",
            )
            if full:
                _require(_file(path) == expected, "loaded CPU cast file bytes changed")
        if full:
            _require(
                _read_maps(*self._mapped_files) == self._maps,
                "selected loaded CPU mappings changed",
            )

    def close(self) -> None:
        self._native = self._namespace = None
        self._anchors = ()
        self._files = self._maps = self._mapped_files = ()
        self._closed = True
