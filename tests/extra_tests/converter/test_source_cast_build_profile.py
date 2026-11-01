# tests/extra_tests/converter/test_source_cast_build_profile.py

"""Closed build/factory/map fences; synthetic rows never issue authority."""

from dataclasses import replace
import functools
import os
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

import jax
import jaxlib
import pytest

from jax2onnx._compat import _cast_build_profile as builds
from jax2onnx._compat import _cast_metadata as metadata
from jax2onnx._compat.jax import _CAST_CPU_IMPORT_ANCHORS


def _identity(size: int) -> tuple[int, ...]:
    return (1, 2, 0o100755, size, 3, 4)


@pytest.mark.parametrize("row", builds._PROFILES, ids=lambda row: row.cohort)
def test_four_exact_abi_rows_are_separate_pure_evidence(row: Any) -> None:
    assert (
        builds._match_profile(
            row.cache_tag,
            row.version,
            row.version,
            (_identity(3576), row.shim_sha256),
            (_identity(row.common_bytes), row.common_sha256),
        )
        is row
    )
    assert len(row.jax_commit) == len(row.xla_commit) == len(row.llvm_commit) == 40
    assert len(row.options_proof) == len(row.joined_proof) == 64
    assert row.platform == "linux-x86_64"
    assert not hasattr(row, "permit")
    assert len({entry.common_sha256 for entry in builds._PROFILES}) == 4


@pytest.mark.parametrize("row", builds._PROFILES, ids=lambda row: row.cohort)
@pytest.mark.parametrize(
    "fault",
    ["abi", "jax", "jaxlib", "shim", "shim-size", "common", "common-size", "mixed"],
)
def test_unknown_and_mixed_rows_fail_closed(row: Any, fault: str) -> None:
    tag, version, lib_version = row.cache_tag, row.version, row.version
    shim = (_identity(3576), row.shim_sha256)
    common = (_identity(row.common_bytes), row.common_sha256)
    if fault == "abi":
        tag += "t"
    elif fault == "jax":
        version = "0.12.0"
    elif fault == "jaxlib":
        lib_version = "0.12.0"
    elif fault == "shim":
        shim = (shim[0], "0" * 64)
    elif fault == "shim-size":
        shim = (_identity(3577), shim[1])
    elif fault == "common":
        common = (common[0], "0" * 64)
    elif fault == "common-size":
        common = (_identity(row.common_bytes + 1), common[1])
    else:
        other = next(
            entry
            for entry in builds._PROFILES
            if entry is not row and entry.version == row.version
        )
        common = (_identity(other.common_bytes), other.common_sha256)
    with pytest.raises(ValueError, match="build"):
        builds._match_profile(tag, version, lib_version, shim, common)


def _maps_fixture(
    tmp_path: Path,
) -> tuple[bytes, tuple[str, int, int], tuple[str, int, int]]:
    shim = tmp_path / "_jax.so"
    common = tmp_path / "libjax_common.so"
    shim.write_bytes(b"shim-fixture")
    common.write_bytes(b"common-fixture")
    identities = tuple(
        (str(path.resolve()), path.stat().st_dev, path.stat().st_ino)
        for path in (shim, common)
    )
    lines = []
    for index, (path, device, inode) in enumerate(identities, 1):
        lines.append(
            f"{index * 4096:x}-{(index + 1) * 4096:x} r-xp 00000000 {os.major(device):x}:{os.minor(device):x} {inode} {path}\n"
        )
    return "".join(lines).encode(), identities[0], identities[1]


def test_maps_match_selected_numeric_file_identity_and_ignore_unrelated_growth(
    tmp_path: Path,
) -> None:
    payload, shim, common = _maps_fixture(tmp_path)
    expected = builds._selected_mappings(payload, shim, common)
    extra = b"4000-5000 rw-p 00000000 00:00 0 [heap]\n"
    assert builds._selected_mappings(payload + extra, shim, common) == expected
    assert len(expected) == 2 and all(b"x" in row[2] for row in expected)


def test_reviewed_import_shim_requires_its_actual_executable_mapping(
    tmp_path: Path,
) -> None:
    payload, shim, common = _maps_fixture(tmp_path)
    payload = payload.replace(b"r-xp", b"r--p", 1)
    with pytest.raises(ValueError, match="executable"):
        builds._selected_mappings(payload, shim, common)


@pytest.mark.parametrize(
    "fault",
    [
        "deleted-shim",
        "deleted-common",
        "inode",
        "device",
        "no-execute",
        "malformed",
        "overlap",
        "truncated",
        "second-common",
        "too-large",
    ],
)
def test_maps_reject_deleted_malformed_ambiguous_and_other_common(
    tmp_path: Path, fault: str
) -> None:
    payload, shim, common = _maps_fixture(tmp_path)
    if fault == "deleted-shim":
        payload = payload.replace(
            shim[0].encode() + b"\n", shim[0].encode() + b" (deleted)\n"
        )
    elif fault == "deleted-common":
        payload = payload.replace(
            common[0].encode() + b"\n", common[0].encode() + b" (deleted)\n"
        )
    elif fault == "inode":
        common = (common[0], common[1], common[2] + 1)
    elif fault == "device":
        common = (
            common[0],
            os.makedev(os.major(common[1]), os.minor(common[1]) + 1),
            common[2],
        )
    elif fault == "no-execute":
        payload = payload.replace(b"r-xp", b"r--p")
    elif fault == "malformed":
        payload = payload.replace(b"r-xp", b"rwxq", 1)
    elif fault == "overlap":
        payload = payload.replace(b"2000-3000", b"1001-3000")
    elif fault == "truncated":
        payload = payload[:-1]
    elif fault == "second-common":
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        other = other_dir / "libjax_common.so"
        other.write_bytes(b"different")
        info = other.stat()
        payload += f"4000-5000 r-xp 00000000 {os.major(info.st_dev):x}:{os.minor(info.st_dev):x} {info.st_ino} {other}\n".encode()
    else:
        payload = b"x" * (builds._MAP_BYTES + 1)
    with pytest.raises(ValueError):
        builds._selected_mappings(payload, shim, common)


def _schema_row() -> Any:
    return next(row for row in builds._PROFILES if row.version == jax.__version__)


def test_current_standard_factory_is_pinned_without_constructing_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    from jax._src import xla_bridge

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("client")
        raise AssertionError("factory inspection constructed a client")

    monkeypatch.setattr(xla_bridge, "get_backend", forbidden)
    pins = metadata.OrdinaryCpuFactoryPins(_schema_row())
    try:
        pins.assert_current()
    finally:
        pins.close()
    assert not calls and not pins._states and not pins._members


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize(
    "fault",
    [
        "bridge-factory",
        "client-factory",
        "registration-factory",
        "registration-object",
        "native-factory",
        "native-module",
        "distributed",
        "callbacks",
        "same-function-code",
        "bridge-defaults",
        "client-defaults",
        "state-reader",
        "registration-reader",
    ],
)
def test_factory_tampering_never_calls_a_foreign_constructor(
    when: str, fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    native, bridge, client = _CAST_CPU_IMPORT_ANCHORS[:3]
    calls = []

    def foreign(*args: Any, **kwargs: Any) -> Any:
        calls.append("foreign")
        raise AssertionError("foreign CPU constructor invoked")

    pins = metadata.OrdinaryCpuFactoryPins(_schema_row()) if when == "after" else None
    function = bridge.make_cpu_client
    code = function.__code__
    try:
        with monkeypatch.context() as patch:
            if fault == "bridge-factory":
                patch.setattr(bridge, "make_cpu_client", foreign)
            elif fault == "client-factory":
                patch.setattr(client, "make_cpu_client", foreign)
            elif fault == "registration-factory":
                patch.setattr(bridge._backend_factories["cpu"], "factory", foreign)
            elif fault == "registration-object":
                patch.setitem(
                    bridge._backend_factories,
                    "cpu",
                    replace(bridge._backend_factories["cpu"]),
                )
            elif fault == "native-factory":
                patch.setattr(native, "get_tfrt_cpu_client", foreign)
            elif fault == "native-module":
                patch.setattr(client, "_xla", ModuleType("foreign"))
            elif fault == "distributed":
                patch.setattr(bridge.distributed.global_state, "process_id", 2)
            elif fault == "callbacks":
                patch.setattr(
                    bridge, "_backend_initialization_hooks", [foreign], raising=False
                )
            elif fault == "bridge-defaults":
                patch.setattr(bridge.make_cpu_client, "__defaults__", (object(),))
            elif fault == "client-defaults":
                patch.setattr(
                    client.make_cpu_client,
                    "__defaults__",
                    (False, None, 0, 1, None, None, None, None, None),
                )
            elif fault == "state-reader":
                patch.setattr(
                    type(bridge.distributed.global_state),
                    "__getattribute__",
                    foreign,
                    raising=False,
                )
            elif fault == "registration-reader":
                patch.setattr(
                    type(bridge._backend_factories["cpu"]),
                    "__getattribute__",
                    foreign,
                    raising=False,
                )
            else:
                function.__code__ = code.replace(
                    co_consts=(*code.co_consts, "unexecuted-factory-change")
                )
            with pytest.raises(ValueError):
                if pins is None:
                    metadata.OrdinaryCpuFactoryPins(_schema_row())
                else:
                    pins.assert_current()
    finally:
        function.__code__ = code
        if pins is not None:
            pins.close()
    assert not calls


@pytest.mark.parametrize(
    "fault", ["foreign-module", "file", "origin", "loader", "native-factory"]
)
def test_loaded_module_is_the_imported_standard_extension_before_any_file_claim(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _CAST_CPU_IMPORT_ANCHORS[0]
    with monkeypatch.context() as patch:
        if fault == "foreign-module":
            native = ModuleType("jaxlib._jax")
        elif fault == "file":
            patch.setattr(native, "__file__", "/not-the-imported-shim.so")
        elif fault == "origin":
            patch.setattr(native.__spec__, "origin", "/not-the-imported-shim.so")
        elif fault == "loader":
            patch.setattr(native.__spec__, "loader", object())
        else:
            patch.setattr(native, "get_tfrt_cpu_client", object())
        with pytest.raises((ValueError, OSError)):
            builds.LoadedCastBuildPins(native, jax.__version__, jaxlib.__version__)


def test_exact_current_build_holds_full_bytes_and_selected_maps_until_release() -> None:
    import sys

    row = next(
        (
            entry
            for entry in builds._PROFILES
            if entry.cache_tag == sys.implementation.cache_tag
            and entry.version == jax.__version__
        ),
        None,
    )
    if row is None:
        pytest.skip("outside the four reviewed build cohorts")
    if not row.provider_proof:
        with pytest.raises(ValueError, match="provider proof pending"):
            builds.LoadedCastBuildPins(
                _CAST_CPU_IMPORT_ANCHORS[0], jax.__version__, jaxlib.__version__
            )
        return
    pins = builds.LoadedCastBuildPins(
        _CAST_CPU_IMPORT_ANCHORS[0], jax.__version__, jaxlib.__version__
    )
    try:
        assert pins.profile is row and pins._maps
        pins.assert_current(full=True)
    finally:
        pins.close()
    assert pins._native is None and not pins._maps and not pins._files


@pytest.mark.parametrize("fault", ["selected-maps", "common-bytes", "maps-unavailable"])
def test_final_loaded_build_check_rejects_map_and_full_byte_changes(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    row = next(
        (
            entry
            for entry in builds._PROFILES
            if entry.cache_tag == sys.implementation.cache_tag
            and entry.version == jax.__version__
        ),
        None,
    )
    if row is None or not row.provider_proof:
        pytest.skip("exact installed provider row has not been reviewed")
    pins = builds.LoadedCastBuildPins(
        _CAST_CPU_IMPORT_ANCHORS[0], jax.__version__, jaxlib.__version__
    )
    actual_file = metadata._file
    try:
        with monkeypatch.context() as patch:
            if fault == "common-bytes":

                def changed_file(path: Path) -> Any:
                    identity, digest = actual_file(path)
                    return identity, "0" * 64 if path == pins._common else digest

                patch.setattr(metadata, "_file", changed_file)
            elif fault == "selected-maps":
                changed = (pins._maps[0][0] + 1, *pins._maps[0][1:])
                patch.setattr(
                    builds, "_read_maps", lambda *_: (changed, *pins._maps[1:])
                )
            else:

                def unavailable(*args: Any) -> Any:
                    raise OSError("proc maps unavailable")

                patch.setattr(builds, "_read_maps", unavailable)
            with pytest.raises((ValueError, OSError)):
                pins.assert_current(full=True)
    finally:
        pins.close()


@pytest.mark.parametrize("when", ["before", "after"])
def test_actual_source_epoch_rejects_old_hook_removal_or_modern_empty_injection(
    when: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bridge, _ = _CAST_CPU_IMPORT_ANCHORS[:3]
    row = _schema_row()
    pins = metadata.OrdinaryCpuFactoryPins(row) if when == "after" else None
    try:
        with monkeypatch.context() as patch:
            if row.version == "0.10.2":
                assert type(vars(bridge)["_backend_initialization_hooks"]) is list
                assert not vars(bridge)["_backend_initialization_hooks"]
                patch.delattr(bridge, "_backend_initialization_hooks")
            else:
                assert row.version == "0.12.0.1"
                assert "_backend_initialization_hooks" not in vars(bridge)
                patch.setattr(
                    bridge, "_backend_initialization_hooks", [], raising=False
                )
            with pytest.raises(ValueError, match="CPU initialization"):
                if pins is None:
                    metadata.OrdinaryCpuFactoryPins(row)
                else:
                    pins.assert_current()
    finally:
        if pins is not None:
            pins.close()


@pytest.mark.parametrize("selection", ["empty", "subset"])
def test_builtin_plugin_callback_import_order_subsets_need_no_registration(
    selection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bridge, _ = _CAST_CPU_IMPORT_ANCHORS[:3]
    callbacks = bridge._plugin_callbacks
    original = tuple(callbacks)
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("backend")
        raise AssertionError("callback verifier constructed a client")

    try:
        callbacks[:] = [] if selection == "empty" else original[:1]
        monkeypatch.setattr(bridge, "get_backend", forbidden)
        pins = metadata.OrdinaryCpuFactoryPins(_schema_row())
        try:
            assert pins._callback_registry is callbacks
            assert len(pins._callback_entries) == len(callbacks)
            pins.assert_current()
        finally:
            pins.close()
        assert not pins._callback_entries and not pins._callback_modules and not calls
    finally:
        callbacks[:] = original


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize(
    "fault", ["foreign", "foreign-partial", "batch-keywords", "partitioner-handler"]
)
def test_foreign_plugin_callback_actions_are_rejected_without_invocation(
    when: str,
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jax2onnx._compat.jax import _CAST_CPU_CALLBACK_ANCHORS

    _, bridge, _ = _CAST_CPU_IMPORT_ANCHORS[:3]
    callbacks = bridge._plugin_callbacks
    original = tuple(callbacks)
    calls = []

    def foreign(*args: Any, **kwargs: Any) -> Any:
        calls.append("foreign")
        raise AssertionError("foreign plugin callback invoked")

    pins = metadata.OrdinaryCpuFactoryPins(_schema_row()) if when == "after" else None
    try:
        if fault == "foreign":
            callback = foreign
        elif fault == "foreign-partial":
            callback = functools.partial(foreign, "untrusted")
        elif fault == "batch-keywords":
            callback = functools.partial(
                _CAST_CPU_CALLBACK_ANCHORS[0][1], "untrusted", extra=True
            )
        else:
            callback = functools.partial(
                _CAST_CPU_CALLBACK_ANCHORS[1][1],
                name="CustomSPMDPartitioning",
                prop_user_sharding=foreign,
                partition=foreign,
                infer_sharding_from_operands=foreign,
                can_side_effecting_have_replicated_sharding=True,
            )
        callbacks.append(callback)
        with pytest.raises(ValueError):
            if pins is None:
                metadata.OrdinaryCpuFactoryPins(_schema_row())
            else:
                pins.assert_current()
    finally:
        callbacks[:] = original
        if pins is not None:
            pins.close()
    assert not calls


@pytest.mark.parametrize(
    "fault", ["registry", "entry", "keywords", "source-member", "source-code"]
)
def test_captured_builtin_callback_objects_and_sources_cannot_change(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bridge, _ = _CAST_CPU_IMPORT_ANCHORS[:3]
    callbacks = bridge._plugin_callbacks
    assert callbacks  # Both actual supported ordinary imports register builtins.
    original = tuple(callbacks)
    pins = metadata.OrdinaryCpuFactoryPins(_schema_row())
    callback = callbacks[0]
    keywords = tuple(callback.keywords.items())
    source = next(iter(pins._callback_modules.values()))
    namespace, member, function = next(
        entry
        for entry in pins._members
        if entry[0] is vars(source) and type(entry[2]) is FunctionType
    )
    code = function.__code__
    calls = []

    def foreign(*args: Any, **kwargs: Any) -> Any:
        calls.append("foreign")
        raise AssertionError("replacement callback source invoked")

    try:
        with monkeypatch.context() as patch:
            if fault == "registry":
                patch.setattr(bridge, "_plugin_callbacks", list(callbacks))
            elif fault == "entry":
                callbacks[0] = functools.partial(
                    callback.func, *callback.args, **callback.keywords
                )
            elif fault == "keywords":
                callback.keywords["unexpected"] = True
            elif fault == "source-member":
                patch.setitem(namespace, member, foreign)
            else:
                function.__code__ = code.replace(
                    co_consts=(*code.co_consts, "unexecuted-callback-change")
                )
            with pytest.raises(ValueError):
                pins.assert_current()
    finally:
        function.__code__ = code
        callback.keywords.clear()
        callback.keywords.update(keywords)
        callbacks[:] = original
        pins.close()
    assert not calls


def test_absent_callback_source_owner_is_not_imported_by_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    pins = metadata.OrdinaryCpuFactoryPins(_schema_row())
    module_name = next(iter(pins._callback_modules))
    pins.close()
    with monkeypatch.context() as patch:
        patch.delitem(sys.modules, module_name)
        with pytest.raises(ValueError, match="source module unavailable"):
            metadata.OrdinaryCpuFactoryPins(_schema_row())
        assert module_name not in sys.modules


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("module", ["native", "client"])
def test_builtin_callback_native_import_bindings_cannot_be_replaced(
    when: str,
    module: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native, _, client = _CAST_CPU_IMPORT_ANCHORS[:3]
    pins = metadata.OrdinaryCpuFactoryPins(_schema_row()) if when == "after" else None
    calls = []

    def foreign(*args: Any, **kwargs: Any) -> Any:
        calls.append("foreign")
        raise AssertionError("foreign native callback action invoked")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                native if module == "native" else client,
                "register_custom_call_as_batch_partitionable",
                foreign,
            )
            with pytest.raises(ValueError):
                if pins is None:
                    metadata.OrdinaryCpuFactoryPins(_schema_row())
                else:
                    pins.assert_current()
    finally:
        if pins is not None:
            pins.close()
    assert not calls


# These byte/stream fixtures exercise parsing only, never loaded-build authority.
def _many_maps_fixture(tmp_path: Path, row_count: int, placement: str):
    payload, shim, common = _maps_fixture(tmp_path)
    selected = []
    for start, line in zip((0x10000000, 0x20000000), payload.splitlines(keepends=True)):
        selected.append(
            f"{start:x}-{start + 4096:x} ".encode() + line.split(b" ", 1)[1]
        )
    start = {"before": 0x10000, "between": 0x10010000, "after": 0x30000000}[placement]
    unrelated = b"".join(
        f"{start + index * 2:x}-{start + index * 2 + 1:x} rw-p 00000000 00:00 0 [heap]\n".encode()
        for index in range(row_count - 2)
    )
    baseline = b"".join(selected)
    if placement == "before":
        payload = unrelated + baseline
    elif placement == "between":
        payload = selected[0] + unrelated + selected[1]
    else:
        payload = baseline + unrelated
    return payload, baseline, shim, common


@pytest.mark.parametrize("row_count", [32769, 34436, 60000])
@pytest.mark.parametrize("placement", ["before", "between", "after"])
def test_maps_scan_all_unrelated_growth_without_changing_selected_evidence(
    tmp_path: Path, row_count: int, placement: str
) -> None:
    payload, baseline, shim, common = _many_maps_fixture(tmp_path, row_count, placement)
    assert payload.count(b"\n") == row_count
    if row_count == 60000:
        assert len(payload) > builds._MAP_BYTES
    assert builds._selected_mappings(
        payload, shim, common
    ) == builds._selected_mappings(baseline, shim, common)


@pytest.mark.parametrize(
    "fault",
    [
        "malformed",
        "overlap",
        "utf8",
        "foreign-common",
        "deleted-shim",
        "deleted-common",
        "inode",
        "device",
        "missing-shim-execute",
        "missing-common-execute",
        "truncated",
    ],
)
def test_maps_reject_late_faults_after_growth_and_both_selected_targets(
    tmp_path: Path, fault: str
) -> None:
    payload, _, shim, common = _many_maps_fixture(tmp_path, 60000, "before")
    assert len(payload) > builds._MAP_BYTES and payload.count(b"\n") == 60000
    suffix = b"40000000-40001000 rw-p 00000000 00:00 0 [heap]\n"
    if fault == "malformed":
        suffix = b"40000000-40001000 rwxq 00000000 00:00 0 [heap]\n"
    elif fault == "overlap":
        suffix = b"20000001-40001000 rw-p 00000000 00:00 0 [heap]\n"
    elif fault == "utf8":
        suffix = b"40000000-40001000 rw-p 00000000 00:00 0 /invalid-\xff\n"
    elif fault == "foreign-common":
        other = tmp_path / "other" / "libjax_common.so"
        other.parent.mkdir()
        other.write_bytes(b"not-native-evidence")
        value = other.stat()
        suffix = f"40000000-40001000 r-xp 0 {os.major(value.st_dev):x}:{os.minor(value.st_dev):x} {value.st_ino} {other}\n".encode()
    elif fault in {"deleted-shim", "deleted-common", "inode", "device"}:
        path, device, inode = shim if fault == "deleted-shim" else common
        if fault.startswith("deleted"):
            path += " (deleted)"
        elif fault == "inode":
            inode += 1
        else:
            device = os.makedev(os.major(device), os.minor(device) + 1)
        suffix = f"40000000-40001000 r-xp 0 {os.major(device):x}:{os.minor(device):x} {inode} {path}\n".encode()
    elif fault.startswith("missing"):
        target = shim[0] if fault == "missing-shim-execute" else common[0]
        rows = payload.splitlines(keepends=True)
        rows = [
            row.replace(b"r-xp", b"r--p") if target.encode() in row else row
            for row in rows
        ]
        payload = b"".join(rows)
    else:
        suffix = suffix[:-1]
    with pytest.raises((ValueError, UnicodeDecodeError)):
        builds._selected_mappings(payload + suffix, shim, common)


@pytest.mark.parametrize("separator", [b"\r\n", b"\r"])
def test_maps_reject_non_linux_carriage_return_separators(
    tmp_path: Path, separator: bytes
) -> None:
    payload, shim, common = _maps_fixture(tmp_path)
    # Preserve the final LF so this specifically witnesses the CR policy.
    payload = payload.replace(b"\n", separator, 1)
    with pytest.raises(ValueError, match="line separator"):
        builds._selected_mappings(payload, shim, common)


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_maps_bound_each_row_at_exact_original_byte_limit(
    tmp_path: Path, extra_bytes: int
) -> None:
    payload, shim, common = _maps_fixture(tmp_path)
    row = b"4000-5000 rw-p 0 00:00 0\n"
    row = row.replace(
        b" rw-p", b" " * (builds._MAP_BYTES - len(row) + 1 + extra_bytes) + b"rw-p", 1
    )
    assert len(row) == builds._MAP_BYTES + extra_bytes
    expected = builds._selected_mappings(payload, shim, common)
    if extra_bytes:
        with pytest.raises(ValueError, match="row size"):
            builds._selected_mappings(payload + row, shim, common)
    else:
        assert builds._selected_mappings(payload + row, shim, common) == expected


@pytest.mark.parametrize("selected_count", [2, 3])
def test_maps_bound_retained_selected_rows_with_reduced_local_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected_count: int
) -> None:
    payload, shim, common = _maps_fixture(tmp_path)
    monkeypatch.setattr(builds, "_MAP_SELECTED_ROWS", 2)
    if selected_count == 3:
        path, device, inode = common
        payload += f"4000-5000 r-xp 0 {os.major(device):x}:{os.minor(device):x} {inode} {path}\n".encode()
        with pytest.raises(ValueError, match="selected.*extent"):
            builds._selected_mappings(payload, shim, common)
    else:
        assert len(builds._selected_mappings(payload, shim, common)) == 2


@pytest.mark.parametrize("phase", ["before-read", "after-read", "at-eof"])
def test_maps_deadline_never_returns_partial_selected_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from types import SimpleNamespace

    payload, shim, common = _maps_fixture(tmp_path)
    ticks = {
        "before-read": [0, 10],
        "after-read": [0, 0, 10],
        "at-eof": [0, 0, 0, 0, 0, 0, 0, 10],
    }[phase]

    def clock() -> float:
        return ticks.pop(0) if ticks else 10

    monkeypatch.setattr(builds, "time", SimpleNamespace(monotonic=clock))
    with pytest.raises(ValueError, match="scan deadline"):
        builds._selected_mappings(payload, shim, common)
    assert not ticks


@pytest.mark.parametrize(
    "fault", ["none", "io-after-targets", "truncated", "non-bytes"]
)
def test_real_maps_reader_opens_once_and_requires_clean_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    import io

    payload, shim, common = _maps_fixture(tmp_path)
    expected = builds._selected_mappings(payload, shim, common)
    if fault == "truncated":
        payload += b"4000-5000 rw-p 0 00:00 0"
    source = io.BytesIO(payload)
    opened = []
    limits = []
    closed = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def readline(self, limit):
            limits.append(limit)
            line = source.readline(limit)
            if not line and fault == "io-after-targets":
                raise OSError("synthetic late maps IO failure")
            if not line and fault == "non-bytes":
                return None
            return line

    def open_maps(path, mode, *, buffering):
        opened.append((str(path), mode, buffering))
        return Stream()

    original_open = Path.open
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_maps)
        if fault == "none":
            assert builds._read_maps(shim, common) == expected
        else:
            with pytest.raises((ValueError, OSError)):
                builds._read_maps(shim, common)
    assert Path.open is original_open
    assert opened == [("/proc/self/maps", "rb", 64 * 1024)]
    assert limits and all(limit == builds._MAP_BYTES + 1 for limit in limits)
    assert closed == [True]
