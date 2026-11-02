# tests/conftest.py

import os
import warnings
from pathlib import Path

import pytest

# Force deterministic CPU execution so numerical parity tests are stable
# regardless of whether CUDA is installed locally.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# JAX/XLA can emit an unraisable KeyboardInterrupt from its GC callback during
# interpreter shutdown even after a fully green test run. Keep the suite output
# clean by filtering just that teardown artifact.
warnings.filterwarnings(
    "ignore",
    message=r"Exception ignored in: <function _xla_gc_callback",
    category=pytest.PytestUnraisableExceptionWarning,
)


_SOURCE_METADATA_ACCEPTANCE = pytest.StashKey[dict]()


def pytest_addoption(parser):
    group = parser.getgroup("source metadata acceptance")
    group.addoption("--source-metadata-acceptance", metavar="REPORT")
    group.addoption("--source-metadata-baseline", metavar="PATH")
    group.addoption("--source-metadata-adapters", metavar="PATH")
    group.addoption("--source-metadata-scope", choices=("registered-predicate88",))
    group.addoption(
        "--source-metadata-case",
        action="append",
        default=[],
        metavar="origin::ordinal::scenario",
    )


def pytest_configure(config):
    report = config.getoption("source_metadata_acceptance")
    if report is None:
        if any(
            config.getoption(name)
            for name in (
                "source_metadata_baseline",
                "source_metadata_adapters",
                "source_metadata_case",
                "source_metadata_scope",
            )
        ):
            raise pytest.UsageError(
                "SOURCE metadata options require --source-metadata-acceptance"
            )
        return
    if not report:
        raise pytest.UsageError(
            "SOURCE metadata acceptance needs a nonempty report path"
        )
    if getattr(config.option, "numprocesses", None):
        raise pytest.UsageError(
            "SOURCE metadata acceptance requires one actual pytest session"
        )
    path = Path(report).resolve()
    if path.exists():
        raise pytest.UsageError("SOURCE metadata acceptance report already exists")
    owner = (
        Path(__file__).parent
        / "extra_tests/converter/test_source_precision_metadata.py"
    )
    requested_args = list(config.args)
    # Opt-in acceptance has one existing collection owner. Do not collect the
    # ordinary generated/model suite merely to deselect it afterwards.
    config.args[:] = [str(owner.resolve())]
    state = {
        "report": path,
        "reports": [],
        "items": [],
        "configured": False,
        "collection_reports": [],
        "collection_finished": False,
        "requested_pytest_args": requested_args,
        "collection_owner": str(owner.resolve()),
    }
    config.stash[_SOURCE_METADATA_ACCEPTANCE] = state
    from tests.extra_tests.converter.test_source_precision_metadata import (
        freeze_source_metadata_acceptance,
    )

    freeze_source_metadata_acceptance(state)


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makeitem(collector, name, obj):
    state = collector.config.stash.get(_SOURCE_METADATA_ACCEPTANCE, None)
    owner = (
        Path(__file__).parent
        / "extra_tests/converter/test_source_precision_metadata.py"
    )
    if (
        state is not None
        and name == "TestRegisteredSourceMetadata"
        and Path(str(collector.path)).resolve() == owner.resolve()
    ):
        try:
            collector.module.configure_source_metadata_acceptance(
                collector.config, obj, state
            )
        except Exception as exc:
            state["admission_error"] = f"{type(exc).__name__}: {exc}"
            raise


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    state = config.stash.get(_SOURCE_METADATA_ACCEPTANCE, None)
    if state is None:
        return
    # This is an execution boundary, not only a reporting filter. With no
    # explicit choices no metadata model, ordinary regression or other item runs.
    wanted = set(state.get("identities", ()))
    selected, excluded = [], []
    owner = (
        Path(__file__).parent
        / "extra_tests/converter/test_source_precision_metadata.py"
    )
    for item in items:
        descriptor = getattr(getattr(item, "obj", None), "_source_metadata_case", None)
        if (
            Path(str(item.path)).resolve() == owner.resolve()
            and descriptor is not None
            and descriptor.identity in wanted
        ):
            selected.append(item)
        else:
            excluded.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=excluded)
    state["items"] = list(selected)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    state = collector.config.stash.get(_SOURCE_METADATA_ACCEPTANCE, None)
    if state is not None:
        state["collection_reports"].append(outcome.get_result())


def pytest_collection_finish(session):
    state = session.config.stash.get(_SOURCE_METADATA_ACCEPTANCE, None)
    if state is not None:
        state["items"] = list(session.items)
        state["collection_finished"] = True


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    state = item.config.stash.get(_SOURCE_METADATA_ACCEPTANCE, None)
    if state is not None and item in state["items"]:
        state["reports"].append(outcome.get_result())


def pytest_sessionfinish(session, exitstatus):
    state = session.config.stash.get(_SOURCE_METADATA_ACCEPTANCE, None)
    if state is None:
        return
    from tests.extra_tests.converter.test_source_precision_metadata import (
        finish_source_metadata_acceptance,
    )

    finish_source_metadata_acceptance(session, state, exitstatus)
