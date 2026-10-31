# internal_docs/precision/poc/migration_ux.py

"""Runnable compatibility entry point for the private precision UX experiment.

The shared implementation lives with converter test support, allowing normal
CI regressions to use it without importing documentation modules.
"""

from tests.extra_tests.converter.precision_ux_support import (
    OMITTED as OMITTED,
    CallReport as CallReport,
    CapturedWarning as CapturedWarning,
    ExportResult as ExportResult,
    FailureCategory as FailureCategory,
    Input as Input,
    InputSummary as InputSummary,
    MigrationError as MigrationError,
    Route as Route,
    _run_cpu as _run_cpu,
    compact_report as compact_report,
    export_candidate as export_candidate,
    main as main,
)

if __name__ == "__main__":
    main()
