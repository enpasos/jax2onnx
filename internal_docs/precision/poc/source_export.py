# internal_docs/precision/poc/source_export.py

"""Compatibility imports for the runnable source-precision experiments.

The shared bounded harness lives beside the converter regressions so normal
CI exercises the same implementation used by these checkout-only experiments.
This module does not define a public export API.
"""

from tests.extra_tests.converter.precision_test_support import (
    PrototypeLimit as PrototypeLimit,
    RepresentationLimit as RepresentationLimit,
    export_source_dtypes as export_source_dtypes,
)
