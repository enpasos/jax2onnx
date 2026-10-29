# tests/extra_tests/converter/test_source_original22_nnx_preflight.py

"""Original factories survive state preparation without another model call.

This is preparation/reconstruction evidence only. The unchanged numerical,
export, runtime and resource obligations remain in the full original22 owner.
Its static SpectralNorm route does not claim mutable-state preparation support.
"""

from typing import Any

import jax
import numpy as np
import pytest

from jax2onnx._compat.jax import native_transformed_execution_context
from tests.extra_tests.converter.source_eqx18_default_anchor import Eqx18DefaultAnchor
from tests.extra_tests.converter.source_precision_support import (
    SourceProvenance,
    raw_source_case,
    raw_source_scenarios,
    source_configuration,
)
from tests.extra_tests.converter.test_source_fixed_parameter_models import _CONTRACTS
from tests.extra_tests.converter.test_source_mlp_runtime_acceptance import (
    _runtime_configuration,
)
from tests.extra_tests.converter.test_source_original22_state_preparation import (
    _bound_original,
    _contract,
    _identities,
    _module,
    _prepare,
    _snapshot,
)


@pytest.mark.parametrize("index", range(len(_CONTRACTS)))
@pytest.mark.parametrize("caller_x64", (False, True), ids=("raw-x64-off", "raw-x64-on"))
def test_original22_factories_preserve_preparation_and_reconstruction(
    index: int, caller_x64: bool, record_property: Any
) -> None:
    origin, ordinal, digest = _CONTRACTS[index]
    case = raw_source_case(origin, ordinal)
    anchor = (
        Eqx18DefaultAnchor.capture(case, origin, ordinal, digest)
        if index == 18
        else None
    )
    assert _contract(case, default_anchor=anchor) == digest
    scenario = raw_source_scenarios(case, SourceProvenance(origin, ordinal))[
        int(caller_x64)
    ]
    assert scenario.input_specs is not None and len(scenario.input_specs) == 1
    assert scenario.criteria is None and scenario.require_reviewed_criteria
    with (
        _runtime_configuration(caller_x64),
        source_configuration(scenario),
        native_transformed_execution_context(),
    ):
        key = jax.random.key(20261021, impl="threefry2x32") if index == 18 else None
        target = _bound_original(case, index, key, default_anchor=anchor)
        baseline, identities = _snapshot(target), _identities(target)
        fixed_key = getattr(target, "_rngs", None)
        if index == 20:
            # The original full owner exports a static update_stats=False
            # partial. Do not relabel it as an admitted mutable-state adapter.
            route = "unchanged static original; no state adapter"
        else:
            prepared = _prepare(index, target)
            initial = prepared.initial_words
            assert (
                len(initial) == len(prepared.state_specs) == len(prepared.state_names)
            )
            clone = prepared.fresh_original()
            assert _snapshot(clone) == baseline
            if _module(target) is not None:
                assert _module(clone) is not _module(target)
            observed = prepared.words_from_target(target)
            assert len(observed) == len(initial)
            for actual, expected in zip(observed, initial, strict=True):
                assert actual.dtype == expected.dtype and actual.shape == expected.shape
                np.testing.assert_array_equal(
                    np.asarray(actual), np.asarray(expected), strict=True
                )
            route = "prepared and reconstructed without another model call"
        assert _snapshot(target) == baseline
        assert _identities(target) == identities
        assert getattr(target, "_rngs", None) is fixed_key
        assert _contract(case, default_anchor=anchor) == digest
        if anchor is not None:
            anchor.validate(case)
        record_property("original_contract_sha256", digest)
        record_property("preparation_route", route)
