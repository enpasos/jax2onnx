import pytest


@pytest.mark.xfail(
    reason="converter2 uses a different IR builder; legacy OnnxBuilder-specific test not ported"
)
def test_intermediate_not_promoted_converter2_placeholder():
    pytest.skip("placeholder for legacy converter OnnxBuilder behavior")
