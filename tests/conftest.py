import os
import pytest

# Fixture to ensure output directory exists
@pytest.fixture(scope="session", autouse=True)
def setup_output_dir():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
