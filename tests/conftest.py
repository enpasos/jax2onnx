import os

# Force deterministic CPU execution so numerical parity tests are stable
# regardless of whether CUDA is installed locally.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

