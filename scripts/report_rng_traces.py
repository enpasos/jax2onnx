#!/usr/bin/env python
# scripts/report_rng_traces.py

from __future__ import annotations

from jax2onnx.plugins import plugin_system as ps


def main() -> None:
    ps.import_all_plugins()
    traces = ps.list_registered_rng_traces()
    if not traces:
        print("No RNG factories registered.")
        return
    print("Registered RNG factories:")
    for trace in traces:
        print(f"  - {trace}")


if __name__ == "__main__":
    main()
