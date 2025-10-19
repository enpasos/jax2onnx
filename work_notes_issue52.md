## Issue 52 – Scatter Window Broadcast

- Goal: eliminate the scatter-window broadcast mismatch so the exported ONNX model runs in onnxruntime; once that holds, we can un-xfail `tests/extra_tests/loop/test_loop_scatter_payload_regression.py` and restore end-to-end parity.
- Introduced `_axis0_utils` plus loop metadata so slices, squeezes, and elementwise ops re-expand scalars to the loop extent and carry overrides through nested scans.
- Reworked `broadcast_in_dim`/scan lowering to favour recorded loop hints, restamp Expand outputs, and added xfail regression scaffolding; sandbox still hits a late ORT merge (node_Mul_106), so the remaining work is to keep those overrides alive through the loosening pass.
