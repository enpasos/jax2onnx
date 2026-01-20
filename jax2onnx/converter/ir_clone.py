# jax2onnx/converter/ir_clone.py

# SPDX-License-Identifier: Apache-2.0
"""
Graph cloning logic.
"""

from __future__ import annotations

import onnx_ir as ir


def clone_graph(graph: ir.Graph) -> ir.Graph:
    """
    Create a detached copy of an ``onnx_ir.Graph``.

    This implementation constructs a new graph with cloned values and nodes,
    preserving metadata properties (type, shape, meta, metadata_props) which
    are critical for JAX tracing.
    """
    # Assuming onnx-ir implementation supports allow_outer_scope_values
    # and metadata preservation (via local patch).

    return graph.clone(allow_outer_scope_values=True)
