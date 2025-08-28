# Flow: broadcast_in_dim with dynamic B

```mermaid
sequenceDiagram
  autonumber
  participant converter2 as IR Converter (converter2.conversion_api)
  participant ir_context as IRContext (converter2.ir_context)
  participant plugins2 as Plugins v2 set
  converter2->>ir_context: add_input_for_invar records origin: B → (in0, axis=0)
  plugins2->>ir_context: get_value_for_var(x); build target shape via Shape(in0) → Gather([0]) + const [1] + ... → Concat(axis=0)
  plugins2->>ir_context: emit Reshape/Expand nodes; CastLike if needed
  converter2->>ir_context: add_outputs_from_vars; to_model_proto
```
