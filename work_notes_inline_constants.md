# Inline Constant Broadcast Cleanup

- Issue observed in Equinox Linear: scalar initializers (bias) are broadcast via `Concat+Expand` during lowering.
- Desired invariant: constants that require only shape changes should be reshaped at creation time and inlined as initializers (no runtime nodes).
- Potential general approach: detect initializer-only value feeds into unary/broadcast ops (`Expand`, `Reshape`, etc.) and fold them eagerly before serialization.
- Investigate whether other plugins exhibit similar patterns (e.g., NNX modules, Dropout scaling, attention masks).
- Consider adding a converter-level guard/test that fails when an initializer flows through a pure-shape op before consumption.  This has high priority. It should be a central general test. Whereas later fixes should happen at the causal root in each plugin.
