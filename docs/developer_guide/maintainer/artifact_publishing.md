# Artifact Publishing

This page is maintainer-facing. Generated ONNX models are build outputs, not
source documentation. The public docs may point at hosted sample models, but the
models themselves are not part of the `jax2onnx` source tree.

## Policy

- Do not commit `.onnx` or `.onnx.data` files, including files under
  `docs/onnx/`.
- Keep generated models local while developing and validating.
- Public docs may link to stable hosted sample models through generated Netron
  URLs.
- Do not describe upload mechanics, credentials, or release-local promotion
  checklists in user-facing docs.

## Public Links

The generated reference tables use Netron links whose model URLs point at the
hosted sample-model repository. The current base URL lives in
`scripts/generate_readme.py` as `NETRON_BASE_URL`:

```text
https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/
```

When publishing or replacing a sample model, keep the hosted path aligned with
the generated testcase path. Generated tests write local models under
`onnx/<context>/<testcase>.onnx`, where dots in the testcase context become path
separators. `scripts/generate_readme.py` builds the public link from the same
`<context>/<testcase>.onnx` suffix.

```text
<context>/<testcase>.onnx
```

Examples:

```text
primitives/jnp/jnp_sin_basic.onnx
examples/maxtext/maxtext_llama2_7b.onnx
```

## External Data Sidecars

`to_onnx(..., output_path=...)` serializes ONNX protobufs with external-data
support. Initializers above the configured threshold may produce a sidecar named
`<model>.onnx.data` next to the `.onnx` file. If that sidecar exists, the hosted
sample is incomplete unless both files are present at matching relative paths.

Before publishing a model, inspect the local output directory:

```bash
find onnx -name '<testcase>.onnx*' -print
```

The helper `scripts/sync_onnx_models.sh` targets the sibling sample-model repo
and currently copies only `.onnx` files. Use it only for models that do not need
external sidecars, or make sure matching `.onnx.data` files are present in the
sample-model repo before relying on the public Netron link.

## Validation Before Publishing

1. Generate the model locally from the matching plugin/example test.
2. Run the focused pytest target that validates the export.
3. Check whether the export produced a matching `.onnx.data` sidecar.
4. Open the local model in Netron when graph readability or external data
   placement changed.
5. Publish the `.onnx` and matching `.onnx.data` file together when external
   data is used.
6. Verify the hosted Netron URL after publishing.
7. Regenerate the public reference page if testcase names, contexts, or support
   status changed.

After publishing, run `poetry run mkdocs build --strict` to catch broken local
links or navigation issues.
