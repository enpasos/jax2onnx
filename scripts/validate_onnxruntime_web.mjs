#!/usr/bin/env node
import fs from "node:fs/promises";
import process from "node:process";

import * as ort from "onnxruntime-web/wasm";

import {
  compareTensor,
  makeTensor,
} from "./onnxruntime_web_validation_utils.mjs";

function usage() {
  return "Usage: node scripts/validate_onnxruntime_web.mjs <spec.json>";
}

async function main() {
  const specPath = process.argv[2];
  if (!specPath) {
    throw new Error(usage());
  }

  const spec = JSON.parse(await fs.readFile(specPath, "utf8"));
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;

  const modelBytes = await fs.readFile(spec.modelPath);
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "disabled",
    enableMemPattern: false,
    executionMode: "sequential",
  });

  const feeds = Object.fromEntries(
    spec.inputs.map((inputSpec) => [inputSpec.name, makeTensor(ort, inputSpec)]),
  );
  const outputNames = spec.outputs.map((outputSpec) => outputSpec.name);
  const results = await session.run(feeds, outputNames);

  let maxDiff = 0;
  for (const outputSpec of spec.outputs) {
    const actual = results[outputSpec.name];
    if (!actual) {
      throw new Error(`Missing output '${outputSpec.name}' from onnxruntime-web.`);
    }
    maxDiff = Math.max(
      maxDiff,
      compareTensor(outputSpec.name, actual, outputSpec, spec.rtol, spec.atol),
    );
  }

  console.log(
    JSON.stringify({
      ok: true,
      outputs: spec.outputs.length,
      maxAbsDiff: maxDiff,
    }),
  );
}

main().catch((error) => {
  console.error(error?.stack || String(error));
  process.exit(1);
});
