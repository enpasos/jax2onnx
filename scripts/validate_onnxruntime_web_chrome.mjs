#!/usr/bin/env node
import fs from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

function usage() {
  return "Usage: node scripts/validate_onnxruntime_web_chrome.mjs <spec.json>";
}

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");
const ORT_DIST_DIR = path.resolve(
  REPO_ROOT,
  "node_modules",
  "onnxruntime-web",
  "dist",
);
const VALIDATION_UTILS_PATH = path.resolve(
  SCRIPT_DIR,
  "onnxruntime_web_validation_utils.mjs",
);

const SHARED_HEADERS = {
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Resource-Policy": "same-origin",
};

const HTML = `<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>jax2onnx onnxruntime-web Chrome validation</title>
  </head>
  <body>
    <script type="module">
      import * as ort from "/ort/ort.wasm.min.mjs";
      import { compareTensor, makeTensor } from "/scripts/onnxruntime_web_validation_utils.mjs";

      window.__runOrtWebValidation = async () => {
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = false;
        ort.env.wasm.wasmPaths = "/ort/";

        const specResponse = await fetch("/spec.json");
        if (!specResponse.ok) {
          throw new Error("Failed to fetch validation spec: " + specResponse.status);
        }
        const spec = await specResponse.json();

        const modelResponse = await fetch("/model.onnx");
        if (!modelResponse.ok) {
          throw new Error("Failed to fetch model bytes: " + modelResponse.status);
        }
        const modelBytes = new Uint8Array(await modelResponse.arrayBuffer());
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
            throw new Error("Missing output '" + outputSpec.name + "' from onnxruntime-web.");
          }
          maxDiff = Math.max(
            maxDiff,
            compareTensor(outputSpec.name, actual, outputSpec, spec.rtol, spec.atol),
          );
        }

        return {
          ok: true,
          outputs: spec.outputs.length,
          maxAbsDiff: maxDiff,
        };
      };

      window.__ortWebValidationReady = true;
    </script>
  </body>
</html>`;

function contentTypeFor(filePath) {
  if (filePath.endsWith(".html")) {
    return "text/html; charset=utf-8";
  }
  if (filePath.endsWith(".mjs") || filePath.endsWith(".js")) {
    return "text/javascript; charset=utf-8";
  }
  if (filePath.endsWith(".wasm")) {
    return "application/wasm";
  }
  if (filePath.endsWith(".json") || filePath.endsWith(".map")) {
    return "application/json; charset=utf-8";
  }
  return "application/octet-stream";
}

function sendText(response, statusCode, body, contentType = "text/plain") {
  response.writeHead(statusCode, {
    ...SHARED_HEADERS,
    "Content-Type": contentType,
  });
  response.end(body);
}

async function sendFile(response, filePath) {
  const data = await fs.readFile(filePath);
  response.writeHead(200, {
    ...SHARED_HEADERS,
    "Content-Type": contentTypeFor(filePath),
  });
  response.end(data);
}

function resolveChild(root, relPath) {
  const resolved = path.resolve(root, relPath);
  const rootWithSep = root.endsWith(path.sep) ? root : `${root}${path.sep}`;
  if (resolved !== root && !resolved.startsWith(rootWithSep)) {
    throw new Error(`Path escapes server root: ${relPath}`);
  }
  return resolved;
}

function createValidationServer(spec, specPath) {
  const modelPath = path.resolve(spec.modelPath);
  const validationSpecPath = path.resolve(specPath);

  return http.createServer(async (request, response) => {
    try {
      if (!request.url) {
        sendText(response, 400, "Missing URL");
        return;
      }

      const requestUrl = new URL(request.url, "http://127.0.0.1");
      const route = decodeURIComponent(requestUrl.pathname);

      if (route === "/") {
        sendText(response, 200, HTML, "text/html; charset=utf-8");
        return;
      }

      if (route === "/model.onnx") {
        await sendFile(response, modelPath);
        return;
      }

      if (route === "/spec.json") {
        await sendFile(response, validationSpecPath);
        return;
      }

      if (route === "/scripts/onnxruntime_web_validation_utils.mjs") {
        await sendFile(response, VALIDATION_UTILS_PATH);
        return;
      }

      if (route.startsWith("/ort/")) {
        const relPath = route.slice("/ort/".length);
        await sendFile(response, resolveChild(ORT_DIST_DIR, relPath));
        return;
      }

      sendText(response, 404, `Not found: ${route}`);
    } catch (error) {
      sendText(response, 500, error?.stack || String(error));
    }
  });
}

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        reject(new Error("Failed to allocate validation server port."));
        return;
      }
      resolve(address.port);
    });
  });
}

function closeServer(server) {
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
}

function launchOptionsFromEnv() {
  const browserName = (
    process.env.JAX2ONNX_ONNXRUNTIME_WEB_BROWSER || "chromium"
  )
    .trim()
    .toLowerCase();
  const options = {
    headless: true,
    args: ["--disable-dev-shm-usage"],
  };

  if (browserName === "chromium") {
    return { browserName, options };
  }
  if (browserName === "chrome" || browserName === "msedge") {
    return { browserName, options: { ...options, channel: browserName } };
  }
  throw new Error(
    "Unsupported JAX2ONNX_ONNXRUNTIME_WEB_BROWSER value " +
      `'${browserName}'. Use 'chromium', 'chrome', or 'msedge'.`,
  );
}

async function main() {
  const specPath = process.argv[2];
  if (!specPath) {
    throw new Error(usage());
  }

  const spec = JSON.parse(await fs.readFile(specPath, "utf8"));
  const server = createValidationServer(spec, specPath);
  const pageErrors = [];
  let browser;
  let serverStarted = false;

  try {
    const port = await listen(server);
    serverStarted = true;
    const { browserName, options } = launchOptionsFromEnv();
    browser = await chromium.launch(options);
    const page = await browser.newPage();
    page.on("console", (message) => {
      if (message.type() === "error") {
        pageErrors.push(message.text());
      }
    });
    page.on("pageerror", (error) => {
      pageErrors.push(error?.stack || String(error));
    });

    await page.goto(`http://127.0.0.1:${port}/`);
    await page.waitForFunction("window.__ortWebValidationReady === true", null, {
      timeout: 15000,
    });
    const summary = await page.evaluate(async () => window.__runOrtWebValidation());

    console.log(
      JSON.stringify({
        ...summary,
        runner: "chrome",
        browser: browserName,
      }),
    );
  } catch (error) {
    if (pageErrors.length > 0) {
      console.error(pageErrors.join("\n"));
    }
    throw error;
  } finally {
    if (browser) {
      await browser.close();
    }
    if (serverStarted) {
      await closeServer(server);
    }
  }
}

main().catch((error) => {
  console.error(error?.stack || String(error));
  process.exit(1);
});
