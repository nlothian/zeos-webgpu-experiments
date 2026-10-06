// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The model's tokenizer with no model behind it, for the CPython benchmarks.
 *
 * Tokenizes exactly as `OptZeosWorker.tokenize` (`encodePlain`) and decodes pieces exactly
 * as `OptZeosWorker.piece`, so token counts and the grammar mask's vocabulary are the
 * ones the page sees, without loading 2.4 GB of weights. One JSON request per stdin line,
 * one JSON reply per stdout line:
 *
 *   {"op": "info"}            -> the export's meta.json fields JsMachine reads
 *   {"op": "pieces"}          -> every piece below vocabSize
 *   {"op": "tokenize", "text"} -> ids
 *
 *   node bench/tokenizer_server.mjs --model DIR
 */

import { readFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const COOP = fileURLToPath(new URL("../../coop-count-web/", import.meta.url));
const { values } = parseArgs({
  options: { model: { type: "string", default: join(COOP, "models", "Qwen3.5-4B-ZEOS-OPT") } },
});

const { Tokenizer } = await import(
  pathToFileURL(join(COOP, "node_modules", "@huggingface", "tokenizers", "dist", "tokenizers.mjs")).href
);
const { encodePlain } = await import(pathToFileURL(join(COOP, "web", "transformers_worker.js")).href);
const json = async (name) => JSON.parse(await readFile(join(values.model, name), "utf8"));
const meta = await json("meta.json");
const tokenizer = new Tokenizer(await json("tokenizer.json"), await json("tokenizer_config.json"));

function serve(request) {
  if (request.op === "info") {
    return {
      blockSize: meta.blockSize,
      padId: meta.padId,
      controlIds: meta.controlIds,
      eosId: meta.eosId,
      vocabSize: meta.vocabSize,
    };
  }
  if (request.op === "pieces") {
    const out = [];
    for (let id = 0; id < meta.vocabSize; id++) out.push(tokenizer.decode([id], { skip_special_tokens: false }));
    return out;
  }
  if (request.op === "tokenize") return Array.from(encodePlain(tokenizer, request.text));
  throw new Error(`unknown op ${request.op}`);
}

const lines = createInterface({ input: process.stdin });
for await (const line of lines) {
  let reply;
  try {
    reply = { ok: true, value: serve(JSON.parse(line)) };
  } catch (error) {
    reply = { ok: false, error: String(error?.stack ?? error) };
  }
  process.stdout.write(`${JSON.stringify(reply)}\n`);
}
