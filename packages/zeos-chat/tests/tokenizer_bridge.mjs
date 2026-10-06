// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * A model export's tokenizer and nothing else, as a child process for the Python tests
 * (`tests/qwen_vocab.py`). It speaks `node_bridge.mjs`'s frames and answers `info`,
 * `tokenize` (the workers' `encodePlain`), `piece`, `pieceBytes`, the bulk `pieces` and
 * `partialPieces`, and `spell`: text with the added tokens parsed, which is how a reply
 * the model writes is spelled. No weights are loaded.
 *
 *   node packages/zeos-chat/tests/tokenizer_bridge.mjs MODEL_DIR
 */

import fs from "node:fs";
import path from "node:path";

import { decodeFrame, encodeFrame, frameLength, serveRequest } from "../../zeos-browser/web/frames.js";
import { encodePlain, pieceBytes } from "../../zeos-browser/web/transformers_worker.js";

// zeos-browser owns the npm packages; this file is not under it, so a bare specifier
// would not resolve.
const { Tokenizer } = await import(
  new URL("../../zeos-browser/node_modules/@huggingface/tokenizers/dist/tokenizers.mjs", import.meta.url).href
);

const dir = process.argv[2];
const json = (name) => JSON.parse(fs.readFileSync(path.join(dir, name), "utf8"));
const meta = json("meta.json");
const tokenizer = new Tokenizer(json("tokenizer.json"), json("tokenizer_config.json"));
const size = meta.tokenizerSize ?? meta.vocabSize;

const worker = {
  meta: { tokenizerSize: size },
  backend: "tokenizer",
  info: () => ({
    blockSize: meta.blockSize,
    padId: meta.padId,
    controlIds: meta.controlIds,
    eosId: meta.eosId,
    vocabSize: size,
  }),
  tokenize: (text) => encodePlain(tokenizer, text),
  piece: (id) => tokenizer.decode([id], { skip_special_tokens: false }),
  pieceBytes: (id) => pieceBytes(tokenizer, id),
  spell: (text) => Int32Array.from(tokenizer.encode(text, { add_special_tokens: false }).ids),
};

process.stdout.write(encodeFrame({ ready: true, backend: worker.backend }));
let buffer = new Uint8Array(0);
let chain = Promise.resolve();
process.stdin.on("data", (chunk) => {
  const joined = new Uint8Array(buffer.byteLength + chunk.byteLength);
  joined.set(buffer);
  joined.set(chunk, buffer.byteLength);
  buffer = joined;
  for (let length = frameLength(buffer); length > 0; length = frameLength(buffer)) {
    const request = decodeFrame(buffer.subarray(0, length));
    buffer = buffer.slice(length);
    chain = chain.then(async () => {
      process.stdout.write(encodeFrame(await serveRequest(worker, request)));
    });
  }
});
process.stdin.on("end", () => chain.then(() => process.exit(0)));
