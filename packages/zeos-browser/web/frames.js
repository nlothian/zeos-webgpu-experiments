// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * One frame format for every place a worker call crosses a byte boundary: a Node pipe
 * (`node_bridge.mjs`) and a SharedArrayBuffer (`model_channel.js`). A frame is a
 * little-endian uint32 header length, a JSON header, then the raw bytes of every blob the
 * header lists in `blobs`. A typed array travels as a blob, so a vocabulary-sized token
 * mask or an attention vector never becomes JSON; in the header it is
 * `{"$blob": i, "type": "Float32Array"}`.
 */

const TYPES = { Uint8Array, Int32Array, Float32Array };

function encodeValue(value, blobs) {
  if (value === null || value === undefined) return null;
  for (const [name, Type] of Object.entries(TYPES)) {
    if (value instanceof Type) {
      blobs.push(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
      return { $blob: blobs.length - 1, type: name };
    }
  }
  if (Array.isArray(value)) return value.map((v) => encodeValue(v, blobs));
  if (typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, encodeValue(v, blobs)]));
  }
  return value;
}

function decodeValue(value, blobs) {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map((v) => decodeValue(v, blobs));
  if ("$blob" in value) {
    const bytes = blobs[value.$blob];
    const Type = TYPES[value.type];
    // Copied, so the result owns aligned memory whatever the frame was read from.
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    return new Type(copy.buffer, 0, bytes.byteLength / Type.BYTES_PER_ELEMENT);
  }
  return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, decodeValue(v, blobs)]));
}

/** A header object (whose typed arrays may sit anywhere in `header.value` or
 * `header.args`) as one Uint8Array frame. */
export function encodeFrame(header) {
  const blobs = [];
  const body = { ...header };
  if ("value" in body) body.value = encodeValue(body.value, blobs);
  if ("args" in body) body.args = encodeValue(body.args, blobs);
  body.blobs = blobs.map((b) => b.byteLength);
  const json = new TextEncoder().encode(JSON.stringify(body));
  const total = 4 + json.byteLength + blobs.reduce((a, b) => a + b.byteLength, 0);
  const out = new Uint8Array(total);
  new DataView(out.buffer).setUint32(0, json.byteLength, true);
  out.set(json, 4);
  let at = 4 + json.byteLength;
  for (const blob of blobs) {
    out.set(blob, at);
    at += blob.byteLength;
  }
  return out;
}

/** The length of the first complete frame in `bytes`, or 0 if it is not all there yet. */
export function frameLength(bytes) {
  if (bytes.byteLength < 4) return 0;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = view.getUint32(0, true);
  if (bytes.byteLength < 4 + headerLength) return 0;
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + headerLength)));
  const total = 4 + headerLength + header.blobs.reduce((a, b) => a + b, 0);
  return bytes.byteLength < total ? 0 : total;
}

/** Inverse of `encodeFrame`, for one complete frame. */
export function decodeFrame(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.slice(4, 4 + headerLength)));
  const blobs = [];
  let at = 4 + headerLength;
  for (const length of header.blobs) {
    blobs.push(bytes.subarray(at, at + length));
    at += length;
  }
  if ("value" in header) header.value = decodeValue(header.value, blobs);
  if ("args" in header) header.args = decodeValue(header.args, blobs);
  delete header.blobs;
  return header;
}

/** Answer one decoded request against a worker. Three calls sit outside the interface:
 * `pieces`, every piece at once, because asking one round trip at a time costs seconds;
 * `partialPieces`, `[id, bytes]` for every id whose piece is not whole characters (it
 * holds U+FFFD), so `pieceBytes` is answered without a round trip per id; and `backend`,
 * which execution provider the worker runs on.
 *
 * A *begun* decode step (`request.begun`, sent by `SyncModelWorker.beginDecodeStep` or
 * `NodeWorker.beginDecodeStep`) can be cancelled while it runs. A function cannot cross a
 * frame, so the transport passes `shouldStop`, which reads its own abort signal (a slot of
 * the shared buffer, or the latest cancel frame), and it is added to the step's options
 * here. */
export async function serveRequest(worker, request, { shouldStop = null } = {}) {
  try {
    let value;
    if (request.begun && request.method === "decodeStep" && shouldStop !== null) {
      const [jobId, opts] = request.args;
      value = await worker.decodeStep(jobId, { ...(opts ?? {}), shouldStop });
    } else if (request.method === "pieces") {
      value = [];
      for (let id = 0; id < worker.meta.tokenizerSize; id++) value.push(worker.piece(id));
    } else if (request.method === "partialPieces") {
      value = [];
      for (let id = 0; id < worker.meta.tokenizerSize; id++) {
        if (worker.piece(id).includes("\ufffd")) value.push([id, worker.pieceBytes(id)]);
      }
    } else if (request.method === "backend") {
      value = worker.backend;
    } else {
      value = await worker[request.method](...request.args);
    }
    return { id: request.id, ok: true, value: value === undefined ? null : value };
  } catch (error) {
    return { id: request.id, ok: false, error: `${error.name}: ${error.message}` };
  }
}
