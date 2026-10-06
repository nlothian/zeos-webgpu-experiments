// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * `mask_core.measure` under Pyodide in Node: the interpreter the page's mask runs in.
 *
 * Copies `token_mask.py`, the ABI modules it needs and `mask_core.py` into Pyodide's
 * filesystem (with an empty `zeos/__init__.py`, whose real one asks importlib.metadata
 * for a version no wheel is installed to answer), reads the vocabulary and walks
 * `mask_cost.py` left in `.cache/mask_inputs.json`, and times the copy as
 * `PyodideBridge._flags` makes it, `Uint8Array.new(to_js(flags))`. Writes
 * `results/mask_cost_pyodide.json`.
 *
 *   node demo/space-invaders-web/bench/mask_pyodide.mjs   # after mask_cost.py
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const BENCH = fileURLToPath(new URL(".", import.meta.url));
const REPO = join(BENCH, "..", "..", "..");
const BROWSER = join(REPO, "packages", "zeos-browser");
const pyodideDir = join(BROWSER, "node_modules", "pyodide");
const { loadPyodide } = await import(pathToFileURL(join(pyodideDir, "pyodide.mjs")).href);
const pyodide = await loadPyodide({ indexURL: `${pyodideDir}/` });

const root = "/bench";
const files = {
  "zeos/__init__.py": "",
  "zeos/core/__init__.py": readFileSync(join(REPO, "src/zeos/core/__init__.py"), "utf8"),
  "zeos/core/ids.py": readFileSync(join(REPO, "src/zeos/core/ids.py"), "utf8"),
  "zeos/machine/__init__.py": readFileSync(join(REPO, "src/zeos/machine/__init__.py"), "utf8"),
  "zeos/machine/base.py": readFileSync(join(REPO, "src/zeos/machine/base.py"), "utf8"),
  "zeos/machine/abi.py": readFileSync(join(REPO, "src/zeos/machine/abi.py"), "utf8"),
  "zeos_browser/__init__.py": "",
  "zeos_browser/token_mask.py": readFileSync(join(BROWSER, "src/zeos_browser/token_mask.py"), "utf8"),
  "mask_core.py": readFileSync(join(BENCH, "mask_core.py"), "utf8"),
};
for (const [name, text] of Object.entries(files)) {
  const path = `${root}/${name}`;
  pyodide.FS.mkdirTree(path.replace(/\/[^/]*$/, ""));
  pyodide.FS.writeFile(path, text);
}
pyodide.FS.writeFile(`${root}/inputs.json`, readFileSync(join(BENCH, ".cache", "mask_inputs.json")));

const result = await pyodide.runPythonAsync(`
import sys, json
sys.path.insert(0, "${root}")
from js import Uint8Array
from pyodide.ffi import to_js
import mask_core

inputs = json.load(open("${root}/inputs.json"))
out = mask_core.measure(
    inputs["pieces"],
    reserved=inputs["reserved"],
    control=inputs["control"],
    aliases=inputs["aliases"],
    valued=inputs["valued"],
    walks=inputs["walks"],
    copy=lambda m: Uint8Array.new(to_js(m)),
)
out["runtime"] = "pyodide " + sys.version.split()[0]
json.dumps(out)
`);
const out = JSON.parse(result);
writeFileSync(join(BENCH, "results", "mask_cost_pyodide.json"), `${JSON.stringify(out, null, 2)}\n`);
const { walks, payload_states, ...summary } = out;
console.log(JSON.stringify(summary, null, 2));
for (const [cmd, rows] of Object.entries(walks)) {
  for (const row of rows) console.log(cmd, JSON.stringify(row.pieces), `new=${row.new_states}`, `cold=${row.cold_ms.toFixed(0)}ms`);
}
console.log(payload_states.map((p) => `${p.chars}:${p.cached ? "hit" : p.ms.toFixed(0)}`).join(" "));
