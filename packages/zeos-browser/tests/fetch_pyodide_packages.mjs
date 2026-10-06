// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// Run by `npm install` (postinstall). The pyodide npm package carries the interpreter
// but not its package wheels; loading PyYAML once downloads it from the Pyodide CDN
// into node_modules/pyodide, where every later load finds it without a network. The
// determinism test skips until this has happened.

import { loadPyodide } from "pyodide";

const pyodide = await loadPyodide();
await pyodide.loadPackage(["pyyaml"]);
console.log(`Pyodide ${pyodide.version}: PyYAML cached for offline use`);
