# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Serve ``web/dist/`` cross-origin isolated, which the page needs.

    uv run python demo/space-invaders-web/serve.py [--port 8766] [--dir web/dist]

Both the model thread and the Stop button reach the Pyodide worker through a
SharedArrayBuffer, which a browser offers only to a page sent with the two headers
below. On a host that cannot send them, ``coi_serviceworker.js`` adds them in the
browser instead. Port 8766, so it runs beside coop-count-web's 8765.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import sys
from pathlib import Path

DIST = Path(__file__).resolve().parent / "web" / "dist"
PORT = 8766


class IsolatedHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # Revalidate on each load, so a rebuild reaches the workers on the next reload;
        # an unchanged model file is still answered 304 Not Modified, without the bytes.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--dir", type=Path, default=DIST)
    args = parser.parse_args(argv)
    handler = functools.partial(IsolatedHandler, directory=str(args.dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"serving {args.dir} on http://localhost:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
