# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

# pyright: basic
# PyTorch, transformers and numpy are the export group's, untyped where this reaches.

"""Write the reference logits ``export/bench/checks.html`` holds the OPT+ZEOS graph to.

    uv sync --all-packages --group export
    uv run python packages/zeos-browser/export/opt_zeos_reference.py

For the prompts in ``export/bench/reference_prompts.json``: ``transformers``' last-position
logits of Qwen3.5-4B in float32 for ``short`` and ``long``, and ``export_model.ZeosQwen``'s
for ``secret`` with the ``secretHidden`` positions hidden and open. Written to
``models/.reference/opt-zeos-<key>.npz``, where ``key`` is the first 16 hex digits of the
SHA-256 of the prompts' ids (special tokens parsed) and the hidden positions as JSON, so
the page finds the file for the prompts it tokenised and a changed prompt gets a new
file. Needs the bf16 weights at ``models/Qwen3.5-4B`` and about 20 GB of memory, for a
few minutes; an existing file is left alone.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
PROMPTS = HERE / "bench" / "reference_prompts.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--hf", type=Path, default=PACKAGE / "models" / "Qwen3.5-4B")
    parser.add_argument("--out", type=Path, default=PACKAGE / "models" / ".reference")
    args = parser.parse_args(argv)

    import numpy as np
    import tokenizers
    import torch
    import transformers

    prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))
    tok = tokenizers.Tokenizer.from_file(str(args.hf / "tokenizer.json"))

    def ids(text: str) -> list[int]:
        return list(tok.encode(text, add_special_tokens=False).ids)

    short, long, secret = ids(prompts["short"]), ids(prompts["long"]), ids(prompts["secret"])
    hidden = list(prompts["secretHidden"])
    key = hashlib.sha256(json.dumps([short, long, secret, hidden]).encode()).hexdigest()
    path = args.out / f"opt-zeos-{key[:16]}.npz"
    if path.is_file():
        print(f"{path} exists")
        return 0
    spec = importlib.util.spec_from_file_location("export_model", HERE / "export_model.py")
    assert spec is not None and spec.loader is not None
    export_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32)
    hf.eval()
    with torch.no_grad():
        want = {
            name: hf(torch.tensor([prompt])).logits[0, -1].float().numpy()
            for name, prompt in (("short", short), ("long", long))
        }
        core = export_model.ZeosQwen(hf, int8_embedding=False)
        core.eval()
        mask = [i not in hidden for i in range(len(secret))]
        zeos_open, _ = core.run(secret)
        zeos_hidden, _ = core.run(secret, mask=mask)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        short=want["short"],
        long=want["long"],
        zeos_open=zeos_open.numpy(),
        zeos_hidden=zeos_hidden.numpy(),
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
