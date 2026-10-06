# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Read a journal and the attention file beside it (the ``docs/evidence`` runs), and say what the
measured attention did to integrity.

    uv run python -m zeos_coop_count_web.evidence docs/evidence/pipe-untrusted.jsonl

For every ``integrity.demoted`` event it prints the segments named in ``because``, where
each came from, and the mass each received between the job's previous block boundary
and this one -- the quantity the kernel compares with ``theta_read`` -- recomputed from
the per-step measurements. It also prints the largest such mass any segment less trusted
than its reader received at any boundary, which is the threshold above which no
demotion in the run could have fired, the mass the job's newest, not yet masked block
received, and, given ``--hidden``, every step's mass on blocks a mask hid.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["main", "report"]


def _read(path: Path) -> list[dict[str, Any]]:
    text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def attention_file(journal: Path) -> Path:
    """The attention file beside a journal, compressed or not."""
    plain = journal.with_suffix(".attention.jsonl")
    packed = plain.with_name(plain.name + ".gz")
    return packed if packed.is_file() and not plain.is_file() else plain


def report(journal: Path, attention: Path, *, hidden: Sequence[str] = ()) -> str:
    events = _read(journal)
    rows = {row["seq"]: row for row in _read(attention)}
    origin: dict[int, str] = {}
    integrity: dict[int, int] = {}
    for e in events:
        if e["kind"] == "machine.inject":
            origin[e["segment"]] = (
                f"{e['pipe']} (ring {e['ring']}, {e['principal']}): {' '.join(e['text'])[:60]!r}"
            )
            integrity[e["segment"]] = e["integrity"]
        elif e["kind"] == "segment.opened" and e["segment"] not in origin:
            origin[e["segment"]] = "the job's own output"
            integrity[e["segment"]] = e["integrity"]

    lines: list[str] = []
    job_integrity: dict[int, int] = {}
    pending: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    last_fold: dict[int, dict[str, float]] = {}
    largest: tuple[float, int, str, int] = (0.0, -1, "", -1)
    measured = 0
    for e in events:
        kind = e["kind"]
        if kind == "job.spawned":
            job_integrity[e["job"]] = e["integrity"]
        elif kind == "machine.decode" and e["seq"] in rows:
            measured += 1
            for segment, mass in rows[e["seq"]]["segments"].items():
                pending[e["job"]][segment] += mass
        elif kind == "machine.block_boundary":
            folded = pending.pop(e["job"], {})
            for segment, mass in folded.items():
                if integrity.get(int(segment), 0) > job_integrity.get(e["job"], 0):
                    largest = max(largest, (mass, e["job"], segment, e["seq"]))
            # The kernel demotes right after the boundary event, on this block's mass.
            last_fold[e["job"]] = dict(folded)
        elif kind == "integrity.demoted":
            folded = last_fold.get(e["job"], {})
            lines.append(
                f"- seq {e['seq']}: job {e['job']} demoted {e['from_integrity']} -> "
                f"{e['to_integrity']} because of segment(s) {e['because']}"
            )
            for segment in e["because"]:
                lines.append(
                    f"  - segment {segment}, from {origin.get(segment, '?')}, integrity "
                    f"{integrity.get(segment, '?')}: {folded.get(str(segment), 0.0):.4f} of "
                    "attention since the job's previous boundary"
                )
            job_integrity[e["job"]] = e["to_integrity"]

    head = [
        f"{journal.name}: {sum(1 for e in events if e['kind'] == 'machine.decode')} decode "
        f"steps, {measured} with measured attention",
    ]
    demotions = [e for e in events if e["kind"] == "integrity.demoted"]
    head.append(f"integrity demotions: {len(demotions)}")
    if largest[1] >= 0:
        head.append(
            f"largest mass a less trusted segment received in one block: {largest[0]:.4f} "
            f"(segment {largest[2]}, job {largest[1]}, boundary at seq {largest[3]})"
        )
    else:
        head.append("no segment less trusted than its reader received any attention")
    unmasked = [row.get("unmasked", 0.0) for row in rows.values()]
    if unmasked:
        positive = [u for u in unmasked if u > 0]
        head.append(
            f"mass on the newest, not yet masked block: mean {sum(unmasked) / len(unmasked):.4f}"
            f" per step, max {max(unmasked):.4f}; non-zero on {len(positive)} of "
            f"{len(unmasked)} steps"
        )
    for spec in hidden:
        job, _, block = spec.partition(":")
        steps = [row for row in rows.values() if row["job"] == int(job)]
        on = [row["blocks"].get(block, 0.0) for row in steps]
        head.append(
            f"block {block} of job {job} (hidden by --hide): measured on {len(steps)} steps, "
            f"largest mass {max(on, default=0.0)!r}, steps with non-zero mass "
            f"{sum(1 for v in on if v != 0.0)}"
        )
    return "\n".join(head + lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("journal", type=Path)
    parser.add_argument("--attention", type=Path)
    parser.add_argument("--hidden", action="append", default=[], metavar="JOB:BLOCK")
    args = parser.parse_args(argv)
    attention = args.attention or attention_file(args.journal)
    sys.stdout.write(report(args.journal, attention, hidden=args.hidden))
    return 0


if __name__ == "__main__":
    sys.exit(main())
