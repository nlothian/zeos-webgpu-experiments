#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.
#
# Rewrites every journal and attention file in this directory. Needs the export
# (export/export_model.py) and `npm install` in demo/coop-count-web. Each run is the
# model worker on onnxruntime-web's WebAssembly backend, one thread, under Node; the
# same binary the page runs, which is why the page's journal for a case is byte for byte
# the one written here (see README.md). Runs in parallel; a few minutes in all.
set -eu
cd "$(dirname "$0")/../.."
OUT=docs/evidence
CASES=../coop-count/cases
PIPE=$CASES/coop-count-pipe
run() { uv run python -m zeos_coop_count_web.node_run --quiet "$@"; }

run $CASES/coop-count-scripted --events $CASES/coop-count-scripted/events.jsonl \
    --journal $OUT/scripted.jsonl &
run $PIPE --events $PIPE/events.jsonl --journal $OUT/pipe.jsonl &
run $CASES/coop-count-vector --journal $OUT/vector.jsonl &
# The console declared untrusted: what arrives on the keyboard pipes is ring 3.
run $PIPE --events $PIPE/events.jsonl --journal $OUT/pipe-untrusted.jsonl \
    --ring keys.interrupt=EXTERNAL --ring keys.number=EXTERNAL &
run $PIPE --events $PIPE/events.jsonl --journal $OUT/pipe-untrusted-theta-0.5.jsonl \
    --ring keys.interrupt=EXTERNAL --ring keys.number=EXTERNAL --theta-read 0.5 &
# Kernel block 2 of counter-a (job 1), inside its descriptor body, taken out of every mask.
run $PIPE --events $PIPE/events.jsonl --journal $OUT/pipe-hidden.jsonl --hide 1:2 &
wait
for journal in $OUT/*.jsonl; do
  case $journal in *.attention.jsonl) continue ;; esac
  case $journal in *hidden*) hidden="--hidden 1:2" ;; *) hidden="" ;; esac
  uv run python -m zeos_coop_count_web.evidence "$journal" $hidden
done > $OUT/report.txt
# The attention files are a megabyte each. The three the write-up reads are kept,
# compressed; the rest are summarised in report.txt and dropped.
for name in pipe-untrusted pipe-untrusted-theta-0.5 pipe-hidden; do
  gzip -9 -n -f $OUT/$name.attention.jsonl
done
rm -f $OUT/*.attention.jsonl
