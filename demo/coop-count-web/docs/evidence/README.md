# Measured attention in the kernel: the evidence

Journals of the coop-count cases run by `JsMachine` over the model worker
(Qwen2.5-0.5B-Instruct, int8, onnxruntime-web's WebAssembly backend, one thread), and the
attention each decode step measured. They were written by `run.sh` with `node_run`, which
ran the model worker under Node; the model now runs on WebGPU only, so neither is in the
repository any more (both are at `a397805`), and these files are the record of those runs.
`report.txt` is `python -m zeos_coop_count_web.evidence` over each journal.

The journal has no field for attention: its events are the kernel's, and `machine.decode`
records the words decoded, not what the step attended. So `node_run` wrote the
measurement beside the journal, one line per decode step keyed by the sequence number of
that step's `machine.decode` event: the mass per kernel block that `JsMachine` returned in
`DecodeResult.attention`, the mass per segment as the kernel summed it, and the mass on
blocks the installed mask was not computed over. `*.attention.jsonl.gz` are those files
for the three runs below; the stock runs' are summarised in `report.txt`.

| file | case | what differs from the case as committed |
|---|---|---|
| `scripted.jsonl` | `coop-count-scripted`, its `events.jsonl` | nothing |
| `pipe.jsonl` | `coop-count-pipe`, its `events.jsonl` | nothing |
| `vector.jsonl` | `coop-count-vector`, no schedule | nothing |
| `pipe-untrusted.jsonl` | `coop-count-pipe`, its `events.jsonl` | `keys.interrupt` and `keys.number` declared `EXTERNAL` (ring 3) |
| `pipe-untrusted-theta-0.5.jsonl` | the same | and `theta_read` 0.5 |
| `pipe-hidden.jsonl` | `coop-count-pipe`, its `events.jsonl` | kernel block 2 of counter-a removed from every mask |

Every run stops at 400 ticks, as `demo/coop-count/run_all.py` stops a live seat; the
counters never exit. Each journal's 398 `machine.decode` events all have a measured
attention line, and every line sums to 1.0 within 1e-3 (`JsMachine` refuses one that does
not).

## An integrity demotion driven by measured attention

As committed, every pipe in the coop-count cases is `TRUSTED` (ring 2), the jobs start at
integrity 2, and so nothing a job can read is less trusted than the job: no run of the
stock cases can demote, on any backend. `pipe-untrusted` declares the console untrusted.
The keypress at 120 ms fires the `keyboard-interrupt` vector, the kernel spawns
`reset-count` (job 3) and injects the keypress, `attention`, as segment 19: ring 3,
integrity 3, principal `user` (seq 217). The handler's first decode step put 0.3647 of
its attention on the block holding segment 19, the block boundary that followed found
that at or above `theta_read` = 0.2, and the kernel journalled

```json
{"because":[19],"from_integrity":2,"job":3,"kind":"integrity.demoted","seq":223,"to_integrity":3}
```

`because` names segment 19, and the attention file gives the mass:
`{"seq": 221, "job": 3, ... "segments": {"18": 0.6353, "19": 0.3647}}` (segment 18 is the
handler's descriptor body). The kernel sums a segment's mass over every block it touches,
so segments 20 and 21, injected into the same block afterwards, are credited the same
block's mass on later steps; the demotion came before they existed.

`pipe-untrusted-theta-0.5` is the same run with `theta_read` 0.5, above the largest mass
any segment less trusted than its reader received in one block (0.3647). No demotion
fires, and the journal is the 0.2 run's event for event, without the
`integrity.demoted`: the handler decodes the same words either way, since its writes were
not refused.

## A masked block receives exactly zero

`pipe-hidden` removes kernel block 2 of counter-a (job 1), inside its descriptor body,
from every mask the kernel installs for it, through `set_mask` like any other narrowing.
The kernel installs counter-a's first mask after injecting the body and before its first
decode, so the block is hidden from every step. Over the run's 387 decode steps of job 1
the measured mass on block 2 is 0.0 on every step: the worker sets the block's keys to
`-inf` before the softmax in every layer, so no head attends it and the reported entry is
an exact zero rather than a small number. Hiding part of the prompt changes what the
model writes, which is why this run's counter-a decodes 387 steps against 66 in `pipe`.

## The block being written into

The kernel builds a mask from the segments that exist when it installs it, so the block a
job is decoding into is in no mask until the next refresh; `JsMachine` treats such blocks
as visible (README, *the mask's horizon*). That block received 0.17 of each step's
attention on average (at most 0.39) and some attention on 336 of the 398 steps of
`pipe`. Hidden, it would take away the tokens the job has just written.

## What remains unmeasured

Kernel-side summaries of attention over eviction decisions and taint spread are still
untested at scale, and this backend is a small quantised model in a browser, not a
production serving stack.
