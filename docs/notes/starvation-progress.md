# Starvation counts preemptions since the last progress

Design note for the starvation rule in `Transformer-OS.md` §5.5 and its
implementation in `src/zeos/core/kernel.py` (`Kernel._handle_request`,
`Kernel._preempt`).

## Problem

§5.5 asks for a loud scheduler fault when a low-priority job is preempted more
than K times: an *interrupt storm* in which the job is interrupted so often that
it never gets anything done. The kernel implemented this as a lifetime budget.
`Scheduler.preempt()` incremented `Job.preempt_count`, nothing ever reset it,
and `Kernel._preempt()` raised `scheduler_fault_starvation` once the count passed
`KernelConfig.starvation_limit` (8). With no `on_fault` the job escalated to
FAULTED.

That faults the opposite of a storm. A long-lived job that is interrupted now and
then, and completes work in between, is exactly what a preemptive kernel is for,
yet it died at its ninth interruption however much it had done.

## Evidence

The Space Invaders pilot (`demo/space-invaders`) is a preemptible job beside an
`evade` reflex that preempts it on every incoming bomb. The pilot makes moves
between bombs.

- In the browser port the pilot was faulted 6-12 s into play on the ablation
  board, and around tick 120 on the default board.
- A native episode reproduces it: stub machine (`stubs.stub_machine("shoot",
  delay=0.005)`), `ZeosRealtimeRunner(seed=7, tick_seconds=0.02, max_ticks=600,
  rules=Rules(fire_chance=0.9))`. Under the lifetime count the pilot was
  faulted at its 9th preemption, after 18-19 moves. Under the rule below the
  same episode ran to the end with the pilot alive, after 18-21 preemptions and
  17-25 moves (the runner is wall-clock paced, so counts vary run to run). With
  `fire_chance` 0.7 and 0.5 it was likewise faulted before and alive after.
- The rule still fires on a real storm. With a turn that outlasts a tick
  (`delay=0.05`) and `fire_chance=0.9`, the reflex fires nearly every tick and
  each resume notice changes the pilot's transcript, so its turn restarts and it
  goes nine preemptions without completing a move. That run still faults with
  the new rule, and it should.

## Rule

`Job.preempt_count` counts **preemptions since the job last made progress**.

**Progress is a request actually carried out.** The kernel resets the count by
calling `Kernel._progressed` at the points where a request took effect, and
nowhere else:

| Request | Counts as progress | Does not count |
| --- | --- | --- |
| write | landed in its pipe (`_do_write`, after `_put`), including the write half of write-then-read; sent over the link; a guard's verdict write | parked behind backpressure or on a gate until it lands or is allowed; refused |
| gated write | the gate allows it (`_settle_gate`) | vetoed, gate timed out (`_expire_gates`), guard unreachable and the failure policy denies |
| read / select | consumed data (`_consume_read`), even when the data is then alarmed on as a spoof; blocked waiting for data | unbound pipe, sink pipe |
| acquire | granted; queued behind a holder | re-acquiring a resource already held; undeclared or unknown resource; the deadlock victim |
| release | gave back a resource the job held | a resource not held, or one that does not exist (both silent no-ops) |
| spawn | a child or compartment started | outside the descriptor's children |
| page-in (`fault`, `need`) | content loaded (`_complete_page_in`), even when the refault also raises a thrash fault | a miss or a duplicate answered with a notice; a `fault` naming no segment |
| exit | ends the job, so nothing is left to count | |

A malformed request is never progress.

The fault itself is unchanged except for its wording: `_preempt` raises
`scheduler_fault_starvation` when the count passes K, through the same fault
machinery, with the detail "preempted N times without progress (limit K)"
(previously "preempted N times, over the limit of K"). A job preempted K+1 times
with no request carried out in between faults exactly as it did before.

Why a request carried out: it is the kernel's own unit of a job getting
something done. A job acts on the world only through requests ("effects are
syscalls"), and a request is what an interrupt storm prevents from ever
completing. A blocking read, select or acquire counts as soon as it parks,
because waiting for input is that request's normal outcome and a blocked job
cannot be preempted. A parked write counts only when it lands or its gate
allows it, because a gate can still veto it. A write parked behind backpressure
lands as the first thing the job does when it wakes, before it can be preempted
again.

## Rejected alternatives

- **Lifetime count** (the original). Faults any long-lived job that is
  legitimately interrupted, which is the opposite of a storm.
- **Reset on every decoded token.** Too weak: a storm can let a token through
  between interrupts, and a job that decodes one token per gap gets nothing
  done.
- **Reset on resume.** Makes the limit meaningless, since every preemption ends
  in a resume.
- **Decay with time or ticks.** The silent aging §5.5 explicitly refuses: the
  count would forget a storm instead of reporting it.
- **Fault-as-proxy**: count any request that raised no fault against the job.
  This was the first implementation of this rule, and review found that it gets
  the edges wrong in both directions. A write parked on a gate raises no fault,
  even if the gate later vetoes it or times out, so a job under `on_fault: retry`
  whose every write is vetoed would never starve. Silent no-ops raise no fault
  either: releasing a resource that is not held, re-acquiring one that is, or a
  `fault` naming no segment. In the other direction, a read that consumed data
  but tripped the spoof alarm, a page-in that loaded content alongside a thrash
  fault, and the write half of a write-then-read whose read was refused were all
  carried out, yet would not count. Explicit marking at the points where a
  request took effect says what progress is instead of inferring it.
- **Count a refused request as progress.** A job retrying a refused call, under
  `on_fault: retry` or `continue`, would reset its own count forever while
  achieving nothing.

## Compatibility

- No event type or field changed. The journal changes in two ways. Where the
  lifetime count used to fault a job that had made progress since its earlier
  preemptions, that `FaultRaised`/`FaultDispatched`/`JobStateChanged` sequence no
  longer appears, and the job runs on. And the starvation `FaultRaised.detail`
  text now reads "preempted N times without progress (limit K)". No test,
  golden trace or demo pinned the old text.
- The golden shape trace (`tests/replay/golden/smoke.shape`) is unchanged.
- The existing starvation tests are unchanged and still pass: the stale-stack
  victim in `tests/integration/test_suspension_stack.py` only ever emits, so it
  is a storm under either rule.
- The Fleet body-eviction count (`LeaseBook.note_eviction`, checked in
  `Kernel.revoke_lease`) is a separate lifetime counter and is not reset by
  progress.

## Tests

`tests/integration/test_starvation_progress.py`. Every assertion is on the
journal: how many of the victim's preemptions are journalled before its
starvation fault, if there is one.

- a storm (K+1 preemptions, only decoded tokens between them) still faults;
- a job preempted 5K times with a write between each pair never faults;
- exactly K preemptions without progress is within the limit;
- K, progress, K more is within the limit, and one more after that faults (a
  lifetime total of 2K+1, with the one write journalled between the Kth and the
  (K+1)th preemption);
- a refused write is not progress;
- a vetoed gated write, under `on_fault: retry`, is not progress;
- a gated write that the gate allows is progress;
- a read that consumed data is progress even when it raises a spoof fault;
- releasing a resource that is not held, or does not exist, is not progress.

Against the lifetime count, the second, fourth, seventh and eighth tests fail.
Against fault-as-proxy, the vetoed-write, spoofed-read and both no-op release
tests fail.
