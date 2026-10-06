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
  faulted at its 9th preemption, after 19 moves. Under the rule below the same
  episode ran to the end with the pilot alive, 18-19 preemptions and 22-25 moves
  (the runner is wall-clock paced, so counts vary slightly run to run). With
  `fire_chance` 0.7 and 0.5 it was likewise faulted before and alive after.
- The rule still fires on a real storm. With a turn that outlasts a tick
  (`delay=0.05`) and `fire_chance=0.9`, the reflex fires nearly every tick and
  each resume notice changes the pilot's transcript, so its turn restarts and it
  goes nine preemptions without completing a move. That run still faults with
  the new rule, and it should.

## Rule

`Job.preempt_count` counts **preemptions since the job last made progress**.

**Progress is a request the kernel serviced without refusing it.** In
`_handle_request`, after a decoded request (any op other than `NONE`) has been
serviced, the count is reset to zero if servicing it raised no fault against the
job. That covers a write that landed or was parked (backpressure, a gate), a
read that was satisfied or blocked, a select, an acquire granted or queued, a
release, a spawn, a page-in (`fault`/`need`), and an exit. A request that raised
a fault -- refused by a capability check, malformed, naming an unbound pipe or a
spawn outside the children -- is not progress. `Job.faults_raised`, incremented
in `_raise_fault`, is how the kernel tells the two apart.

The fault itself is unchanged: `_preempt` raises `scheduler_fault_starvation`
when the count passes K, with the same detail text, through the same fault
machinery. A job preempted K+1 times with no serviced request in between faults
exactly as it did before.

Why a serviced request: it is the kernel's own unit of a job getting something
done. A job acts on the world only through requests ("effects are syscalls"),
and a request is what an interrupt storm prevents from ever completing. Blocking
is covered too, because a job only blocks by issuing a request the kernel
accepts and parks.

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
- **Count a refused request as progress.** A job retrying a refused call, under
  `on_fault: retry` or `continue`, would reset its own count forever while
  achieving nothing.

## Compatibility

- No event type or field changed. A journal changes only where the lifetime
  count used to fault a job that had serviced a request since its last
  preemptions: that `FaultRaised`/`FaultDispatched`/`JobStateChanged` sequence no
  longer appears, and the job runs on.
- The golden shape trace (`tests/replay/golden/smoke.shape`) is unchanged.
- The existing starvation tests are unchanged and still pass: the stale-stack
  victim in `tests/integration/test_suspension_stack.py` only ever emits, so it
  is a storm under either rule.
- The Fleet body-eviction count (`LeaseBook.note_eviction`, checked in
  `Kernel.revoke_lease`) is a separate lifetime counter and is not reset by
  progress.

## Tests

`tests/integration/test_starvation_progress.py`:

- a storm (K+1 preemptions, only decoded tokens between them) still faults;
- a job preempted 5K times with a write between each pair never faults;
- exactly K preemptions without progress is within the limit;
- K, progress, K more is within the limit, and one more after that faults (with a
  lifetime total of 2K+1);
- a refused write is not progress.

The second and fourth fail against the lifetime count.
