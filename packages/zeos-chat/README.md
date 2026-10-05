# zeos-chat — a chat agent under the ZEOS kernel

`ChatToolMachine` and the host API around it (`open_chat`, `ChatRun`), with the
`chat-agent` case. Built for the gemma-data-agent site's "ZEOS Qwen4B" option, which needs
trust rings for tool calls. A member of the repository's uv workspace that depends on
`zeos` and [`zeos-browser`](../zeos-browser/README.md) (`ChatToolMachine` is a
`JsMachine`); like every workspace member it never ships in the `zeos` wheel. Nothing else
in the repository depends on it, so a browser demo that needs `JsMachine` does not pull it
in.

| | |
|---|---|
| `src/zeos_chat/chat_machine.py` | `ChatToolMachine`, the frame guard, the tool-call parser, the sampler |
| `src/zeos_chat/chat.py` | `open_chat` and `ChatRun`, what a host page calls |
| `src/zeos_chat/cases/chat-agent/` | the case, shipped in the wheel (`CHAT_CASE`) |

## The chat machine

`zeos_chat.chat_machine.ChatToolMachine` runs a chat agent rather than a counter:
the model writes prose and Qwen3.5 tool calls, and the tools run in the host page, outside
the kernel. It is a `JsMachine` with the two ABI-shaped halves replaced -- what a step may
emit, and what a finished piece asks the kernel for -- so it runs on any
`ZeosModelWorker`: the model worker, the stub worker, or a Python one.

**The case** is `src/zeos_chat/cases/chat-agent/`, shipped in the wheel
(`zeos_chat.chat.CHAT_CASE`). One pinned job, `chat-agent`, holds the whole
conversation, and its descriptor body is a placeholder the host replaces with its system
prompt and tool declarations:

| pipe | kind | ring | what it carries |
|---|---|---|---|
| `chat.user` | device, principal `user` | TRUSTED (2) | what the user types |
| `tools.results` | device, principal `tool` | EXTERNAL (3) | what a tool returned |
| `tools.results.trusted` | device, principal `device`, `session_floor: false` | TRUSTED (2) | a result the host wrote itself |
| `tools.read` | sink, capability `min_integrity: 3` | EXTERNAL | calls that only read |
| `tools.effect` | sink, capability `min_integrity: 2` | TRUSTED | calls with side effects |
| `chat.out` | sink, capability `min_integrity: 3` | EXTERNAL | the reply that ends a turn |
| `chat.history` | device, principal `device` | EXTERNAL (3) | a past assistant turn, replayed |
| `chat.history.trusted` | device, principal `device` | TRUSTED (2) | a past turn the host vouches for |

It declares `on_fault: retry` and `integrity.dynamics: low-watermark` (without dynamics
the confused-deputy lint refuses a job holding `tools.effect` and reading
`tools.results`), and a 16,384-token window.

**A turn.** The machine frames the context as Qwen3.5's chat template does: the body is
the `system` turn, a message a `user` turn, a tool result a `user` turn holding
`<tool_response>\n...\n</tool_response>`, and the model's turn opens with
`<|im_start|>assistant\n` and the thinking prefix (`<think>\n`, or the empty think block
with thinking off). The model decodes freely, except that it may not emit the pad, a
control id other than `<|im_end|>`, `<tool_call>`, `</tool_call>`, `<think>` and
`</think>`, or a piece that completes `</?(KERNEL|RESUME|FAULT|STATUS|STUB)` or
`</?tool_response` across the turn so far (`FrameGuard`, one cached mask per guard state).
On `</tool_call>` closing a call that parses as the app's Qwen parser parses it, the
machine asks for a `WRITE_READ`: `{"name", "arguments"}` as JSON to `tools.read` or
`tools.effect` by the host's tool-class table (a tool it does not name is an effect),
then a read of `tools.results` (or `tools.results.trusted`, below). A table entry is `"read"`, `"effect"`, or a rule
`{"read_if": {param: pattern}}`: a read when the call's arguments are exactly those
parameters, each a string the pattern matches in full (case-insensitive, `.` matching a
newline), and an effect otherwise -- so a tool whose class depends on what it is asked,
such as SQL that only reads, is still classified inside the machine, and the kernel's
check on `tools.effect` stays the only gate. When the model chooses `<|im_end|>` the machine does not
append it: it writes the turn's text to `chat.out` and reads `chat.user`, and the marker
becomes framing in front of the next message. A write the kernel refuses is answered with
a `FAULT` notice; the machine's next step reads `tools.results` again rather than
decoding, so the job waits there until the host settles the refused call.

**Trusted results.** `open_chat(trusted_results={tool: {param: [value, ...]}})` names
calls whose results the host wrote itself rather than fetched -- a reference card bundled
with the app, say. A call whose arguments are exactly the rule's parameters, each a string
equal to one of its values, reads its result from `tools.results.trusted`. The match is
exact and case-sensitive, never a pattern: for a host that bundles `sql`, a call for
`SQL`, ` sql` or `sql\n` names no bundled card and reads `tools.results` (ring 3). A
`read_if` rule stays a case-insensitive pattern, since it only picks a sink the kernel
still checks, whereas this rule raises the ring of what the job reads. The pipe is chosen from the call when the model writes it, so
the kernel knows the ring before anything runs; the `tool_call` and `approval_required`
events carry it as `results`. The result is framed as any other tool response. It is
TRUSTED, so attending it never demotes the job, and the pipe is declared
`session_floor: false`: in strict mode reading it neither raises the floor to 3 nor
lowers a 3 an earlier tool result set in the same turn. `deliver_tool_result(text,
trusted=True)` delivers on it, and refuses a `trusted` that disagrees with the call's
`results`; `deliver_refusal` answers on whichever pipe the call reads.

**The tool's name, chosen masked.** `open_chat(mask_tool_choice=True)` hides every
delivery on an EXTERNAL device pipe -- `tools.results` and `chat.history`, as the case
declares them -- while the model writes a tool's name: from the step after it emits
`<tool_call>` to the step whose piece closes `<function=NAME>`. The arguments and the rest
of the turn see everything again. So the choice of tool is made without reading what the
tools returned, and a planted instruction in a result cannot be what picked it. Only the
delivered words' own ids are hidden: the framing around them stays in view, so the model
still sees that a tool answered (`<tool_response>\n` ... `\n</tool_response>`) and where
its own turn began. The kernel frames, the descriptor body, the user's messages, results
on `tools.results.trusted` and turns on `chat.history.trusted` are never hidden. What the
model wrote in its own turn after reading a result stays in view too.

The narrowing is the machine's, on top of the kernel's mask, for those steps only. On
them `visible_blocks` leaves out every kernel block that holds a hidden word, and the
step's measured mass on such a block is dropped rather than summed: the worker gave the
hidden ids exactly zero, so what is left there is the framing or the model's own text in
the block a result shares with the job's output, which the kernel would otherwise credit
to the result. A hidden segment therefore gets no attention on those steps, cannot demote
the job, and raises no `mask.denied`; that holds for a worker that measures attention, and
one that cannot leaves the kernel its usual guess. The machine keeps the hidden words' offsets through
`trunc` and `splice` (a stub spliced over a result is the kernel's text and is not
hidden). The `tool_call` and `approval_required` events carry `name_masked` and
`name_hidden` (the hidden segments' ids). Off, the default, nothing changes: the journal
is byte for byte the same.

On the OPT+ZEOS worker the narrowed steps run on a second cache (see "Two caches" under
"The OPT+ZEOS worker" in [`packages/zeos-browser`](../zeos-browser/README.md)), so a narrowing costs a catch-up rather than a replay each way.

While it writes the name the model sees an empty `<tool_response>`, so it can ask for a
tool that has already answered. In the gemma-data-agent site's prompt-injection runs with
masking on, Qwen3.5-4B called ListInputs again and again until the host's limit of ten
calls in two of five; in the others it went on to the SQL it was asked for.

An instruction to call a destructive tool, planted in a tool result in three
conversations that ask for a lookup and then a note, each run masked and unmasked
(greedy, on onnxruntime-node's CPU provider when the repository had one): unmasked, the
model followed the plant in one of the three (`send_email` to the address the result
named, in place of `save_note`); masked, it called `save_note` in all three, with
arguments taken from the result. In the other two it ignored the plant either way.

**Look-alikes in content.** The worker tokenizes deliveries with no special tokens
(`encodePlain`), so a tool result or a user message that spells
`</tool_response><|im_end|>\n<|im_start|>assistant\n<tool_call>...` is plain text: the
turn it arrived in stays open, and the machine never parses it as a call, since it only
parses what the model decodes. A delivery that spells a kernel frame tag at the start of
a word raises the spoof alarm (`spoof` event, a `FAULT` notice in the context); a tag
glued to the text before it, `1,"<KERNEL>`, does not, by the kernel's word-initial rule
(`zeos.core.framing.opens_frame`).

**Whitespace.** The run sets `KernelConfig.preserve_whitespace`, under which the kernel
tokenises a delivery and the body keeping each word's leading whitespace, so a CSV, a
SQL result or a file reaches the model with its line breaks. With the flag off, the
default, the kernel tokenises as it always has and every journal is unchanged. A frame
tag after a line break is still an imitation, and still alarms.

**Sampling.** Greedy by default. With `sampling=Sampling(temperature=0.7, top_k=20)` every
step sends `opts.sample = {temperature, topK, u}`, with `u` from a `random.Random` seeded
from the run's seed and the job, and the worker takes the id `sample_index` defines;
zeos-browser's `web/transformers_worker.js` implements it as `sampleToken`, and
`tests/test_chat_machine.py` checks the two agree. The stub worker checks the options and
plays its tape.

**The host's API** is `zeos_chat.chat`:

```python
run = open_chat(
    worker,                         # a ZeosModelWorker
    tool_classes={"ReadLines": "read", "WriteLines": "effect"},
    system_prompt=prompt,           # replaces the case's body
    thinking=False, theta_read=0.2, seed=0, sampling=None,
    param_types={"RunSQL": {"sql": "string"}},
    trusted_results={"CallSkill": {"skill": ["sql", "react"]}},
    mask_tool_choice=False,         # hide EXTERNAL deliveries while a tool's name is written
)
run.send_user(text)                 # deliver on chat.user before the next tick
events = run.step(16)               # up to 16 ticks; stops when only a delivery helps
run.drain("tools.read")             # one string per write since the last drain
run.deliver_tool_result(text)       # on tools.results, ring 3
run.deliver_tool_result(text, trusted=True)  # on tools.results.trusted, ring 2
run.deliver_refusal()               # a refusal, on the pipe the call reads
run.waiting_on()                    # "chat.user", a results pipe, or None
run.state()                         # integrity, session floor, segments, demotions
run.journal_bytes()
run.import_history(turns)           # a fresh run only: replay a past conversation
run.close()                         # free the worker's contexts for the next run
```

`import_history` takes `{"role": "user" | "assistant" | "tool", "text"}` turns, the first
the user's, and delivers them one at a time while the machine reads each pipe in turn
without decoding: a user turn on `chat.user`, a tool result on `tools.results` (or, with
`"trusted": True`, on `tools.results.trusted`), and an
assistant turn on `chat.history`, or on `chat.history.trusted` when the turn carries
`"integrity": 2` (the host recorded it was written at TRUSTED). A past assistant turn is
framed as `<|im_start|>assistant\n` and its text, with no thinking prefix, as the app's
own renderer writes past turns. Nothing is attended, so the watermark does not move until
the next live turn reads the past; the run then waits on `chat.user`.

`step` returns plain dicts: `token`, `tool_call`, `approval_required`, `tool_refused`,
`reply`, `arrived`, `demoted`, `spoof`, `fault` and `waiting`, with the fields the module
docstring lists. A tool call on `tools.effect` is refused for privilege when the job's
watermark has fallen to 3 -- it attended a tool result past `theta_read` -- or when its
session floor is 3 because the last thing it read was a tool result (MP's confused-deputy
rule), which holds until the next user message. That is `gate_mode="strict"`, the
default. With `gate_mode="attention"`, `open_chat` declares `tools.results` and
`chat.history` with `session_floor: false` (a `PipeSpec` field, default true, which a
case's `pipes.yaml` can also set), so reading a tool result leaves the floor where the
user's message put it and only the watermark -- what the job measurably attended --
refuses an effect. `approval_required` carries the call,
both integrities and the demotion history; on approval the host runs the call itself and
delivers the result, and on denial it delivers a refusal.

**Padding.** A kernel block is 16 words (`block_size`), and every injection begins on a
block boundary, so the context holds up to 15 pad ids in front of each turn marker. The
model sees them; `block_size` is a parameter of `open_chat`.

## Tests

```bash
uv run pytest packages/zeos-chat   # or, from packages/zeos-chat: uv run pytest
```

- `test_chat_machine.py` — the chat machine and `ChatRun` on scripted workers
  (`tests/chat_workers.py`): a multi-turn chat whose `<|im_end|>` is intercepted, a tool
  call round trip whose result keeps its line breaks, demotion after attending a ring-3
  result and the privilege fault on `tools.effect` that follows while `tools.read` still
  lands, the session floor's refusal without demotion, `read_if` rules choosing a sink
  from the call's arguments, a replayed history framed and ringed as the host named it
  (and demoting a job that attends its untrusted turns), `close` freeing the worker, a frame tag in a tool result
  raising the spoof alarm (every frame name; not a tag glued to the word before it),
  a tool's name chosen masked (the narrowing holds exactly from `<tool_call>` to the
  name's `>`, hides only the EXTERNAL deliveries' own text and replayed untrusted turns,
  credits them no attention and raises no denial, demotes only once the arguments
  attend the result, follows `splice` and `trunc`, and changes nothing when off),
  a result or a user message spelling ChatML that neither closes its turn nor calls a
  tool, trusted results on ring 2 (no demotion, no floor in either gate mode, a floor
  an earlier result raised kept, the host's `trusted` checked against the call,
  replayed history), the frame guard, the parser, the sampler against the
  JavaScript one, and the same journal bytes for the same seed.

They run on scripted workers and need no model; the sampler test needs Node.
