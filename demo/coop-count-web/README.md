# coop-count-web — the coop-count demo, browser-safe

A member of the ZEOS uv workspace. It holds the parts of the coop-count demo that a
browser can run under Pyodide, which means nothing native: the package depends on `zeos`
and nothing else, so its wheel installs where `llama-cpp-python`, `huggingface-hub` and
`anthropic` cannot.

- `zeos_coop_count_web.js_machine.JsMachine` — a `MachineBackend` whose model is a
  JavaScript object (a `ZeosModelWorker`) reached through Pyodide. The interface, and
  what `JsMachine` asks of an implementation, is in the module docstring.
- `zeos_coop_count_web.fake_worker.FakeWorker` — a `ZeosModelWorker` written in Python
  with no model, which plays the case's tapes, so `JsMachine` runs under CPython.
- `zeos_coop_count_web.token_mask` — the syscall ABI as a per-step `allowedTokens`
  mask, the browser's stand-in for the GBNF grammar the llama seat uses.
- `zeos_coop_count_web.live.LiveRun` — the `zeos-count` run loop cut into single turns,
  so keypresses can arrive between ticks.

```bash
uv sync --all-packages
cd demo/coop-count-web
uv run pytest
```
