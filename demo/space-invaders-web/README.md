# Space Invaders in the browser

**Status: under construction.** This directory holds the scaffold of a port; nothing here
runs a game yet.

The port puts [`../space-invaders`](../space-invaders/) in a static web page: the ZEOS
kernel and the game run under Pyodide, and the pilot is served by
`Qwen3.5-4B-ZEOS-OPT` over WebGPU through the worker channel of
[`../coop-count-web`](../coop-count-web/). The game keeps real time, so a slow model falls
behind and the reflex preempts it, as in the native demo. A prompt-loop arm on the same
model is the comparison.

What exists:

- `src/zeos_space_invaders_web/contracts.py`: the Python seams between the pieces (the
  non-blocking worker, the machine, the prompt arm, the runners, frames and results), and
  `load_board`.
- [`CONTRACTS.md`](CONTRACTS.md): the JavaScript half (the channel's slot layout, the
  begin/poll/cancel decode step and its race rules, the page and worker messages) and who
  owns which file.
- `src/zeos_space_invaders_web/boards/`: byte-for-byte copies of the native
  `settings_default.json` (12x16, 0.5 s tick) and `settings_ablation.json` (9x8, 0.2 s).

The architecture -- one threadless loop in the Pyodide worker owning both the game clock
and the kernel, with the model's decode steps begun, polled and cancelled over the
SharedArrayBuffer -- is set out in `CONTRACTS.md`.

```bash
uv run pytest demo/space-invaders-web
```
