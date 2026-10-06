# Demos

Demonstrations built on the zeos kernel live here. Each has its own `pyproject.toml` and
is a member of the repository's uv workspace, so `uv sync --all-packages` at the root
installs the kernel and every demo into one venv, and a clone is enough to run one.

| | |
|---|---|
| [`counter/`](counter/) | The smallest thing the kernel will run: jobs that count to five, spawn a successor and exit. Works with a scripted tape, the Claude API, and a local llama.cpp Qwen. A kernel journal can be produced with or without a model in the loop. |
| [`space-invaders/`](space-invaders/) | A real-time game a served model is too slow to play. A deliberating job asks the model and is preempted mid-sentence by a reflex that needs no forward pass, then resumes with the kernel's diff of what moved. Three drivers over one game — a human, a prompt loop, a descriptor tree — so the comparison is one architecture against another over a fixed world, and the case states its own success criteria for the journal to be judged against. |
| [`coop-count/`](coop-count/) | A real `MachineBackend` over llama.cpp, and two counting agents that hand a baton back and forth. The long-form tutorial builds the backend from nothing, and the same application appears twice — once driven by blocking pipe reads, once by vectors. |
| [`coop-count-web/`](coop-count-web/) | The kernel and the coop-count tape case in a static web page under Pyodide, with the journal streaming, the debugger, and a keypress for the interrupt. Its `JsMachine` hands token-level work to a JavaScript worker object, proved with a stub worker; a run in the browser writes the same journal bytes as under CPython. |
| [`space-invaders-web/`](space-invaders-web/) | `space-invaders` in a static web page: the kernel, the game and a wall-clock loop under Pyodide, and `Qwen3.5-4B-ZEOS-OPT` on WebGPU behind a non-blocking, cancellable decode channel. Both players run on the in-browser model -- the descriptor tree, whose reflex preempts the pilot mid-step, and the prompt loop -- on the default (12×16, 0.5 s) and ablation (9×8, 0.2 s) boards, with the case's criteria, the journal and the debugger on the page. Tested under CPython with a stub worker, under Pyodide in Node over the real channel, and on the 4B in headed Chrome on WebGPU (`tests/si_webgpu.mjs`), the only backend the model runs on. |

A demo carries its own cases. The kernel does not: see the note in the top-level README
about why no application-specific case ships in `src/`.
