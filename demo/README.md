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

A demo carries its own cases. The kernel does not: see the note in the top-level README
about why no application-specific case ships in `src/`.
