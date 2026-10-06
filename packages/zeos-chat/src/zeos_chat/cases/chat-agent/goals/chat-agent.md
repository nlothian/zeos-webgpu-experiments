---
name: chat-agent
priority: 50
# Resident from boot and released by the first message on chat.user; after that every
# turn ends in a read of chat.user, so one job holds the whole conversation.
pinned: true
# A refused write is answered with a FAULT notice and the job carries on, which is what
# lets the host settle a refused tool call -- approved and run by the host, or declined
# -- by delivering on tools.results.
on_fault: retry
integrity:
  start: 2
  # The default, written down because it is what makes this job legal: it holds a
  # capability at min_integrity 2 and reads a ring-3 pipe, which the confused-deputy lint
  # refuses with `static`. Once the job attends a tool result past theta_read it falls
  # to 3, and tools.effect refuses it from then on.
  dynamics: low-watermark
pipes:
  stdin: chat.user
  stdout: chat.out
  results: tools.results
  results_trusted: tools.results.trusted
  read: tools.read
  effect: tools.effect
  history: chat.history
  history_trusted: chat.history.trusted
capabilities:
  # Reading tools are open to a job at any integrity.
  - pipe: tools.read
    min_integrity: 3
  # Side effects need a job no dirtier than a person's own words.
  - pipe: tools.effect
    min_integrity: 2
  - pipe: chat.out
    min_integrity: 3
context:
  window: 16384
  stub_budget: 512
  min_span_age: 256
---

This body is a placeholder. The host replaces it with the system prompt and the tool
declarations before the run opens (`zeos_coop_count_web.chat.open_chat`, `system_prompt`).
