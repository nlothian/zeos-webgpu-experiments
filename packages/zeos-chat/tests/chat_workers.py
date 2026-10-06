# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Workers for the chat machine's tests: a character vocabulary and scripted replies.

``ScriptedChatWorker`` has the reserved pieces of a Qwen3.5 export, one piece per
printable ASCII character, a newline and a tab, and any extra pieces a test names. It
tokenizes one id per character and parses no special token, as ``encodePlain`` does.
Its replies are one flat tape: each reply in order, spelled greedily from the longest
pieces (so ``<tool_call>`` is the added token), followed by ``<|im_end|>`` unless it
ends in a tool call, where the machine stops asking. Blocks are one position each.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from zeos_browser.fake_worker import ModelInfo

from zeos_chat.chat_machine import sample_index

SPECIAL = (
    "<pad>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
    "<tool_response>",
    "</tool_response>",
    "<unk>",
)
PAD_ID, ENDOFTEXT_ID, IM_START_ID, IM_END_ID = 0, 1, 2, 3
UNK_ID = SPECIAL.index("<unk>")
CHARS = tuple(chr(c) for c in range(32, 127)) + ("\n", "\t")

#: An attention rule: given the context's ids and the step's allowed positions, the mass
#: per position, or None for a worker that cannot measure.
Attend = Callable[[Sequence[int], Sequence[bool]], list[float] | None]


def attend_nothing(ids: Sequence[int], allowed: Sequence[bool]) -> None:
    return None


def attend_uniformly(ids: Sequence[int], allowed: Sequence[bool]) -> list[float]:
    count = sum(allowed)
    return [1.0 / count if a else 0.0 for a in allowed]


def attend_first(ids: Sequence[int], allowed: Sequence[bool]) -> list[float]:
    """Everything on the first allowed position, which is the system prompt's."""
    first = min(i for i, a in enumerate(allowed) if a)
    return [1.0 if i == first else 0.0 for i in range(len(allowed))]


@dataclass(frozen=True, slots=True)
class ChatStep:
    tokenId: int
    attention: list[float] | None


@dataclass
class _Context:
    ids: list[int] = field(default_factory=list[int])
    tape: list[int] = field(default_factory=list[int])


class ScriptedChatWorker:
    def __init__(
        self,
        replies: Sequence[str],
        *,
        extra: Sequence[str] = (),
        attend: Attend = attend_nothing,
        control: Sequence[str] = ("<|endoftext|>", "<|im_start|>", "<|im_end|>"),
    ) -> None:
        self.vocab = SPECIAL + CHARS + tuple(p for p in extra if p not in SPECIAL + CHARS)
        self.ids = {p: i for i, p in enumerate(self.vocab)}
        self._chars = {c: self.ids[c] for c in CHARS}
        self._control = tuple(self.ids[p] for p in control)
        self._attend = attend
        self._tape: list[int] = []
        for reply in replies:
            self._tape += self.spell(reply)
            if not reply.endswith("</tool_call>"):
                self._tape.append(IM_END_ID)
        self.contexts: dict[str, _Context] = {}
        self.seen: list[Mapping[str, Any]] = []

    def spell(self, text: str) -> list[int]:
        """Greedy longest-piece spelling, which is how a reply's pieces are chosen."""
        out: list[int] = []
        longest = max(len(p) for p in self.vocab)
        at = 0
        while at < len(text):
            for size in range(min(longest, len(text) - at), 0, -1):
                piece = text[at : at + size]
                if piece in self.ids and self.ids[piece] != PAD_ID:
                    out.append(self.ids[piece])
                    at += size
                    break
            else:
                raise ValueError(f"no piece spells {text[at]!r}")
        return out

    def info(self) -> ModelInfo:
        return ModelInfo(
            blockSize=1,
            padId=PAD_ID,
            controlIds=self._control,
            eosId=IM_END_ID,
            vocabSize=len(self.vocab),
        )

    def tokenize(self, text: str) -> list[int]:
        return [self._chars.get(c, UNK_ID) for c in text]

    def piece(self, tokenId: int) -> str:
        return self.vocab[tokenId]

    def createContext(self, jobId: str) -> None:
        assert jobId not in self.contexts
        self.contexts[jobId] = _Context(tape=list(self._tape))

    def destroyContext(self, jobId: str) -> None:
        del self.contexts[jobId]

    def length(self, jobId: str) -> int:
        return len(self.contexts[jobId].ids)

    def append(self, jobId: str, ids: Sequence[int]) -> None:
        self.contexts[jobId].ids.extend(int(i) for i in ids)

    def truncate(self, jobId: str, n: int) -> None:
        del self.contexts[jobId].ids[n:]

    def fork(self, parentId: str, childId: str) -> None:
        parent = self.contexts[parentId]
        self.contexts[childId] = _Context(ids=list(parent.ids), tape=list(parent.tape))

    def text(self, jobId: str) -> str:
        return "".join(self.vocab[i] for i in self.contexts[jobId].ids)

    def _allowed(self, ctx: _Context, opts: Mapping[str, Any]) -> list[bool]:
        blocks = opts["allowedBlocks"]
        assert blocks is None or len(blocks) == len(ctx.ids)
        return [blocks is None or bool(blocks[i]) for i in range(len(ctx.ids))]

    def decodeStep(self, jobId: str, opts: Mapping[str, Any]) -> ChatStep:
        self.seen.append(opts)
        ctx = self.contexts[jobId]
        if not ctx.tape:
            raise RuntimeError(f"{jobId}: the script has no reply left")
        token_id = ctx.tape.pop(0)
        allowed = opts["allowedTokens"]
        if not allowed[token_id]:
            raise ValueError(f"the mask refuses the script's next piece {self.vocab[token_id]!r}")
        return ChatStep(tokenId=token_id, attention=self._attend(ctx.ids, self._allowed(ctx, opts)))


class SamplingWorker(ScriptedChatWorker):
    """Logits that are a hash of the context, so the choice is the sampler's: letters
    and spaces only, and an end of turn favoured once the turn has run for a while."""

    def __init__(self, *, turn_length: int = 30, **kwargs: Any) -> None:
        super().__init__([], **kwargs)
        self.turn_length = turn_length
        self._letters = frozenset(self.ids[c] for c in "abcdefghij ")

    def decodeStep(self, jobId: str, opts: Mapping[str, Any]) -> ChatStep:
        self.seen.append(opts)
        ctx = self.contexts[jobId]
        digest = hashlib.sha256(bytes(i % 256 for i in ctx.ids)).digest()
        since = len(ctx.ids) - max(i for i, t in enumerate(ctx.ids) if t == IM_START_ID)
        logits = [-1e9] * len(self.vocab)
        for k, token_id in enumerate(sorted(self._letters)):
            logits[token_id] = digest[k] / 32.0
        logits[IM_END_ID] = 20.0 if since > self.turn_length else -1e9
        sample = opts.get("sample")
        allowed = opts["allowedTokens"]
        if sample is None:
            token_id = max(
                (i for i in range(len(logits)) if allowed[i]), key=lambda i: (logits[i], -i)
            )
        else:
            token_id = sample_index(
                logits,
                allowed,
                temperature=sample["temperature"],
                top_k=int(sample["topK"]),
                u=sample["u"],
            )
        return ChatStep(tokenId=token_id, attention=None)


class ByteWorker(ScriptedChatWorker):
    """A ``ScriptedChatWorker`` that also has a token for each byte from 0x80 up, whose
    piece is U+FFFD, as a byte-level vocabulary's are. A reply character outside the
    vocabulary is spelled as its UTF-8 bytes, one token each."""

    def __init__(self, replies: Sequence[str], **kwargs: Any) -> None:
        self._byte_base = len(SPECIAL + CHARS) + len(kwargs.get("extra", ()))
        super().__init__([], **kwargs)
        self.vocab = self.vocab + ("�",) * 128
        for reply in replies:
            self._tape += self.spell(reply)
            if not reply.endswith("</tool_call>"):
                self._tape.append(IM_END_ID)

    def byte_id(self, byte: int) -> int:
        return self._byte_base + byte - 0x80

    def spell(self, text: str) -> list[int]:
        out: list[int] = []
        for run in re.findall(r"[\x00-\x7f]+|[^\x00-\x7f]", text):
            if run.isascii():
                out += super().spell(run)
            else:
                out += [self.byte_id(b) for b in run.encode("utf-8")]
        return out

    def pieceBytes(self, tokenId: int) -> bytes:
        if not self._byte_base <= tokenId < self._byte_base + 128:
            raise IndexError(f"{tokenId} is not a byte token")
        return bytes([tokenId - self._byte_base + 0x80])
