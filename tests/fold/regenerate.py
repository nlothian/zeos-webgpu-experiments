# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate ``src/zeos/core/_fold_table.py``, the vendored table ``framing.fold`` reads.

    uv run --frozen python -m tests.fold.regenerate

The kernel's spoof fold must give the same answer under every Python that runs it --
CPython 3.12 (Unicode 15.0), 3.13 (15.1) and Pyodide's 3.14 (16.0) -- or the same run
journals a spoof alarm under one and not the other. So ``framing`` calls nothing in
``unicodedata`` and no ``str`` method whose answer depends on the Unicode version: the
fold is this table, generated once from this Python's Unicode data and Unicode's
confusables of the same version, and committed with that version (``UNICODE_VERSION``).
The generator refuses to run under a Python of another Unicode version.

The look-alikes come from UTS #39's confusables.txt, version 15.0.0, committed beside
this file (``confusables-15.0.0.txt``, from ``CONFUSABLES_URL``; its hash is checked
against ``CONFUSABLES_SHA256`` before it is read). It is Unicode data, redistributed under
the Unicode terms of use its header names; keep the file whole, header and all.

A character's fold, as generated here:

1. **Dropped** (folds to nothing) if a reader does not see it: a Default_Ignorable code
   point (``DEFAULT_IGNORABLE``, from DerivedCoreProperties.txt, unchanged since Unicode
   11: soft hyphen, combining grapheme joiner, Hangul fillers, the zero-width and
   bidirectional controls, variation selectors, tags), any other format character
   (category Cf), a control character other than the tab, line feed, vertical tab, form
   feed and carriage return (so the information separators U+001C-U+001F and NEL U+0085,
   which split a word for ``str.split`` but not for a reader, are dropped), and a
   combining mark (Mn, Me), so an accent stacked on a letter does not hide it.
2. Otherwise its compatibility decomposition (NFKD), with those dropped again.
3. Each character of that, folded on its own:

   - **ASCII is only upper-cased.** confusables.txt maps eight ASCII characters
     (``0`` to ``O``; ``1``, ``I`` and ``|`` to ``l``; ``m`` to ``rn``; and the quotes
     and ``%``), but the frame names and their JSON-escaped openings are written in
     ASCII, and folding ASCII into ASCII would change what they are: ``\\u003c`` would
     fold to ``\\UOO3C``, ``RESUME`` to ``RESURNE``, ``<FAULT|`` would run on. So
     ordinary ASCII text folds exactly as it did with case alone, and an ASCII
     confusion (``<kerne1>``) is outside the fold.
   - **Any other character is read by its confusables prototype**, case-sensitively,
     as UTS #39 maps it (``_read``): the prototype decomposed, ``rn`` read as the ``m``
     it stands for (``READS_AS``), any character of it other than ASCII read through
     ``SUPPLEMENT``, and then upper-cased. The prototype is used only when all of that
     is ASCII. So the Cyrillic ``а`` reads as ``A``, the Greek ``ν`` as ``V``, ``ɱ`` as
     ``M``, and the Cyrillic ``І``, the Hebrew paseq ``׀`` and the Arabic-Indic one ``١``,
     whose prototype is ``l``, as ``L``. A letter whose prototype would read as a
     tag's ``<``, ``>`` or ``/`` is not read so (``TAG_PUNCTUATION``).
   - **Otherwise upper-cased**, and each character of that read the same way, so the
     Cyrillic ``к``, whose own prototype is the kra ``ĸ``, reads as ``K`` through its
     capital ``К``.
   - **Otherwise left as it is.**

   UTS #39's skeleton is NFD, map, NFD, all case-sensitive. Here case comes after the
   map, because the map is case-sensitive (``η`` is drawn as an ``n`` and ``Η`` as an
   ``H``), and upper-casing the prototype is what makes the fold match whatever the
   case; the capital is tried only for a character whose own prototype is not ASCII.

``SUPPLEMENT`` holds what the curated look-alikes the fold had before (6a27b78) caught
and confusables.txt does not map to ASCII: the Latin small capitals, a few letters of
Cyrillic, Coptic, Armenian, Runic and Canadian Syllabics, and the angle-bracket
ornaments.

Every confusables.txt entry maps one code point, so the per-character table holds
every one; a prototype of several characters (``æ`` to ``ae``, ``ʦ`` to ``ts``) becomes
a fold of several. A confusion of several characters with one (``rn`` with ``m``, in
ASCII) cannot be a per-character fold and is not one: see ``READS_AS``.

The table keeps a character only when its fold differs from it and is empty or holds an
ASCII character: everything the frame patterns can match is ASCII, so any other
character may stand for itself. A look-alike missing from it is a gap in the advisory
alarm, not in what a job may do.

Two more sets tell ``framing`` where a reader sees a frame name end (its docstring,
"Where a name ends"). ``ACCENTED`` are the **Latin letters with diacritics**: the
characters whose canonical decomposition (NFD) is an ASCII letter then one or more
combining marks, and whose fold is that letter, upper-cased (``é``, ``ĺ``, ``ü``, ``Å``;
not the Kelvin sign, whose decomposition is a bare ``K``, nor ``ł`` or ``ø``, which have
none). ``ACCENTS`` are the combining marks those decompositions hold (U+0300 grave,
U+0301 acute, U+0308 diaeresis, ...): visible, so they belong to the letter they sit on,
where the other characters the fold drops are invisible.
"""

from __future__ import annotations

import hashlib
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / "src" / "zeos" / "core" / "_fold_table.py"

#: Default_Ignorable_Code_Point, from DerivedCoreProperties.txt.
DEFAULT_IGNORABLE: tuple[tuple[int, int], ...] = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)

#: Whitespace a reader sees: kept, though ``str.split`` treats more as whitespace.
VISIBLE_CONTROLS = frozenset("\t\n\v\f\r")

#: Unicode's confusables (UTS #39), committed beside this file with its notice intact.
CONFUSABLES = Path(__file__).with_name("confusables-15.0.0.txt")
CONFUSABLES_URL = "https://www.unicode.org/Public/security/15.0.0/confusables.txt"
CONFUSABLES_SHA256 = "2b10130885c3370b101c52d7baedc452ab7f0e257b86c1e52ee657ecfc29ce64"
CONFUSABLES_VERSION = "15.0.0"

#: A prototype that stands for an ASCII letter it is not spelled with. Confusables
#: gives ``m`` the prototype ``rn``, so a character drawn as an ``m`` (``ɱ``, ``₥``) has
#: it too; the fold reads it as the ``M`` a frame name is spelled with.
READS_AS: dict[str, str] = {"rn": "m"}

#: What opens or closes a tag. Confusables maps a few letters to these -- the
#: hiragana ``く`` and ``ぐ`` and the katakana ``ノ`` among them -- and a letter in running
#: text is read as a letter: Japanese text often runs a Latin word straight on after
#: one, and ``書くKernel`` is no tag. So a letter's prototype is not used when it holds
#: one of these; the two such letters the curated look-alikes caught, the Canadian
#: Syllabics ``ᐸ`` and ``ᐳ``, are in ``SUPPLEMENT``.
TAG_PUNCTUATION = frozenset("</>")

#: What the curated look-alikes of 6a27b78 caught and confusables.txt does not map to
#: an ASCII prototype, kept so that no look-alike the fold caught before is lost. Each
#: is read as the folded letter or bracket it is drawn as. An entry is read before the
#: character's own prototype, and in place of a prototype's character, so ``τ`` and
#: ``т``, whose prototype is the small capital ``ᴛ``, read as ``T``, and ``〈`` and ``⟨``,
#: whose prototype is the ornament ``❬``, read as ``<``.
SUPPLEMENT: dict[str, str] = {
    # Latin small capitals: confusables maps them to no ASCII letter (several are the
    # prototypes of Greek, Cyrillic and Cherokee small letters), but they read as the
    # capitals they are drawn as.
    "\u1d00": "A",  # ᴀ
    "\u0299": "B",  # ʙ
    "\u1d05": "D",  # ᴅ
    "\u1d07": "E",  # ᴇ
    "\ua730": "F",  # ꜰ
    "\u0262": "G",  # ɢ
    "\u029c": "H",  # ʜ
    "\u1d0a": "J",  # ᴊ
    "\u1d0b": "K",  # ᴋ, which confusables maps to the kra ĸ
    "\u029f": "L",  # ʟ
    "\u1d0d": "M",  # ᴍ, which confusables maps to the turned ʍ
    "\u0274": "N",  # ɴ
    "\u1d18": "P",  # ᴘ
    "\u1d1b": "T",  # ᴛ
    # Cyrillic.
    "\u04ba": "H",  # Һ shha
    "\u051a": "Q",  # Ԛ qa
    "\u043f": "N",  # п pe, which confusables maps to the Greek π
    "\u04cf": "L",  # ӏ small palochka: confusables reads it as i (by way of the dotless
    # ı); it is drawn as the capital Ӏ is, which confusables reads as l.
    # Coptic capitals.
    "\u2c80": "A",  # Ⲁ alfa
    "\u2c82": "B",  # Ⲃ vida
    "\u2c88": "E",  # Ⲉ eie, which confusables maps to the barred Ꞓ
    # Armenian.
    "\u053c": "L",  # Լ liwn
    # Runic.
    "\u16b1": "R",  # ᚱ raido
    "\u16d2": "B",  # ᛒ berkanan
    # Canadian Syllabics.
    "\u144e": "N",  # ᑎ ti, which confusables maps to the Armenian Ո
    "\u1455": "C",  # ᑕ ta
    "\u1515": "S",  # ᔕ sha
    # Other Latin.
    "\u0237": "J",  # ȷ dotless j
    # Canadian Syllabics letters drawn as angle brackets (see ``TAG_PUNCTUATION``).
    "\u1438": "<",  # ᐸ pa
    "\u1433": ">",  # ᐳ po
    # Angle-bracket ornaments, the prototypes of 〈 〉 ⟨ ⟩, and the curved brackets.
    "\u276c": "<",  # ❬
    "\u276d": ">",  # ❭
    "\u29fc": "<",  # ⧼
    "\u29fd": ">",  # ⧽
}


def _confusables() -> dict[str, str]:
    """confusables.txt's mappings, each source character to its prototype, after
    checking the file is the one this generator was written for."""
    data = CONFUSABLES.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != CONFUSABLES_SHA256:
        raise SystemExit(f"{CONFUSABLES.name}: sha256 {digest}, expected {CONFUSABLES_SHA256}")
    mappings: dict[str, str] = {}
    for line in data.decode("utf-8").splitlines():
        fields = line.split("#", 1)[0].split(";")
        if len(fields) < 3:
            continue
        (source,) = fields[0].split()
        mappings[chr(int(source, 16))] = "".join(chr(int(c, 16)) for c in fields[1].split())
    return mappings


PROTOTYPES = _confusables()


def _ignorable(codepoint: int) -> bool:
    char = chr(codepoint)
    if any(lo <= codepoint <= hi for lo, hi in DEFAULT_IGNORABLE):
        return True
    category = unicodedata.category(char)
    if category == "Cc":
        return char not in VISIBLE_CONTROLS
    return category in ("Cf", "Mn", "Me")


def _decomposed(text: str) -> str:
    """``text``'s compatibility decomposition, with what a reader does not see dropped."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not _ignorable(ord(c)))


def _read(char: str) -> str | None:
    """What a character other than ASCII reads as in folded ASCII: its ``SUPPLEMENT``
    entry, or else its confusables prototype, decomposed, read through ``READS_AS``
    and upper-cased, with any character of it other than ASCII read through
    ``SUPPLEMENT``. ``None`` when neither gives ASCII."""
    if char in SUPPLEMENT:
        return SUPPLEMENT[char]
    target = _decomposed(PROTOTYPES.get(char, ""))
    target = READS_AS.get(target, target)
    parts = [c.upper() if c.isascii() else SUPPLEMENT.get(c) for c in target]
    if not parts or None in parts:
        return None
    read = "".join(p for p in parts if p is not None)
    if unicodedata.category(char).startswith("L") and not TAG_PUNCTUATION.isdisjoint(read):
        return None
    return read


def _fold_char(char: str) -> str:
    """One character of a decomposition, folded: ASCII upper-cased; any other read as
    it is drawn (``_read``), or else upper-cased and each character of that read."""
    if char.isascii():
        return char.upper()
    read = _read(char)
    if read is not None:
        return read
    upper = char.upper()
    parts = [c if c.isascii() else _read(c) for c in upper]
    if None in parts:
        return upper
    return "".join(p for p in parts if p is not None)


def fold_of(codepoint: int) -> str:
    """One character's fold, as the module docstring defines it."""
    if 0xD800 <= codepoint <= 0xDFFF or _ignorable(codepoint):
        return ""
    return "".join(_fold_char(c) for c in _decomposed(chr(codepoint)))


def build() -> tuple[list[tuple[int, int]], dict[int, str]]:
    dropped: list[tuple[int, int]] = []
    mapped: dict[int, str] = {}
    for codepoint in range(sys.maxunicode + 1):
        folded = fold_of(codepoint)
        if folded == chr(codepoint):
            continue
        if not folded:
            if dropped and dropped[-1][1] == codepoint - 1:
                dropped[-1] = (dropped[-1][0], codepoint)
            else:
                dropped.append((codepoint, codepoint))
        elif any(ord(c) < 0x80 for c in folded):
            mapped[codepoint] = folded
    return dropped, mapped


def accents_of(codepoint: int) -> str | None:
    """The combining marks of a Latin letter with diacritics (``ACCENTED``), or ``None``
    when the character is not one."""
    decomposed = unicodedata.normalize("NFD", chr(codepoint))
    base, marks = decomposed[:1], decomposed[1:]
    if codepoint < 0x80 or not marks or not (base.isascii() and base.isalpha()):
        return None
    if not all(unicodedata.category(m) == "Mn" and _ignorable(ord(m)) for m in marks):
        return None
    return marks if fold_of(codepoint) == base.upper() else None


def _ranges(codepoints: Iterable[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for codepoint in sorted(codepoints):
        if ranges and ranges[-1][1] == codepoint - 1:
            ranges[-1] = (ranges[-1][0], codepoint)
        else:
            ranges.append((codepoint, codepoint))
    return ranges


def accented() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """``ACCENTED`` and ``ACCENTS``, as code-point ranges."""
    letters: list[int] = []
    accents: set[int] = set()
    for codepoint in range(sys.maxunicode + 1):
        marks = accents_of(codepoint)
        if marks is not None:
            letters.append(codepoint)
            accents.update(map(ord, marks))
    return _ranges(letters), _ranges(accents)


def render() -> str:
    dropped, mapped = build()
    letters, accents = accented()
    lines = [
        "# SPDX-License-Identifier: AGPL-3.0-only",
        "# Copyright (C) 2026 Metacognition AI",
        "#",
        "# This source code is licensed under the AGPL-3.0-only licence found in the",
        "# LICENSE file in the root directory of this source tree.",
        "",
        '"""The spoof fold\'s table (``zeos.core.framing.fold``). Generated; do not edit.',
        "",
        "    uv run --frozen python -m tests.fold.regenerate",
        "",
        "``tests/fold/regenerate.py`` says what a fold is: what a reader does not see",
        f"dropped, NFKD, and look-alikes read through Unicode's confusables {CONFUSABLES_VERSION}",
        "(UTS #39) and a small supplement, upper-cased. ``DROPPED`` are the code-point",
        "ranges that fold to nothing. ``MAPPED`` holds every other character whose fold",
        "differs from it and holds an ASCII character, as ``code:fold`` entries separated by",
        "``;``, the code point and the fold's code points in hexadecimal, the latter",
        "separated by ``,``. ``ACCENTED`` are the ranges of the Latin letters with",
        "diacritics, and ``ACCENTS`` those of the combining marks they decompose to, which",
        "tell ``framing`` where a reader sees a name end.",
        '"""',
        "",
        f'UNICODE_VERSION = "{unicodedata.unidata_version}"',
        "",
        "DROPPED: tuple[tuple[int, int], ...] = (",
    ]
    lines += [f"    (0x{lo:X}, 0x{hi:X})," for lo, hi in dropped]
    lines += [")", "", "MAPPED = ("]
    entries = [
        f"{cp:X}:" + ",".join(f"{ord(c):X}" for c in folded) for cp, folded in mapped.items()
    ]
    row = ""
    for entry in entries:
        if len(row) + len(entry) > 84:
            lines.append(f'    "{row}"')
            row = ""
        row += entry + ";"
    lines.append(f'    "{row.rstrip(";")}"')
    lines += [")", ""]
    for name, ranges in (("ACCENTED", letters), ("ACCENTS", accents)):
        lines += [f"{name}: tuple[tuple[int, int], ...] = ("]
        lines += [f"    (0x{lo:X}, 0x{hi:X})," for lo, hi in ranges]
        lines += [")", ""]
    return "\n".join(lines)


def main() -> None:
    if unicodedata.unidata_version != CONFUSABLES_VERSION:
        raise SystemExit(
            f"Unicode {unicodedata.unidata_version} here; the fold is generated from "
            f"{CONFUSABLES_VERSION}, with confusables.txt of that version"
        )
    was = TARGET.read_text("utf-8") if TARGET.is_file() else ""
    now = render()
    TARGET.write_text(now, encoding="utf-8")
    verdict = "unchanged" if was == now else "CHANGED"
    print(f"{TARGET.name}: Unicode {unicodedata.unidata_version}, {verdict}")


if __name__ == "__main__":
    main()
