# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate ``src/zeos/core/_fold_table.py``, the vendored table ``framing.fold`` reads.

    uv run python -m tests.fold.regenerate

The kernel's spoof fold must give the same answer under every Python that runs it --
CPython 3.12 (Unicode 15.0), 3.13 (15.1) and Pyodide's 3.14 (16.0) -- or the same run
journals a spoof alarm under one and not the other. So ``framing`` calls nothing in
``unicodedata`` and no ``str`` method whose answer depends on the Unicode version: the
fold is this table, generated once from this Python's Unicode data and committed with
the version it was made from (``UNICODE_VERSION``).

A character's fold, as generated here:

1. **Dropped** (folds to nothing) if a reader does not see it: a Default_Ignorable code
   point (``DEFAULT_IGNORABLE``, from DerivedCoreProperties.txt, unchanged since Unicode
   11: soft hyphen, combining grapheme joiner, Hangul fillers, the zero-width and
   bidirectional controls, variation selectors, tags), any other format character
   (category Cf), a control character other than the tab, line feed, vertical tab, form
   feed and carriage return (so the information separators U+001C-U+001F and NEL U+0085,
   which split a word for ``str.split`` but not for a reader, are dropped), and a
   combining mark (Mn, Me), so an accent stacked on a letter does not hide it.
2. Otherwise its compatibility decomposition (NFKD), with combining marks dropped, then
   each character upper-cased and mapped through the look-alikes below
   (``LOOK_ALIKES``), applied both before upper-casing (for lower-case shapes) and
   after.

The table keeps a character only when its fold differs from it and is empty or holds an
ASCII character: everything the frame patterns can match is ASCII, so any other
character may stand for itself. The look-alikes are those of Unicode's confusables
(UTS #39) that read as the letters, brackets and slash of the folded frame names, and
the Latin letters a frame name could be confused with, written out here from the
scripts that carry them: Cyrillic, Greek, Coptic, Armenian, Cherokee, Lisu, Runic,
Canadian Syllabics and Latin small capitals. It is a curated subset, not the whole of
confusables.txt; a look-alike missing from it is a gap in the advisory alarm, not in
what a job may do.
"""

from __future__ import annotations

import sys
import unicodedata
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

#: Characters that read as an ASCII letter, bracket or slash, by what they read as.
LOOK_ALIKES: dict[str, str] = {
    # Cyrillic, either case.
    **dict.fromkeys("АаⲀ", "A"),
    **dict.fromkeys("ВвⲂʙ", "B"),
    **dict.fromkeys("СсϹϲⲤⲥ", "C"),
    **dict.fromkeys("ԁ", "D"),
    **dict.fromkeys("ЕеЀЁѐё", "E"),
    **dict.fromkeys("Һһ", "H"),
    **dict.fromkeys("ІіЇї", "I"),
    **dict.fromkeys("Јј", "J"),
    **dict.fromkeys("Кк", "K"),
    **dict.fromkeys("ӀӏǀƖ", "L"),
    **dict.fromkeys("Мм", "M"),
    **dict.fromkeys("п", "N"),
    **dict.fromkeys("Оо", "O"),
    **dict.fromkeys("Рр", "P"),
    **dict.fromkeys("Ԛԛ", "Q"),
    **dict.fromkeys("г", "R"),
    **dict.fromkeys("Ѕѕ", "S"),
    **dict.fromkeys("Тт", "T"),
    **dict.fromkeys("Ԝԝ", "W"),
    **dict.fromkeys("Хх", "X"),
    **dict.fromkeys("Ууү Ү".replace(" ", ""), "Y"),
    # Greek.
    **dict.fromkeys("Αα", "A"),
    **dict.fromkeys("Ββ", "B"),
    **dict.fromkeys("Εε", "E"),
    **dict.fromkeys("Ϝϝ", "F"),
    **dict.fromkeys("Ηη", "N"),
    **dict.fromkeys("Ιι", "I"),
    **dict.fromkeys("Κκϰ", "K"),
    **dict.fromkeys("Μ", "M"),
    **dict.fromkeys("Νν", "N"),
    **dict.fromkeys("Οο", "O"),
    **dict.fromkeys("Ρρ", "P"),
    **dict.fromkeys("Ττ", "T"),
    **dict.fromkeys("Υυ", "U"),
    **dict.fromkeys("Χχ", "X"),
    **dict.fromkeys("Ζ", "Z"),
    # Coptic capitals (their small letters upper-case to these).
    "Ⲉ": "E",
    "Ⲓ": "I",
    "Ⲕ": "K",
    "Ⲙ": "M",
    "Ⲛ": "N",
    "Ⲟ": "O",
    "Ⲣ": "P",
    "Ⲧ": "T",
    "Ⲩ": "Y",
    "Ⲭ": "X",
    # Armenian.
    "Լ": "L",  # Լ
    "Ս": "U",  # Ս
    "ս": "U",  # ս
    "Տ": "S",  # Տ
    "Օ": "O",  # Օ
    "օ": "O",  # օ
    "ո": "N",  # ո
    "հ": "H",  # հ
    # Cherokee capitals (their small letters upper-case to these).
    "Ꭰ": "D",
    "Ꭱ": "R",
    "Ꭲ": "T",
    "Ꭺ": "A",
    "Ꭻ": "J",
    "Ꭼ": "E",
    "Ꮃ": "W",
    "Ꮇ": "M",
    "Ꮋ": "H",
    "Ꮐ": "G",
    "Ꮓ": "Z",
    "Ꮢ": "R",
    "Ꮩ": "V",
    "Ꮪ": "S",
    "Ꮮ": "L",
    "Ꮯ": "C",
    "Ꮲ": "P",
    "Ꮶ": "K",
    "Ᏼ": "B",
    # Lisu: the letters drawn as upright Latin capitals.
    "ꓐ": "B",
    "ꓑ": "P",
    "ꓓ": "D",
    "ꓔ": "T",
    "ꓖ": "G",
    "ꓗ": "K",
    "ꓙ": "J",
    "ꓚ": "C",
    "ꓜ": "Z",
    "ꓝ": "F",
    "ꓟ": "M",
    "ꓠ": "N",
    "ꓡ": "L",
    "ꓢ": "S",
    "ꓣ": "R",
    "ꓦ": "V",
    "ꓧ": "H",
    "ꓪ": "W",
    "ꓫ": "X",
    "ꓬ": "Y",
    "ꓮ": "A",
    "ꓰ": "E",
    "ꓲ": "I",
    "ꓳ": "O",
    "ꓴ": "U",
    # Runic.
    "ᚱ": "R",
    "ᛁ": "I",
    "ᛒ": "B",
    "ᛕ": "K",
    "ᛖ": "M",
    # Canadian Syllabics.
    "ᐯ": "V",
    "ᑌ": "U",
    "ᑎ": "N",
    "ᑕ": "C",
    "ᑭ": "P",
    "ᒍ": "J",
    "ᒪ": "L",
    "ᔕ": "S",
    "ᕼ": "H",
    "ᖇ": "R",
    "ᗅ": "A",
    "ᗪ": "D",
    "ᗰ": "M",
    "ᗷ": "B",
    # Latin small capitals, which no case mapping reaches.
    "ᴀ": "A",
    "ᴄ": "C",
    "ᴅ": "D",
    "ᴇ": "E",
    "ꜰ": "F",
    "ɢ": "G",
    "ʜ": "H",
    "ɪ": "I",
    "ᴊ": "J",
    "ᴋ": "K",
    "ʟ": "L",
    "ᴍ": "M",
    "ɴ": "N",
    "ᴏ": "O",
    "ᴘ": "P",
    "ʀ": "R",
    "ꜱ": "S",
    "ᴛ": "T",
    "ᴜ": "U",
    "ᴠ": "V",
    "ᴡ": "W",
    "ʏ": "Y",
    "ᴢ": "Z",
    # Other Latin.
    "ı": "I",  # dotless i
    "ȷ": "J",  # dotless j
    # Brackets and the slash.
    **dict.fromkeys("‹〈⟨❬❮˂ᐸ⧼", "<"),
    **dict.fromkeys("›〉⟩❭❯˃ᐳ⧽", ">"),
    **dict.fromkeys("∕⁄⧸╱⟋〳", "/"),
}


def _ignorable(codepoint: int) -> bool:
    char = chr(codepoint)
    if any(lo <= codepoint <= hi for lo, hi in DEFAULT_IGNORABLE):
        return True
    category = unicodedata.category(char)
    if category == "Cc":
        return char not in VISIBLE_CONTROLS
    return category in ("Cf", "Mn", "Me")


def _look(char: str) -> str:
    return LOOK_ALIKES.get(char, char)


def fold_of(codepoint: int) -> str:
    """One character's fold, as the module docstring defines it."""
    if 0xD800 <= codepoint <= 0xDFFF or _ignorable(codepoint):
        return ""
    out: list[str] = []
    for char in unicodedata.normalize("NFKD", chr(codepoint)):
        if _ignorable(ord(char)):
            continue
        for upper in _look(char).upper():
            out.extend(_look(c) for c in upper.upper())
    return "".join(out)


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


def render() -> str:
    dropped, mapped = build()
    lines = [
        "# SPDX-License-Identifier: AGPL-3.0-only",
        "# Copyright (C) 2026 Metacognition AI",
        "#",
        "# This source code is licensed under the AGPL-3.0-only licence found in the",
        "# LICENSE file in the root directory of this source tree.",
        "",
        '"""The spoof fold\'s table (``zeos.core.framing.fold``). Generated; do not edit.',
        "",
        "    uv run python -m tests.fold.regenerate",
        "",
        "``tests/fold/regenerate.py`` says what a fold is. ``DROPPED`` are the code-point",
        "ranges that fold to nothing. ``MAPPED`` holds every other character whose fold",
        "differs from it and holds an ASCII character, as ``code:fold`` entries separated by",
        "``;``, the code point and the fold's code points in hexadecimal, the latter",
        "separated by ``,``.",
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
    return "\n".join(lines)


def main() -> None:
    was = TARGET.read_text("utf-8") if TARGET.is_file() else ""
    now = render()
    TARGET.write_text(now, encoding="utf-8")
    verdict = "unchanged" if was == now else "CHANGED"
    print(f"{TARGET.name}: Unicode {unicodedata.unidata_version}, {verdict}")


if __name__ == "__main__":
    main()
