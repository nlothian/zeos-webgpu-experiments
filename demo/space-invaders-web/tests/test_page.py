# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The page's Python half, what build.py puts beside it, and the page's own files."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, cast, get_type_hints

import pytest
from zeos_browser import model_source

from zeos_space_invaders_web import contracts, page
from zeos_space_invaders_web.contracts import (
    ARMS,
    BOARDS,
    Arm,
    BoardName,
    DecisionRecord,
    Frame,
    LocalStop,
    OpenRun,
    RunResult,
)

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
COOP = REPO / "demo" / "coop-count-web"
BROWSER = REPO / "packages" / "zeos-browser"
WEB = HERE / "web"


class FastClock:
    """A ``Clock`` whose sleep advances time at once, so a 600-tick run takes no time."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        self.slept += seconds


def fake_run(
    arm: Arm, board: BoardName, seed: int | None = 7, **kwargs: Any
) -> tuple[RunResult, list[Frame]]:
    frames: list[Frame] = []

    def sink(frame: Frame) -> None:
        frames.append(frame)

    run = page.open_run(arm, board, seed, None, on_frame=sink, clock=FastClock(), **kwargs)
    run.warm()
    try:
        return run.run(), frames
    finally:
        run.close()


# --- page.py ----------------------------------------------------------------------------


def test_open_run_is_the_contracts_open_run() -> None:
    opener: OpenRun = page.open_run
    assert callable(opener)
    assert page.RUN_BUILDERS == {"zeos": page.ZeosRun, "prompt": page.PromptRun}
    with pytest.raises(ValueError, match="unknown arm"):
        page.open_run("random", "default", 7, None)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="unknown board"):
        page.open_run("zeos", "huge", 7, None)  # pyright: ignore[reportArgumentType]


def _check_shape(value: object, hints: dict[str, Any], name: str) -> None:
    assert isinstance(value, dict), name
    assert set(value) == set(hints), f"{name} keys {sorted(value)} != {sorted(hints)}"  # pyright: ignore[reportUnknownArgumentType]


@pytest.mark.parametrize("board", BOARDS)
@pytest.mark.parametrize("arm", ARMS)
def test_fake_run_frames_conform_to_frame(arm: Arm, board: BoardName) -> None:
    result, frames = fake_run(arm, board)
    frame_hints = get_type_hints(Frame)
    decision_hints = get_type_hints(DecisionRecord)
    spec = contracts.load_board(board, 7)
    assert frames[0]["tick"] == 0, "the board before the first tick is drawn too"
    for frame in frames:
        _check_shape(frame, frame_hints, "Frame")
        json.dumps(frame)  # the frame message is structured-cloned JSON
        assert (frame["arm"], frame["board"]) == (arm, board)
        rows = frame["text"].split("\n")
        assert len(rows) == spec.rules.h and {len(r) for r in rows} == {4 * spec.rules.w}
        assert frame["kills"] == frame["score"] // 10
        for decision in frame["decisions"]:
            _check_shape(decision, decision_hints, "DecisionRecord")
            assert decision["lag_ticks"] == decision["tick_applied"] - decision["tick"]
            assert decision["by"] in {"pilot", "evade", "prompt"}
    ticks = [f["tick"] for f in frames]
    assert ticks == sorted(ticks)
    last = frames[-1]
    assert (result.ticks, result.lives, result.kills) == (
        last["tick"],
        last["lives"],
        last["kills"],
    )
    assert result.preemptions == last["preemptions"]
    assert last["over"] or result.ticks >= spec.max_steps
    moves = [d for f in frames for d in f["decisions"]]
    assert result.decisions == len(moves)
    assert result.reflexes == sum(1 for d in moves if d["by"] == "evade")
    if arm == "zeos":
        assert result.parse_rate is None
        assert [v["passed"] for v in result.verdicts] == [None] * 4, "a FakeRun judges nothing"
    else:
        assert result.reflexes == result.preemptions == 0 and result.verdicts == []
        assert result.parse_rate is None or 0.0 <= result.parse_rate <= 1.0


def test_the_zeos_fake_preempts_and_lags_like_the_plan_predicts() -> None:
    result, _ = fake_run("zeos", "default")
    assert result.preemptions > 0 and result.cancellations == result.preemptions
    assert 1.0 <= result.lag_ticks["mean"] <= 5.0
    assert result.lag_ticks["p50"] <= result.lag_ticks["p95"] <= result.lag_ticks["max"]


def test_a_fake_run_is_the_same_for_the_same_seed() -> None:
    a, frames_a = fake_run("zeos", "ablation", 3)
    b, frames_b = fake_run("zeos", "ablation", 3)
    assert a.to_json() == b.to_json() and frames_a == frames_b


def test_stop_ends_a_run_within_a_tick() -> None:
    stop = LocalStop()
    frames: list[Frame] = []

    def sink(frame: Frame) -> None:
        frames.append(frame)
        if frame["tick"] == 5:
            stop.set()

    clock = FastClock()
    run = page.open_run("prompt", "default", 7, None, on_frame=sink, stop=stop, clock=clock)
    result = run.run()
    assert result.ticks == 5 and frames[-1]["tick"] == 5
    stopped = LocalStop()
    stopped.set()
    run = page.open_run("zeos", "default", 7, None, stop=stopped, clock=FastClock())
    run.warm()
    assert run.run().ticks == 0


def test_catch_up_never_plays_past_max_steps() -> None:
    """A loop that falls far behind applies the missed ticks, but no more than are left."""

    class LateClock(FastClock):
        def sleep(self, seconds: float) -> None:
            super().sleep(seconds + 1.0)  # every wake is about five ticks late

    frames: list[Frame] = []

    def sink(frame: Frame) -> None:
        frames.append(frame)

    spec = dataclasses.replace(contracts.load_board("ablation", 7), max_steps=3)
    context = page.RunContext(
        arm="prompt",
        spec=spec,
        worker=None,
        stub=True,
        on_frame=sink,
        stop=None,
        clock=LateClock(),
    )
    result = page.FakeRun(context).run()
    assert result.ticks == 3 and frames[-1]["tick"] == 3
    assert result.catchup_ticks == 2


def test_stop_reports_moves_that_landed_after_the_last_tick() -> None:
    """A reply that lands between ticks is played at once and reported with the next
    frame; when Stop comes first, the run still reports it."""
    holder: list[page.FakeRun] = []

    class StopOnPending:
        def is_set(self) -> bool:
            return bool(holder and holder[0]._pending)  # pyright: ignore[reportPrivateUsage]

    frames: list[Frame] = []

    def sink(frame: Frame) -> None:
        frames.append(frame)

    run = page.open_run(
        "zeos", "default", 7, None, on_frame=sink, stop=StopOnPending(), clock=FastClock()
    )
    assert isinstance(run, page.FakeRun)
    holder.append(run)
    result = run.run()
    moves = [d for f in frames for d in f["decisions"]]
    assert moves and moves[-1]["by"] == "pilot"
    assert result.decisions == len(moves)
    assert not run._pending  # pyright: ignore[reportPrivateUsage]


def test_a_registered_builder_replaces_the_fake() -> None:
    seen: list[page.RunContext] = []

    def builder(context: page.RunContext) -> page.FakeRun:
        seen.append(context)
        return page.FakeRun(context)

    real = page.RUN_BUILDERS["prompt"]
    page.RUN_BUILDERS["prompt"] = builder
    try:
        # A run with no worker is a FakeRun whatever is registered; any worker reaches
        # the builder.
        assert isinstance(page.open_run("prompt", "ablation", None, None), page.FakeRun)
        assert not seen
        page.open_run("prompt", "ablation", None, cast(Any, object()), stub=True)
    finally:
        page.RUN_BUILDERS["prompt"] = real
    (context,) = seen
    assert (context.arm, context.spec.name, context.spec.seed, context.stub) == (
        "prompt",
        "ablation",
        None,
        True,
    )
    assert isinstance(context.clock, contracts.MonotonicClock)


def test_lag_stats() -> None:
    assert page.lag_stats([]) == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    assert page.lag_stats([3, 1, 2]) == {"mean": 2.0, "p50": 2.0, "p95": 3.0, "max": 3.0}
    assert page.lag_stats([float(i) for i in range(1, 21)])["p95"] == 19.0


def test_json_for_the_page() -> None:
    posted: list[str] = []
    result, frames = fake_run("zeos", "ablation")
    page.json_sink(posted.append)(frames[-1])
    assert json.loads(posted[0]) == frames[-1]

    finished = json.loads(page.finished_json(result))
    assert set(finished) == {"result", "verdicts", "journal", "payload"}
    assert finished["result"]["ticks"] == result.ticks
    assert "journal" not in finished["result"] and "verdicts" not in finished["result"]
    assert finished["verdicts"] == result.verdicts and finished["journal"] == result.journal
    payload = json.loads(finished["payload"])
    assert payload["structure"]["case"] == "space-invaders"
    assert "frames" not in payload, "no journal: the debugger draws the wiring only"

    described = json.loads(page.describe_json())
    assert described["boards"]["default"] == {
        "w": 12,
        "h": 16,
        "tick": 0.5,
        "max_steps": 600,
        "seed": 7,
        "lives": 3,
        "monsters": 15,
    }
    assert described["criteria"] == [
        "fire-reaches-a-handler",
        "the-reflex-displaces-deliberation",
        "the-dodge-lands-inside-the-budget",
        "the-pilot-is-told-the-ship-moved",
    ]


def test_the_debugger_draws_a_journal_when_there_is_one() -> None:
    """``payload_json`` decodes encoded kernel records, which is what ``RunResult.journal``
    holds (``ZeosDriver.journal()``); a coop-count run stands in for a kernel run here."""
    from zeos.journal.codec import encode_record
    from zeos_browser import page as coop_page

    case = REPO / "demo" / "coop-count" / "cases" / "coop-count-scripted"
    live = coop_page.open_run(str(case), "scripted")
    while not live.finished:
        live.step()
    journal = [encode_record(r.seq, r.event) for r in live.journal.records]
    payload = json.loads(page.payload_json(journal, case))
    assert payload["structure"]["case"] == "coop-count-scripted"
    assert payload["frames"]["count"] == len(journal)


# --- the page's files and build.py --------------------------------------------------------


def test_the_page_loads_the_pyodide_coop_count_web_and_zeos_browser_pin() -> None:
    """One Pyodide release across both pages and the Node harness (zeos-browser's npm)."""
    pin = r"cdn\.jsdelivr\.net/pyodide/v([0-9.]+)/full/pyodide\.mjs"
    (ours,) = re.findall(pin, (WEB / "si_worker.js").read_text(encoding="utf-8"))
    (theirs,) = re.findall(pin, (COOP / "web" / "pyodide_worker.js").read_text(encoding="utf-8"))
    assert ours == theirs
    npm = json.loads((BROWSER / "package.json").read_text(encoding="utf-8"))
    assert npm["devDependencies"]["pyodide"] == ours


def test_the_page_speaks_the_contracts_messages() -> None:
    worker = (WEB / "si_worker.js").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    for message in ("ready", "warming", "started", "frame", "decision", "finished", "error"):
        assert f'post("{message}"' in worker, message
        assert f"{message}:" in app, message
    for message in ("boot", "attachModel", "start"):
        assert f'send("{message}"' in app, message
    assert f"const CONTROL_STOP = {contracts.CONTROL_STOP};" in worker
    assert f"const CONTROL_STOP = {contracts.CONTROL_STOP};" in app
    assert f"const CONTROL_BYTES = {contracts.CONTROL_BYTES};" in app


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, dict[str, Any]]]:
    if shutil.which("uv") is None:
        pytest.skip("build.py builds the wheels with uv")
    build = _import_build()
    model = tmp_path_factory.mktemp("models") / "Tiny-ZEOS-OPT"
    model.mkdir()
    (model / "meta.json").write_text("{}", encoding="utf-8")
    dist = tmp_path_factory.mktemp("si") / "dist"
    yield dist, build.build(dist, model=model)


def _import_build() -> ModuleType:
    sys.path.insert(0, str(HERE))
    try:
        import build
    finally:
        sys.path.remove(str(HERE))
    return build


def test_build_assembles_everything_the_page_fetches(built: tuple[Path, dict[str, Any]]) -> None:
    dist, manifest = built
    build = _import_build()
    names: list[str] = manifest["wheels"]
    assert [n.split("-")[0] for n in names] == [
        "zeos",
        "zeos_browser",
        "zeos_space_invaders",
        "zeos_space_invaders_web",
    ]
    for name in names:
        assert (dist / "wheels" / name).is_file()
    (ours,) = (dist / "wheels" / n for n in names if n.startswith("zeos_space_invaders_web-"))
    archive = zipfile.ZipFile(ours)
    assert "zeos_space_invaders_web/page.py" in archive.namelist()
    assert "zeos_space_invaders_web/boards/default.json" in archive.namelist()

    case = build.CASE
    assert manifest["cases"] == {
        "space-invaders": sorted(
            p.relative_to(case).as_posix() for p in case.rglob("*") if p.is_file()
        )
    }
    for file in manifest["cases"]["space-invaders"]:
        assert (dist / "cases" / "space-invaders" / file).read_bytes() == (case / file).read_bytes()
    for asset in build.DEBUGGER.iterdir():
        assert (dist / "debugger" / asset.name).read_bytes() == asset.read_bytes()
    for name in build.GENERIC:
        assert (dist / name).read_bytes() == (BROWSER / "web" / name).read_bytes(), name
    for name in build.PAGE:
        assert (dist / name).read_bytes() == (WEB / name).read_bytes(), name
    if all(source.is_dir() for source, _ in build.VENDOR.values()):
        # --model: the local export is linked in and loaded by default; the Hub stays offered.
        assert manifest["model"] == "Tiny-ZEOS-OPT"
        assert manifest["model_source"] == "local"
        assert manifest["model_sources"]["local"] == {
            "name": "Tiny-ZEOS-OPT",
            "path": "models/Tiny-ZEOS-OPT/",
        }
        assert set(manifest["model_sources"]) == {"huggingface", "local"}
        assert (dist / "models" / "Tiny-ZEOS-OPT").is_symlink()
        assert manifest["ort_url"] == model_source.ort_url(build.NODE_MODULES)
        assert not (dist / "vendor" / "onnxruntime-web").exists()
        assert (dist / "vendor" / "tokenizers" / "tokenizers.min.mjs").is_file()
    else:
        assert manifest["model"] is None
    assert manifest["stub"] == build.STUB_WORKER.is_file()
    assert (dist / "stub" / "pilot_stub_worker.js").is_file() == manifest["stub"]
    assert json.loads((dist / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_everything_the_page_references_was_built(built: tuple[Path, dict[str, Any]]) -> None:
    dist, _ = built
    shell = (dist / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'src="([^"]+)"', shell) + re.findall(r'href="([^":]+)"', shell)
    assert "app.js" in refs and "style.css" in refs
    for ref in refs:
        assert (dist / ref).is_file(), f"index.html refers to {ref}, which was not built"
    page_scripts = ("app.js", "si_worker.js", "board.js", "stub_thread.js")
    # zeos-browser's modules app.js imports, whose own imports must be built too.
    model_scripts = ("model_host.js", "model_thread.js", "model_cache.js")
    for script in page_scripts + model_scripts:
        text = (dist / script).read_text(encoding="utf-8")
        for local in re.findall(r'from "\./([^"]+)"', text) + re.findall(
            r'import\("\./([^"]+)"\)', text
        ):
            assert (dist / local).is_file(), f"{script} imports {local}"
        if script in page_scripts:
            for fetched in re.findall(r'"(\w+\.(?:js|json))"', text):
                assert (dist / fetched).is_file(), f"{script} fetches {fetched}"
    for element in re.findall(r'\$\("([\w-]+)"\)', (dist / "app.js").read_text(encoding="utf-8")):
        assert f'id="{element}"' in shell, f"app.js looks up #{element}, which index.html lacks"


def _pyodide_dir() -> Path | None:
    for base in (HERE, BROWSER):
        candidate = base / "node_modules" / "pyodide"
        if (candidate / "pyodide.mjs").is_file() and any(candidate.glob("pyyaml-*.whl")):
            return candidate
    return None


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
@pytest.mark.skipif(_pyodide_dir() is None, reason="needs npm install in packages/zeos-browser")
def test_a_fake_run_plays_under_pyodide(built: tuple[Path, dict[str, Any]], tmp_path: Path) -> None:
    """The four wheels install and ``open_run`` plays under Pyodide, as in si_worker.js."""
    dist, manifest = built
    script = tmp_path / "run.py"
    script.write_text(
        "import json\n"
        "from zeos_space_invaders_web import page\n"
        "class Clock:\n"
        "    t = 0.0\n"
        "    def now(self): return self.t\n"
        "    def sleep(self, s): self.t += s\n"
        "frames = []\n"
        "run = page.open_run('zeos', 'ablation', 7, None, stub=True,\n"
        "                    on_frame=page.json_sink(frames.append), clock=Clock())\n"
        "run.warm()\n"
        "result = run.run()\n"
        "finished = json.loads(page.finished_json(result, '/cases/space-invaders'))\n"
        "print(json.dumps({'frames': len(frames), 'ticks': result.ticks,\n"
        "                  'verdicts': len(finished['verdicts'])}))\n",
        encoding="utf-8",
    )
    command = ["node", str(HERE / "tests" / "pyodide_run.mjs")]
    for name in manifest["wheels"]:
        command += ["--wheel", str(dist / "wheels" / name)]
    command += ["--copy", f"{dist / 'cases' / 'space-invaders'}:/cases/space-invaders", str(script)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["frames"] == out["ticks"] + 1 and out["verdicts"] == 4
