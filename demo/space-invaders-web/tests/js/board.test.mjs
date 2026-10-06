// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// web/board.js without a DOM: the layout arithmetic, and the drawing against a context
// that records its calls.
//
//   node --test demo/space-invaders-web/tests/js/*.test.mjs

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  CELL_UNITS,
  SPRITES,
  SPRITE_SIZE,
  THEME,
  boardSize,
  cellsOf,
  drawBoard,
  drawSprite,
  fitBoard,
  lagStats,
  markerFor,
} from "../../web/board.js";

/** `Game.render` of a 4x3 board: four characters a cell, one line a row. */
function frame(overrides = {}) {
  const text = ["   .  m1   .   .", "   .   .   ^   .", "   .   p   .   ."].join("\n");
  return {
    arm: "zeos",
    board: "ablation",
    tick: 3,
    text,
    lives: 3,
    kills: 0,
    score: 0,
    player: 1,
    monsters: [[0, 1]],
    missile: [1, 2],
    dangers: [[1, 0]],
    can_shoot: false,
    over: false,
    won: false,
    decisions: [],
    catchup: 0,
    preemptions: 0,
    cancellations: 0,
    reflexes: 0,
    ...overrides,
  };
}

function recorder() {
  const calls = [];
  const ctx = {};
  for (const name of ["fillRect", "strokeRect", "setTransform"]) {
    ctx[name] = (...args) => calls.push({ name, args, fillStyle: ctx.fillStyle, strokeStyle: ctx.strokeStyle });
  }
  return { ctx, calls };
}

test("the board's size comes off the frame's text", () => {
  assert.deepEqual(boardSize(frame()), { w: 4, h: 3 });
  const wide = { text: Array(16).fill(" ".repeat(48)).join("\n") };
  assert.deepEqual(boardSize(wide), { w: 12, h: 16 });
});

test("the cell is whole device pixels, a multiple of the sprite grid, and fits", () => {
  for (const [cssW, cssH, w, h, dpr] of [
    [400, 600, 12, 16, 1],
    [400, 600, 12, 16, 2],
    [375, 500, 9, 8, 3],
    [1280, 900, 12, 16, 1.25],
  ]) {
    const layout = fitBoard(cssW, cssH, w, h, dpr);
    assert.ok(Number.isInteger(layout.cell), `cell ${layout.cell}`);
    assert.equal(layout.cell % CELL_UNITS, 0);
    assert.ok(layout.width <= cssW * dpr && layout.height <= cssH * dpr);
    assert.equal(layout.width, layout.cell * w);
    assert.ok(Math.abs(layout.cssWidth - layout.width / dpr) < 1e-9);
    // Fills the box to within one step of the grid.
    assert.ok(
      (layout.cell + CELL_UNITS) * w > cssW * dpr || (layout.cell + CELL_UNITS) * h > cssH * dpr,
      "a bigger cell would also have fitted",
    );
  }
  assert.equal(fitBoard(30, 30, 12, 16, 1).cell, 1, "a box too small still draws");
});

test("cells are drawn in Game.render's order, the ship on the last row", () => {
  assert.deepEqual(cellsOf(frame()), [
    { kind: "bomb", row: 1, col: 0 },
    { kind: "missile", row: 1, col: 2 },
    { kind: "monster", row: 0, col: 1 },
    { kind: "ship", row: 2, col: 1 },
  ]);
  assert.equal(cellsOf(frame({ missile: null, dangers: [] })).length, 2);
});

test("a sprite is drawn run by run inside its margin", () => {
  const { ctx, calls } = recorder();
  const cell = CELL_UNITS * 3;
  drawSprite(ctx, SPRITES.missile, 100, 200, cell);
  // The missile is a 2-wide bar five rows tall: five runs.
  assert.equal(calls.length, 5);
  assert.deepEqual(calls[0].args, [100 + 3 + 3 * 3, 200 + 3, 6, 3]);
  for (const bitmap of Object.values(SPRITES)) {
    assert.equal(bitmap.length, SPRITE_SIZE);
    for (const line of bitmap) assert.equal(line.length, SPRITE_SIZE);
  }
});

test("drawBoard paints the background, then every sprite in its colour", () => {
  const { ctx, calls } = recorder();
  const layout = fitBoard(400, 300, 4, 3, 1);
  drawBoard(ctx, frame(), layout);
  assert.deepEqual(calls[0].args, [0, 0, 4 * layout.cell, 3 * layout.cell]);
  assert.equal(calls[0].fillStyle, THEME.background);
  const colours = new Set(calls.map((c) => c.fillStyle));
  for (const kind of ["monster", "ship", "bomb", "missile"]) assert.ok(colours.has(THEME[kind]), kind);
  // Every rect stays on the board.
  for (const { args } of calls) {
    const [x, y, w, h] = args;
    assert.ok(x >= 0 && y >= 0 && x + w <= 4 * layout.cell && y + h <= 3 * layout.cell, JSON.stringify(args));
  }
  const ship = calls.filter((c) => c.fillStyle === THEME.ship);
  const unit = layout.cell / CELL_UNITS;
  assert.ok(ship.every(({ args }) => args[0] >= layout.cell && args[1] >= 2 * layout.cell + unit));
});

test("a ghost column is outlined only where the ship is not", () => {
  const layout = fitBoard(400, 300, 4, 3, 1);
  const a = recorder();
  drawBoard(a.ctx, frame(), layout, { ghost: 3 });
  assert.equal(a.calls.filter((c) => c.name === "strokeRect").length, 1);
  const b = recorder();
  drawBoard(b.ctx, frame(), layout, { ghost: 1 });
  assert.equal(b.calls.filter((c) => c.name === "strokeRect").length, 0);
});

test("move markers name the author and say when a move was late or preempted", () => {
  const move = { by: "pilot", action: "left", tick: 4, tick_applied: 6, lag_ticks: 2, latency: 1.1, preempted: false, applied: true };
  assert.deepEqual(markerFor(move), { text: "pilot left, 2 ticks late", tone: "pilot" });
  assert.equal(markerFor({ ...move, lag_ticks: 1 }).text, "pilot left, 1 tick late");
  assert.equal(markerFor({ ...move, lag_ticks: 0, applied: false }).text, "pilot left (not applied)");
  assert.deepEqual(markerFor({ ...move, by: "evade", action: "right", preempted: true }), {
    text: "PREEMPTED — evade right",
    tone: "preempted",
  });
  assert.deepEqual(markerFor({ ...move, by: "evade", preempted: false }), { text: "evade left", tone: "evade" });
  assert.equal(markerFor({ ...move, by: "prompt" }).tone, "prompt");
});

test("lag statistics match page.lag_stats: mean and nearest-rank p95", () => {
  assert.deepEqual(lagStats([]), { mean: 0, p95: 0, count: 0 });
  assert.deepEqual(lagStats([3, 1, 2]), { mean: 2, p95: 3, count: 3 });
  const twenty = Array.from({ length: 20 }, (_, i) => i + 1);
  assert.equal(lagStats(twenty).p95, 19);
});
