// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The board, drawn on a canvas from a `Frame` (contracts.py).
 *
 * Everything but `BoardView` is a pure function of its arguments, so
 * tests/js/board.test.mjs can check the layout and the drawing against a recording
 * context without a DOM. Sizes are whole device pixels: the cell is an integer number of
 * sprite pixels, and the canvas's backing store is its CSS size times
 * devicePixelRatio, so the sprites stay sharp at any zoom.
 */

/** Sprite bitmaps, 8x8, one string per row; `#` is lit. */
export const SPRITES = {
  monster: [
    "..#..#..",
    "...##...",
    "..####..",
    ".##..##.",
    "########",
    "#.####.#",
    "#.#..#.#",
    "...##...",
  ],
  ship: [
    "........",
    "...##...",
    "...##...",
    "..####..",
    ".######.",
    "########",
    "########",
    "........",
  ],
  bomb: [
    "...#....",
    "....#...",
    "...#....",
    "....#...",
    "...#....",
    "..###...",
    "...#....",
    "........",
  ],
  missile: [
    "...##...",
    "...##...",
    "...##...",
    "...##...",
    "...##...",
    "........",
    "........",
    "........",
  ],
};

export const SPRITE_SIZE = 8;
/** Sprite pixels per cell side: the sprite plus a one-pixel margin all round, so two
 * neighbouring monsters do not merge. */
export const CELL_UNITS = SPRITE_SIZE + 2;

/** The palette `drawBoard` uses unless given another. */
export const THEME = {
  background: "#0d1117",
  grid: "#1c2330",
  monster: "#7ee787",
  ship: "#79c0ff",
  bomb: "#ff7b72",
  missile: "#f2cc60",
  ghost: "rgba(121, 192, 255, 0.35)",
  text: "#e6edf3",
};

/** The board's columns and rows, read off the frame's text (`Game.render`: one row per
 * line, four characters per cell), which is the one place a frame states them. */
export function boardSize(frame) {
  const rows = frame.text.split("\n");
  return { w: Math.round(rows[0].length / 4), h: rows.length };
}

/**
 * The largest whole-pixel cell that fits `w` x `h` cells into `cssWidth` x `cssHeight`
 * CSS pixels at `dpr`. The cell is a multiple of `CELL_UNITS` once there is room for
 * one, so every sprite pixel is the same whole number of device pixels.
 */
export function fitBoard(cssWidth, cssHeight, w, h, dpr = 1) {
  const fit = Math.floor(Math.min((cssWidth * dpr) / w, (cssHeight * dpr) / h));
  const cell = fit >= CELL_UNITS ? fit - (fit % CELL_UNITS) : Math.max(1, fit);
  const width = cell * w;
  const height = cell * h;
  return { cell, width, height, cssWidth: width / dpr, cssHeight: height / dpr, dpr };
}

/** What is on the board, in drawing order (later covers earlier, as `Game.render`). */
export function cellsOf(frame) {
  const { h } = boardSize(frame);
  const cells = [];
  for (const [row, col] of frame.dangers) cells.push({ kind: "bomb", row, col });
  if (frame.missile) cells.push({ kind: "missile", row: frame.missile[0], col: frame.missile[1] });
  for (const [row, col] of frame.monsters) cells.push({ kind: "monster", row, col });
  cells.push({ kind: "ship", row: h - 1, col: frame.player });
  return cells;
}

/** Draw one sprite into the cell at (`x`, `y`) with fillRect calls, one per horizontal
 * run of lit pixels, inside a one-unit margin. */
export function drawSprite(ctx, bitmap, x, y, cell) {
  const unit = cell / CELL_UNITS;
  x += unit;
  y += unit;
  for (let r = 0; r < bitmap.length; r++) {
    const line = bitmap[r];
    let c = 0;
    while (c < line.length) {
      if (line[c] !== "#") {
        c += 1;
        continue;
      }
      let end = c;
      while (end < line.length && line[end] === "#") end += 1;
      ctx.fillRect(x + c * unit, y + r * unit, (end - c) * unit, unit);
      c = end;
    }
  }
}

/**
 * Draw `frame` into `ctx`, whose coordinates are device pixels, at `layout`
 * (`fitBoard`'s answer). `ghost` is a column to outline on the ship's row: where a move
 * in flight was asked from, so a late move can be seen landing.
 */
export function drawBoard(ctx, frame, layout, { theme = THEME, ghost = null } = {}) {
  const { w, h } = boardSize(frame);
  const { cell } = layout;
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, w * cell, h * cell);
  ctx.fillStyle = theme.grid;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const dot = Math.max(1, Math.round(cell / 16));
      ctx.fillRect(c * cell + (cell - dot) / 2, r * cell + (cell - dot) / 2, dot, dot);
    }
  }
  if (ghost !== null && ghost !== frame.player) {
    ctx.strokeStyle = theme.ghost;
    ctx.lineWidth = Math.max(1, Math.round(cell / 16));
    ctx.strokeRect(ghost * cell + 1, (h - 1) * cell + 1, cell - 2, cell - 2);
  }
  for (const { kind, row, col } of cellsOf(frame)) {
    if (row < 0 || row >= h || col < 0 || col >= w) continue;
    ctx.fillStyle = theme[kind];
    drawSprite(ctx, SPRITES[kind], col * cell, row * cell, cell);
  }
}

/** How a move is labelled under the board: who made it, and whether it was late. */
export function markerFor(decision) {
  if (decision.by === "evade") {
    return {
      text: decision.preempted ? `PREEMPTED — evade ${decision.action}` : `evade ${decision.action}`,
      tone: decision.preempted ? "preempted" : "evade",
    };
  }
  const late = decision.lag_ticks > 0 ? `, ${decision.lag_ticks} tick${decision.lag_ticks === 1 ? "" : "s"} late` : "";
  const dropped = decision.applied ? "" : " (not applied)";
  return { text: `${decision.by} ${decision.action}${late}${dropped}`, tone: decision.by };
}

/** Mean and 95th percentile (nearest rank) of a list of lags; zeros for none. */
export function lagStats(lags) {
  if (lags.length === 0) return { mean: 0, p95: 0, count: 0 };
  const sorted = [...lags].sort((a, b) => a - b);
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
  const rank = Math.max(0, Math.ceil(0.95 * sorted.length) - 1);
  return { mean, p95: sorted[rank], count: sorted.length };
}

/** A canvas that redraws the last frame whenever its box or devicePixelRatio changes. */
export class BoardView {
  constructor(canvas, { maxHeight = () => Math.max(320, window.innerHeight * 0.62) } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.frame = null;
    this.ghost = null;
    this.maxHeight = maxHeight;
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas.parentElement);
    const watchDpr = () => {
      matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`).addEventListener(
        "change",
        () => {
          this.draw();
          watchDpr();
        },
        { once: true },
      );
    };
    watchDpr();
  }

  show(frame, ghost = null) {
    this.frame = frame;
    this.ghost = ghost;
    this.draw();
  }

  draw() {
    if (this.frame === null) return;
    const { w, h } = boardSize(this.frame);
    const dpr = window.devicePixelRatio || 1;
    const box = this.canvas.parentElement.clientWidth;
    const layout = fitBoard(box, this.maxHeight(), w, h, dpr);
    if (this.canvas.width !== layout.width || this.canvas.height !== layout.height) {
      this.canvas.width = layout.width;
      this.canvas.height = layout.height;
    }
    this.canvas.style.width = `${layout.cssWidth}px`;
    this.canvas.style.height = `${layout.cssHeight}px`;
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    drawBoard(this.ctx, this.frame, layout, { ghost: this.ghost });
  }
}
