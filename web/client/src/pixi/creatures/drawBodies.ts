/**
 * 9 body-plan draw functions + shared painterly helpers.
 * Each function draws a single creature at the given size, direction, state, and animation frame.
 * Canvas 2D — called once per (plan, state, dir, frame) combination during init.
 */
import { Direction, VisualState, ANIM_FRAMES } from "./types";
import type { DrawParams, DrawnDirection, BodyPlan } from "./types";
import { PALETTES, STATE_MODIFIERS } from "./palette";

// ─── Shared helpers ────────────────────────────────────────────────────

/** Radial gradient fill for soft organic shapes. */
function softBody(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  rx: number, ry: number,
  colorCenter: string, colorEdge: string,
) {
  const r = Math.max(rx, ry);
  const grad = ctx.createRadialGradient(cx - rx * 0.15, cy - ry * 0.15, r * 0.05, cx, cy, r);
  grad.addColorStop(0, colorCenter);
  grad.addColorStop(1, colorEdge);
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
}

/** Bioluminescent dot with glow halo. */
function bioSpot(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  dotR: number, glowR: number,
  color: string, glowColor: string,
) {
  // Outer glow
  const grad = ctx.createRadialGradient(x, y, 0, x, y, glowR);
  grad.addColorStop(0, glowColor);
  grad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, glowR, 0, Math.PI * 2);
  ctx.fill();
  // Core dot
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, dotR, 0, Math.PI * 2);
  ctx.fill();
}

/** Soft transparent stroke around a path (call after defining path). */
function glowEdge(ctx: CanvasRenderingContext2D, color: string, width: number) {
  ctx.save();
  ctx.globalAlpha = 0.35;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();
  ctx.restore();
}

/** Tapered bezier curve limb with gradient. */
function organicLimb(
  ctx: CanvasRenderingContext2D,
  x1: number, y1: number,
  cpx: number, cpy: number,
  x2: number, y2: number,
  widthStart: number, widthEnd: number,
  color: string,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineCap = "round";

  // Draw tapered by rendering multiple thin strokes
  const steps = 8;
  for (let i = 0; i < steps; i++) {
    const t0 = i / steps;
    const t1 = (i + 1) / steps;
    const w = widthStart + (widthEnd - widthStart) * ((t0 + t1) / 2);
    ctx.lineWidth = w;
    ctx.beginPath();
    const p0 = bezierPoint(x1, y1, cpx, cpy, x2, y2, t0);
    const p1 = bezierPoint(x1, y1, cpx, cpy, x2, y2, t1);
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.stroke();
  }
  ctx.restore();
}

function bezierPoint(
  x1: number, y1: number,
  cpx: number, cpy: number,
  x2: number, y2: number,
  t: number,
) {
  const mt = 1 - t;
  return {
    x: mt * mt * x1 + 2 * mt * t * cpx + t * t * x2,
    y: mt * mt * y1 + 2 * mt * t * cpy + t * t * y2,
  };
}

/** Direction-aware eye. Iris shifts to indicate facing. */
function drawEye(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  r: number,
  direction: DrawnDirection,
  irisColor: string,
  state: VisualState,
) {
  // Sclera
  ctx.fillStyle = "#fff";
  const scleraRx = r;
  const scleraRy = state === VisualState.FLEE ? r * 0.65 : r * 0.85;
  ctx.beginPath();
  ctx.ellipse(cx, cy, scleraRx, scleraRy, 0, 0, Math.PI * 2);
  ctx.fill();

  // Iris offset based on direction
  let irisOx = 0;
  let irisOy = 0;
  const shift = r * 0.2;
  if (direction === Direction.RIGHT) irisOx = shift;
  else if (direction === Direction.UP) irisOy = -shift;
  else irisOy = shift * 0.5; // DOWN — slightly below center

  const irisR = state === VisualState.HUNT ? r * 0.55 : r * 0.45;
  ctx.fillStyle = irisColor;
  ctx.beginPath();
  ctx.arc(cx + irisOx, cy + irisOy, irisR, 0, Math.PI * 2);
  ctx.fill();

  // Highlight
  ctx.fillStyle = "rgba(255,255,255,0.8)";
  ctx.beginPath();
  ctx.arc(cx + irisOx - r * 0.15, cy + irisOy - r * 0.15, r * 0.2, 0, Math.PI * 2);
  ctx.fill();
}

// ─── Animation helpers ────────────────────────────────────────────────

function animPhase(frame: number): number {
  return Math.sin((frame / ANIM_FRAMES) * Math.PI * 2);
}

function breathe(frame: number, amplitude: number = 0.04): number {
  return 1 + animPhase(frame) * amplitude;
}

function bob(frame: number, size: number, amplitude: number = 0.02): number {
  return animPhase(frame) * size * amplitude;
}

// ─── State pose helpers ───────────────────────────────────────────────

function stateScaleOffset(state: VisualState): number {
  return STATE_MODIFIERS[state].scale;
}

/** Saturation/brightness is applied as ImageData post-process in spriteSheet.ts. */

// ─── Body plan draw functions ─────────────────────────────────────────

function drawPredator(p: DrawParams) {
  const { ctx, size: s, frame, direction: dir, state } = p;
  const pal = PALETTES.predator;
  const cx = s / 2, cy = s / 2, r = s * 0.38;
  const bv = bob(frame, s);
  const sc = stateScaleOffset(state);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(sc, sc);
  ctx.translate(-cx, -cy);

  // Body shape varies by direction
  if (dir === Direction.UP) {
    // Back view — wider, no face
    softBody(ctx, cx, cy + bv, r * 0.85, r * 0.9, pal.primary, pal.secondary);
    // Dorsal spines
    for (let i = -2; i <= 2; i++) {
      const sx = cx + i * s * 0.08;
      ctx.fillStyle = pal.accent;
      ctx.beginPath();
      ctx.moveTo(sx - s * 0.02, cy - r * 0.3 + bv);
      ctx.lineTo(sx, cy - r * 0.7 + bv);
      ctx.lineTo(sx + s * 0.02, cy - r * 0.3 + bv);
      ctx.closePath();
      ctx.fill();
    }
  } else if (dir === Direction.RIGHT) {
    // Side view — streamlined
    softBody(ctx, cx, cy + bv, r * 0.95, r * 0.6, pal.primary, pal.secondary);
    // Jaw
    ctx.fillStyle = pal.secondary;
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.7, cy + bv);
    ctx.lineTo(cx + r * 1.1, cy + r * 0.15 + bv);
    ctx.lineTo(cx + r * 0.7, cy + r * 0.3 + bv);
    ctx.closePath();
    ctx.fill();
    // Dorsal spine
    ctx.fillStyle = pal.accent;
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.1, cy - r * 0.5 + bv);
    ctx.lineTo(cx + r * 0.1, cy - r * 0.6 + bv);
    ctx.lineTo(cx + r * 0.3, cy - r * 0.5 + bv);
    ctx.closePath();
    ctx.fill();
    // Claw
    organicLimb(ctx, cx - r * 0.3, cy + r * 0.5 + bv, cx - r * 0.5, cy + r * 0.8 + bv, cx - r * 0.6, cy + r * 0.7 + bv, s * 0.04, s * 0.015, pal.accent);
    // Eye
    drawEye(ctx, cx + r * 0.4, cy - r * 0.15 + bv, s * 0.06, dir, pal.eye, state);
  } else {
    // DOWN — front view (original wedge-like design, enhanced)
    // Body
    ctx.fillStyle = pal.primary;
    ctx.beginPath();
    ctx.moveTo(cx, cy - r + bv);
    ctx.quadraticCurveTo(cx + r * 0.95, cy - r * 0.2 + bv, cx + r * 0.85, cy + r * 0.6 + bv);
    ctx.lineTo(cx + r * 0.3, cy + r * 0.5 + bv);
    ctx.lineTo(cx - r * 0.3, cy + r * 0.5 + bv);
    ctx.lineTo(cx - r * 0.85, cy + r * 0.6 + bv);
    ctx.quadraticCurveTo(cx - r * 0.95, cy - r * 0.2 + bv, cx, cy - r + bv);
    ctx.closePath();
    ctx.fill();
    glowEdge(ctx, pal.accent, s * 0.02);

    // Belly
    softBody(ctx, cx, cy + r * 0.15 + bv, r * 0.35, r * 0.3, pal.belly, pal.secondary);

    // Claws
    for (const side of [-1, 1]) {
      organicLimb(ctx, cx + side * r * 0.6, cy + r * 0.55 + bv, cx + side * r * 0.7, cy + r * 0.75 + bv, cx + side * r * 0.75, cy + r * 0.85 + bv, s * 0.04, s * 0.015, pal.accent);
    }

    // Eyes
    for (const dx of [-0.22, 0.22]) {
      drawEye(ctx, cx + r * dx, cy - r * 0.2 + bv, s * 0.055, dir, pal.eye, state);
    }
  }

  ctx.restore();
}

function drawGrazer(p: DrawParams) {
  const { ctx, size: s, frame, direction: dir, state } = p;
  const pal = PALETTES.grazer;
  const cx = s / 2, cy = s / 2, r = s * 0.36;
  const pulse = breathe(frame);
  const sc = stateScaleOffset(state);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(sc, sc);
  ctx.translate(-cx, -cy);

  if (dir === Direction.UP) {
    softBody(ctx, cx, cy, r * pulse * 0.9, r * pulse, pal.primary, pal.secondary);
    // Leaf-ear appendages from behind
    for (const side of [-1, 1]) {
      organicLimb(ctx, cx + side * r * 0.3, cy - r * 0.4, cx + side * r * 0.6, cy - r * 0.8, cx + side * r * 0.5, cy - r * 0.9, s * 0.04, s * 0.02, pal.accent);
    }
    // Vein patterns
    ctx.strokeStyle = pal.accent;
    ctx.lineWidth = s * 0.015;
    for (let i = 0; i < 3; i++) {
      const angle = (i * Math.PI * 2) / 3 + Math.PI / 4;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(angle) * r * 0.2, cy + Math.sin(angle) * r * 0.2);
      ctx.lineTo(cx + Math.cos(angle) * r * 0.55 * pulse, cy + Math.sin(angle) * r * 0.55 * pulse);
      ctx.stroke();
    }
    // Stubby legs from behind
    for (const side of [-1, 1]) {
      organicLimb(ctx, cx + side * r * 0.35, cy + r * 0.7, cx + side * r * 0.4, cy + r * 0.9, cx + side * r * 0.35, cy + r * 1.0, s * 0.05, s * 0.035, pal.secondary);
    }
  } else if (dir === Direction.RIGHT) {
    // Side view — dome shape
    softBody(ctx, cx, cy, r * 0.85 * pulse, r * pulse * 0.9, pal.primary, pal.secondary);
    // Leaf ear
    organicLimb(ctx, cx + r * 0.1, cy - r * 0.6, cx + r * 0.3, cy - r * 1.0, cx + r * 0.15, cy - r * 1.05, s * 0.04, s * 0.02, pal.accent);
    // Legs (front and back)
    organicLimb(ctx, cx - r * 0.35, cy + r * 0.6, cx - r * 0.4, cy + r * 0.85, cx - r * 0.35, cy + r * 0.95, s * 0.05, s * 0.035, pal.secondary);
    organicLimb(ctx, cx + r * 0.3, cy + r * 0.6, cx + r * 0.35, cy + r * 0.85, cx + r * 0.3, cy + r * 0.95, s * 0.05, s * 0.035, pal.secondary);
    // Eye
    drawEye(ctx, cx + r * 0.35, cy - r * 0.2, s * 0.085, dir, pal.eye, state);
  } else {
    // DOWN — front view
    softBody(ctx, cx, cy, r * pulse * 0.9, r * pulse, pal.primary, pal.secondary);

    // Leaf-ear appendages
    for (const side of [-1, 1]) {
      organicLimb(ctx, cx + side * r * 0.35, cy - r * 0.5, cx + side * r * 0.65, cy - r * 0.85, cx + side * r * 0.55, cy - r * 0.95, s * 0.04, s * 0.018, pal.accent);
    }

    // Vein patterns
    ctx.strokeStyle = pal.accent;
    ctx.lineWidth = s * 0.015;
    for (let i = 0; i < 3; i++) {
      const angle = (i * Math.PI * 2) / 3 + Math.PI / 6;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(angle) * r * 0.3, cy + Math.sin(angle) * r * 0.3);
      ctx.lineTo(cx + Math.cos(angle) * r * 0.65 * pulse, cy + Math.sin(angle) * r * 0.65 * pulse);
      ctx.stroke();
    }

    // Stubby legs
    for (const side of [-1, 1]) {
      organicLimb(ctx, cx + side * r * 0.4, cy + r * 0.65, cx + side * r * 0.45, cy + r * 0.85, cx + side * r * 0.4, cy + r * 0.95, s * 0.05, s * 0.035, pal.secondary);
    }

    // Big central eye
    drawEye(ctx, cx, cy - r * 0.15, s * 0.09, dir, pal.eye, state);
  }

  ctx.restore();
}

function drawOmnivore(p: DrawParams) {
  const { ctx, size: s, frame, direction: dir, state } = p;
  const pal = PALETTES.omnivore;
  const cx = s / 2, cy = s / 2, r = s * 0.36;
  const wobble = animPhase(frame) * 0.04 * r;
  const sc = stateScaleOffset(state);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(sc, sc);
  ctx.translate(-cx, -cy);

  if (dir === Direction.UP) {
    // Hexagonal back
    ctx.fillStyle = pal.primary;
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (i * Math.PI * 2) / 6 - Math.PI / 6;
      const px = cx + Math.cos(a) * r * 0.85 + wobble * 0.3;
      const py = cy + Math.sin(a) * r * 0.85;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
    glowEdge(ctx, pal.accent, s * 0.015);

    // Four legs from behind
    for (const [ox, oy] of [[-0.4, 0.6], [0.4, 0.6], [-0.3, 0.7], [0.3, 0.7]]) {
      organicLimb(ctx, cx + r * ox, cy + r * oy, cx + r * ox * 1.1, cy + r * 0.9, cx + r * ox * 1.0, cy + r * 1.0, s * 0.04, s * 0.025, pal.secondary);
    }
  } else if (dir === Direction.RIGHT) {
    // Side view — stocky oval
    softBody(ctx, cx + wobble, cy, r * 0.9, r * 0.7, pal.primary, pal.secondary);
    // Belly patch
    softBody(ctx, cx + wobble, cy + r * 0.1, r * 0.5, r * 0.35, pal.belly, pal.secondary);
    // Legs
    organicLimb(ctx, cx - r * 0.35, cy + r * 0.5, cx - r * 0.4, cy + r * 0.8, cx - r * 0.35, cy + r * 0.9, s * 0.045, s * 0.03, pal.secondary);
    organicLimb(ctx, cx + r * 0.35, cy + r * 0.5, cx + r * 0.4, cy + r * 0.8, cx + r * 0.35, cy + r * 0.9, s * 0.045, s * 0.03, pal.secondary);
    // Eye
    drawEye(ctx, cx + r * 0.45 + wobble, cy - r * 0.2, s * 0.04, dir, pal.eye, state);
  } else {
    // DOWN — front hexagonal
    ctx.fillStyle = pal.primary;
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (i * Math.PI * 2) / 6 - Math.PI / 6;
      const px = cx + Math.cos(a) * r * 0.9 + wobble;
      const py = cy + Math.sin(a) * r * 0.85;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
    glowEdge(ctx, pal.accent, s * 0.015);

    // Belly
    softBody(ctx, cx + wobble, cy + r * 0.1, r * 0.45, r * 0.35, pal.belly, pal.secondary);

    // Four appendages
    for (const [ox, oy] of [[-0.55, -0.35], [0.55, -0.35], [-0.5, 0.55], [0.5, 0.55]]) {
      organicLimb(ctx, cx + r * ox + wobble, cy + r * oy, cx + r * ox * 1.3 + wobble, cy + r * oy * 1.2, cx + r * ox * 1.2 + wobble, cy + r * oy * 1.4, s * 0.04, s * 0.02, pal.secondary);
    }

    // Eyes
    for (const dx of [-0.25, 0.25]) {
      drawEye(ctx, cx + r * dx + wobble, cy - r * 0.3, s * 0.04, dir, pal.eye, state);
    }
  }

  ctx.restore();
}

function drawScavenger(p: DrawParams) {
  const { ctx, size: s, frame, direction: dir, state } = p;
  const pal = PALETTES.scavenger;
  const cx = s / 2, cy = s / 2, r = s * 0.36;
  const bv = bob(frame, s, 0.015);
  const wingBob = animPhase(frame) * s * 0.03;
  const sc = stateScaleOffset(state);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(sc, sc);
  ctx.translate(-cx, -cy);

  if (dir === Direction.UP) {
    // Back — elongated body, wings spread
    softBody(ctx, cx, cy + bv, r * 0.45, r * 0.8, pal.primary, pal.secondary);
    // Wings
    for (const side of [-1, 1]) {
      ctx.fillStyle = pal.belly;
      ctx.beginPath();
      ctx.moveTo(cx + side * r * 0.3, cy - r * 0.2 + bv);
      ctx.quadraticCurveTo(cx + side * r * 1.0, cy - r * 0.6 + wingBob, cx + side * r * 0.9, cy + r * 0.1 + bv);
      ctx.closePath();
      ctx.fill();
    }
    // Orange stripe
    ctx.strokeStyle = pal.accent;
    ctx.lineWidth = s * 0.025;
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.2, cy + r * 0.4 + bv);
    ctx.lineTo(cx + r * 0.2, cy + r * 0.4 + bv);
    ctx.stroke();
  } else if (dir === Direction.RIGHT) {
    // Side view — hunched profile
    softBody(ctx, cx, cy + bv, r * 0.8, r * 0.55, pal.primary, pal.secondary);
    // Beak
    ctx.fillStyle = pal.accent;
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.6, cy - r * 0.1 + bv);
    ctx.lineTo(cx + r * 1.0, cy + r * 0.05 + bv);
    ctx.lineTo(cx + r * 0.6, cy + r * 0.15 + bv);
    ctx.closePath();
    ctx.fill();
    // Wing (one side)
    ctx.fillStyle = pal.belly;
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.1, cy - r * 0.3 + bv);
    ctx.quadraticCurveTo(cx - r * 0.6, cy - r * 0.7 + wingBob, cx - r * 0.7, cy + bv);
    ctx.lineTo(cx - r * 0.2, cy + r * 0.2 + bv);
    ctx.closePath();
    ctx.fill();
    // Eye
    drawEye(ctx, cx + r * 0.35, cy - r * 0.2 + bv, s * 0.035, dir, pal.eye, state);
  } else {
    // DOWN — front view
    softBody(ctx, cx, cy + bv, r * 0.5, r * 0.85, pal.primary, pal.secondary);

    // Wing protrusions
    for (const side of [-1, 1]) {
      ctx.fillStyle = pal.belly;
      ctx.beginPath();
      ctx.moveTo(cx + side * r * 0.35, cy - r * 0.2 + bv);
      ctx.quadraticCurveTo(cx + side * r * 1.0, cy - r * 0.5 + wingBob, cx + side * r * 0.8, cy + r * 0.1 + bv);
      ctx.closePath();
      ctx.fill();
    }

    // Beak protrusion
    ctx.fillStyle = pal.accent;
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.1, cy - r * 0.5 + bv);
    ctx.lineTo(cx, cy - r * 0.75 + bv);
    ctx.lineTo(cx + r * 0.1, cy - r * 0.5 + bv);
    ctx.closePath();
    ctx.fill();

    // Orange stripe
    ctx.strokeStyle = pal.accent;
    ctx.lineWidth = s * 0.025;
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.3, cy + r * 0.3 + bv);
    ctx.lineTo(cx + r * 0.3, cy + r * 0.3 + bv);
    ctx.stroke();

    // Eyes
    for (const dx of [-0.15, 0.15]) {
      drawEye(ctx, cx + r * dx, cy - r * 0.35 + bv, s * 0.035, dir, pal.eye, state);
    }
  }

  ctx.restore();
}

function drawDecomposer(p: DrawParams) {
  const { ctx, size: s, frame, direction: dir, state } = p;
  const pal = PALETTES.decomposer;
  const cx = s / 2, cy = s / 2, r = s * 0.3;
  const shift = animPhase(frame) * s * 0.015;
  const sc = stateScaleOffset(state);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(sc, sc);
  ctx.translate(-cx, -cy);

  // Direction slightly shifts blob arrangement
  const dOx = dir === Direction.RIGHT ? shift : 0;
  const dOy = dir === Direction.UP ? -Math.abs(shift) : 0;

  // Overlapping translucent blobs
  const blobs = [
    { x: cx - r * 0.25 + shift + dOx, y: cy - r * 0.1 + dOy, rx: r * 0.7, ry: r * 0.6, color: pal.primary },
    { x: cx + r * 0.2 - shift + dOx, y: cy + r * 0.2 + dOy, rx: r * 0.65, ry: r * 0.7, color: pal.secondary },
    { x: cx - r * 0.05 + dOx, y: cy - r * 0.15 + shift + dOy, rx: r * 0.55, ry: r * 0.5, color: pal.belly },
    { x: cx + r * 0.15 + dOx, y: cy + r * 0.3 + dOy, rx: r * 0.4, ry: r * 0.38, color: pal.secondary },
  ];

  for (const b of blobs) {
    ctx.fillStyle = b.color;
    ctx.beginPath();
    ctx.ellipse(b.x, b.y, b.rx, b.ry, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  // Pseudopods
  const pseudoAngles = dir === Direction.RIGHT
    ? [0, 0.6, -0.6]
    : dir === Direction.UP
      ? [-1.2, -1.8, -2.4]
      : [2.0, 2.6, 3.2];
  for (const angle of pseudoAngles) {
    const len = r * 0.7 + shift * 0.5;
    organicLimb(ctx, cx + dOx, cy + dOy, cx + Math.cos(angle) * len * 0.6 + dOx, cy + Math.sin(angle) * len * 0.6 + dOy, cx + Math.cos(angle) * len + dOx, cy + Math.sin(angle) * len + dOy, s * 0.04, s * 0.01, pal.accent);
  }

  // Speckles
  ctx.fillStyle = pal.accent;
  for (let i = 0; i < 8; i++) {
    const angle = (i / 8) * Math.PI * 2 + frame * 0.3;
    const d = r * 0.4;
    ctx.beginPath();
    ctx.arc(cx + Math.cos(angle) * d + dOx, cy + Math.sin(angle) * d + dOy, s * 0.018, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function drawSessileAutotroph(p: DrawParams) {
  const { ctx, size: s, frame, direction: dir, state } = p;
  const pal = PALETTES.sessile_autotroph;
  const cx = s / 2, cy = s / 2, r = s * 0.38;
  const glow = 0.5 + animPhase(frame) * 0.3;
  const sc = stateScaleOffset(state);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(sc, sc);
  ctx.translate(-cx, -cy);

  // Rooted base
  ctx.fillStyle = pal.belly;
  ctx.beginPath();
  ctx.ellipse(cx, cy + r * 0.65, r * 0.25, r * 0.1, 0, 0, Math.PI * 2);
  ctx.fill();

  // Trunk
  organicLimb(ctx, cx, cy + r * 0.65, cx + (dir === Direction.RIGHT ? s * 0.03 : dir === Direction.UP ? -s * 0.01 : 0), cy + r * 0.2, cx, cy - r * 0.1, s * 0.08, s * 0.04, pal.primary);

  // Branches — spread depends on direction (slight variation for visual interest)
  const branchSets: Array<{ angle: number; len: number }> = [];
  const baseAngle = dir === Direction.RIGHT ? -0.3 : dir === Direction.UP ? 0.1 : 0;
  branchSets.push(
    { angle: -Math.PI / 2 + baseAngle, len: r * 0.8 },
    { angle: -Math.PI / 2 - 0.5 + baseAngle, len: r * 0.6 },
    { angle: -Math.PI / 2 + 0.5 + baseAngle, len: r * 0.6 },
    { angle: -Math.PI / 2 - 0.9 + baseAngle, len: r * 0.4 },
    { angle: -Math.PI / 2 + 0.9 + baseAngle, len: r * 0.4 },
  );

  ctx.strokeStyle = pal.secondary;
  ctx.lineWidth = s * 0.025;
  ctx.lineCap = "round";
  for (const b of branchSets) {
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.1);
    ctx.lineTo(cx + Math.cos(b.angle) * b.len, cy + Math.sin(b.angle) * b.len);
    ctx.stroke();
  }

  // Luminescent tips
  for (const b of branchSets) {
    const tx = cx + Math.cos(b.angle) * b.len;
    const ty = cy + Math.sin(b.angle) * b.len;
    bioSpot(ctx, tx, ty, s * 0.03, s * 0.06, pal.accent, `rgba(80, 255, 220, ${glow * 0.6})`);
  }

  ctx.restore();
}

function drawMobileAutotroph(p: DrawParams) {
  const { ctx, size: s, frame, direction: dir, state } = p;
  const pal = PALETTES.mobile_autotroph;
  const cx = s / 2, cy = s / 2, r = s * 0.34;
  const sway = animPhase(frame) * s * 0.02;
  const sc = stateScaleOffset(state);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(sc, sc);
  ctx.translate(-cx, -cy);

  // Green glow halo
  const haloGrad = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r);
  haloGrad.addColorStop(0, pal.glow);
  haloGrad.addColorStop(1, "rgba(80, 255, 120, 0)");
  ctx.fillStyle = haloGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();

  // Translucent dome body
  ctx.fillStyle = pal.primary;
  ctx.beginPath();
  if (dir === Direction.RIGHT) {
    ctx.ellipse(cx, cy, r * 0.75, r * 0.65, 0, 0, Math.PI * 2);
  } else if (dir === Direction.UP) {
    ctx.arc(cx, cy - r * 0.05, r * 0.7, 0, Math.PI * 2);
  } else {
    ctx.arc(cx, cy - r * 0.1, r * 0.75, Math.PI, 0);
    ctx.quadraticCurveTo(cx + r * 0.75, cy + r * 0.5, cx, cy + r * 0.4);
    ctx.quadraticCurveTo(cx - r * 0.75, cy + r * 0.5, cx - r * 0.75, cy - r * 0.1);
  }
  ctx.fill();

  // Organelle dots
  ctx.fillStyle = pal.secondary;
  const organelles = [
    { x: -0.15, y: -0.1 }, { x: 0.2, y: 0.0 },
    { x: -0.05, y: 0.15 }, { x: 0.1, y: -0.2 },
    { x: -0.18, y: 0.08 },
  ];
  for (const o of organelles) {
    ctx.beginPath();
    ctx.arc(cx + r * o.x, cy + r * o.y, s * 0.03, 0, Math.PI * 2);
    ctx.fill();
  }

  // Flagella
  const flagellaDir = dir === Direction.UP ? -1 : 1;
  const flagellaBaseY = dir === Direction.UP ? cy - r * 0.6 : cy + r * 0.4;
  ctx.strokeStyle = pal.accent;
  ctx.lineWidth = s * 0.012;
  for (let i = 0; i < 3; i++) {
    const startX = cx - r * 0.2 + i * r * 0.2;
    ctx.beginPath();
    ctx.moveTo(startX, flagellaBaseY);
    ctx.quadraticCurveTo(
      startX + sway, flagellaBaseY + flagellaDir * r * 0.35,
      startX + sway * 2, flagellaBaseY + flagellaDir * r * 0.65,
    );
    ctx.stroke();
  }

  ctx.restore();
}

function drawFilterFeeder(p: DrawParams) {
  const { ctx, size: s, frame, direction: dir, state } = p;
  const pal = PALETTES.filter_feeder;
  const cx = s / 2, cy = s / 2, r = s * 0.38;
  const wave = animPhase(frame) * 0.06;
  const sc = stateScaleOffset(state);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(sc, sc);
  ctx.translate(-cx, -cy);

  // Direction offsets the fan center angle
  let centerAngle = -Math.PI / 2;
  if (dir === Direction.RIGHT) centerAngle = 0;
  else if (dir === Direction.UP) centerAngle = -Math.PI / 2;

  const armCount = 7;
  for (let i = 0; i < armCount; i++) {
    const angle = centerAngle + (i - (armCount - 1) / 2) * 0.32;
    const len = r * (0.7 + Math.sin(i * 1.5 + frame * 0.8) * 0.1);

    // Feathery gradient arm
    const grad = ctx.createLinearGradient(
      cx, cy,
      cx + Math.cos(angle + wave) * len,
      cy + Math.sin(angle + wave) * len,
    );
    grad.addColorStop(0, pal.primary);
    grad.addColorStop(0.6, pal.accent);
    grad.addColorStop(1, pal.glow);

    ctx.strokeStyle = grad;
    ctx.lineWidth = s * 0.035;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle + wave) * len, cy + Math.sin(angle + wave) * len);
    ctx.stroke();

    // Feathery side strands
    ctx.strokeStyle = pal.glow;
    ctx.lineWidth = s * 0.012;
    const midX = cx + Math.cos(angle + wave) * len * 0.6;
    const midY = cy + Math.sin(angle + wave) * len * 0.6;
    const perpAngle = angle + Math.PI / 2;
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(midX, midY);
      ctx.lineTo(midX + Math.cos(perpAngle) * side * s * 0.05, midY + Math.sin(perpAngle) * side * s * 0.05);
      ctx.stroke();
    }
  }

  // Central body
  softBody(ctx, cx, cy + r * 0.08, r * 0.2, r * 0.2, pal.secondary, pal.primary);

  ctx.restore();
}

function drawParasite(p: DrawParams) {
  const { ctx, size: s, frame, direction: dir, state } = p;
  const pal = PALETTES.parasite;
  const cx = s / 2, cy = s / 2, r = s * 0.3;
  const pulse = breathe(frame, 0.06);
  const sc = stateScaleOffset(state);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(sc, sc);
  ctx.translate(-cx, -cy);

  // Direction rotates the star slightly
  let rotOffset = 0;
  if (dir === Direction.RIGHT) rotOffset = 0.3;
  else if (dir === Direction.UP) rotOffset = Math.PI;

  // Spiky star shape
  const points = 5;
  ctx.fillStyle = pal.primary;
  ctx.beginPath();
  for (let i = 0; i < points * 2; i++) {
    const angle = (i * Math.PI) / points - Math.PI / 2 + rotOffset;
    const dist = i % 2 === 0 ? r * pulse : r * 0.45 * pulse;
    const px = cx + Math.cos(angle) * dist;
    const py = cy + Math.sin(angle) * dist;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fill();

  // Dark outline
  ctx.strokeStyle = pal.secondary;
  ctx.lineWidth = s * 0.02;
  ctx.stroke();
  glowEdge(ctx, pal.glow, s * 0.01);

  // Red warning spots
  const spots = [
    { x: 0, y: -0.12 }, { x: 0.14, y: 0.08 },
    { x: -0.14, y: 0.08 }, { x: 0.07, y: -0.04 },
    { x: -0.07, y: -0.04 },
  ];
  for (const sp of spots) {
    bioSpot(ctx, cx + r * sp.x, cy + r * sp.y, s * 0.022 * pulse, s * 0.04 * pulse, pal.accent, `${pal.glow}44`);
  }

  ctx.restore();
}

// ─── Registry ─────────────────────────────────────────────────────────

export const DRAW_FUNCTIONS: Record<BodyPlan, (p: DrawParams) => void> = {
  predator: drawPredator,
  grazer: drawGrazer,
  omnivore: drawOmnivore,
  scavenger: drawScavenger,
  decomposer: drawDecomposer,
  sessile_autotroph: drawSessileAutotroph,
  mobile_autotroph: drawMobileAutotroph,
  filter_feeder: drawFilterFeeder,
  parasite: drawParasite,
};
