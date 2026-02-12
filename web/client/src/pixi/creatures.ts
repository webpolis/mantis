/**
 * Procedural creature sprite generator for all 9 body plans.
 * Generates 64x64 sprite textures + 32x32 icons via offscreen Canvas 2D → PIXI.Texture.
 * 3 idle animation frames per plan (subtle bob/pulse/wobble).
 * Sprite object pool for rendering agents.
 */
import { Container, Sprite, Texture, Graphics, Text, TextStyle } from "pixi.js";
import type { AgentSnapshot, SpeciesInfo } from "../types/simulation";

const SPRITE_SIZE = 64;
const ICON_SIZE = 32;
const ANIM_FRAMES = 3;
const ANIM_PERIOD = 1000; // ms per cycle

export const BODY_PLAN_COLORS: Record<string, string> = {
  predator: "#cc2233",
  grazer: "#66bb77",
  omnivore: "#9966cc",
  scavenger: "#bb9944",
  decomposer: "#557755",
  sessile_autotroph: "#22ccaa",
  mobile_autotroph: "#44bb66",
  filter_feeder: "#5577cc",
  parasite: "#cc4488",
};

const STATE_TINTS: Record<string, number> = {
  hunt: 0xff3333,
  flee: 0xffaa33,
  mate: 0xff66cc,
  flock: 0x44aaff,
};

type BodyPlan = keyof typeof BODY_PLAN_COLORS;

interface SpriteTextures {
  frames: Texture[];   // 3 animation frames at 64x64
  icon: Texture;       // 32x32 icon
}

interface PooledAgent {
  sprite: Sprite;
  highlight: Graphics;
  energyBar: Graphics;
  stateRing: Graphics;
  countLabel: Text;
  inUse: boolean;
}

// --- Procedural drawing functions ---

function drawPredator(ctx: CanvasRenderingContext2D, s: number, frame: number) {
  const cx = s / 2, cy = s / 2, r = s * 0.38;
  const bob = Math.sin(frame * Math.PI * 2 / ANIM_FRAMES) * s * 0.02;

  // Body - angular wedge shape
  ctx.fillStyle = "#cc2233";
  ctx.beginPath();
  ctx.moveTo(cx, cy - r + bob);
  ctx.lineTo(cx + r * 0.85, cy + r * 0.6 + bob);
  ctx.lineTo(cx + r * 0.3, cy + r * 0.4 + bob);
  ctx.lineTo(cx - r * 0.3, cy + r * 0.4 + bob);
  ctx.lineTo(cx - r * 0.85, cy + r * 0.6 + bob);
  ctx.closePath();
  ctx.fill();

  // Darker underbelly
  ctx.fillStyle = "#991a28";
  ctx.beginPath();
  ctx.moveTo(cx - r * 0.3, cy + r * 0.4 + bob);
  ctx.lineTo(cx + r * 0.3, cy + r * 0.4 + bob);
  ctx.lineTo(cx + r * 0.15, cy + r * 0.7 + bob);
  ctx.lineTo(cx - r * 0.15, cy + r * 0.7 + bob);
  ctx.closePath();
  ctx.fill();

  // Claw/fang marks
  ctx.strokeStyle = "#ff6666";
  ctx.lineWidth = s * 0.03;
  ctx.beginPath();
  ctx.moveTo(cx - r * 0.6, cy + r * 0.55 + bob);
  ctx.lineTo(cx - r * 0.75, cy + r * 0.8 + bob);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx + r * 0.6, cy + r * 0.55 + bob);
  ctx.lineTo(cx + r * 0.75, cy + r * 0.8 + bob);
  ctx.stroke();

  // Glowing eyes
  const eyeR = s * 0.05;
  for (const dx of [-0.2, 0.2]) {
    ctx.fillStyle = "#ff2200";
    ctx.beginPath();
    ctx.arc(cx + r * dx, cy - r * 0.15 + bob, eyeR, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ff8866";
    ctx.beginPath();
    ctx.arc(cx + r * dx, cy - r * 0.15 + bob, eyeR * 0.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawGrazer(ctx: CanvasRenderingContext2D, s: number, frame: number) {
  const cx = s / 2, cy = s / 2, r = s * 0.36;
  const pulse = 1 + Math.sin(frame * Math.PI * 2 / ANIM_FRAMES) * 0.04;

  // Rounded body
  ctx.fillStyle = "#66bb77";
  ctx.beginPath();
  ctx.ellipse(cx, cy, r * pulse * 0.9, r * pulse, 0, 0, Math.PI * 2);
  ctx.fill();

  // Leaf pattern markings
  ctx.strokeStyle = "#88ddaa";
  ctx.lineWidth = s * 0.02;
  for (let i = 0; i < 3; i++) {
    const angle = (i * Math.PI * 2) / 3 + Math.PI / 6;
    const sx = cx + Math.cos(angle) * r * 0.3;
    const sy = cy + Math.sin(angle) * r * 0.3;
    const ex = cx + Math.cos(angle) * r * 0.65 * pulse;
    const ey = cy + Math.sin(angle) * r * 0.65 * pulse;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
  }

  // Big eye
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(cx, cy - r * 0.2, s * 0.09, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#223322";
  ctx.beginPath();
  ctx.arc(cx, cy - r * 0.2, s * 0.05, 0, Math.PI * 2);
  ctx.fill();
}

function drawOmnivore(ctx: CanvasRenderingContext2D, s: number, frame: number) {
  const cx = s / 2, cy = s / 2, r = s * 0.36;
  const wobble = Math.sin(frame * Math.PI * 2 / ANIM_FRAMES) * 0.05 * r;

  // Stocky diamond body
  ctx.fillStyle = "#8855aa";
  ctx.beginPath();
  ctx.moveTo(cx + wobble, cy - r);
  ctx.lineTo(cx + r * 0.9, cy + wobble);
  ctx.lineTo(cx + wobble, cy + r * 0.8);
  ctx.lineTo(cx - r * 0.9, cy + wobble);
  ctx.closePath();
  ctx.fill();

  // Belly patch
  ctx.fillStyle = "#aa77cc";
  ctx.beginPath();
  ctx.ellipse(cx + wobble, cy + r * 0.1, r * 0.4, r * 0.3, 0, 0, Math.PI * 2);
  ctx.fill();

  // Small eyes
  for (const dx of [-0.25, 0.25]) {
    ctx.fillStyle = "#eecc88";
    ctx.beginPath();
    ctx.arc(cx + r * dx + wobble, cy - r * 0.35, s * 0.04, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawScavenger(ctx: CanvasRenderingContext2D, s: number, frame: number) {
  const cx = s / 2, cy = s / 2, r = s * 0.36;
  const bob = Math.sin(frame * Math.PI * 2 / ANIM_FRAMES) * s * 0.015;

  // Elongated body
  ctx.fillStyle = "#887744";
  ctx.beginPath();
  ctx.ellipse(cx, cy + bob, r * 0.5, r * 0.9, 0, 0, Math.PI * 2);
  ctx.fill();

  // Angular wing protrusions
  ctx.fillStyle = "#aa9955";
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(cx + side * r * 0.35, cy - r * 0.2 + bob);
    ctx.lineTo(cx + side * r * 1.0, cy - r * 0.5 + bob);
    ctx.lineTo(cx + side * r * 0.8, cy + r * 0.1 + bob);
    ctx.closePath();
    ctx.fill();
  }

  // Orange accent stripe
  ctx.strokeStyle = "#ee8833";
  ctx.lineWidth = s * 0.03;
  ctx.beginPath();
  ctx.moveTo(cx - r * 0.3, cy + r * 0.3 + bob);
  ctx.lineTo(cx + r * 0.3, cy + r * 0.3 + bob);
  ctx.stroke();

  // Eyes
  ctx.fillStyle = "#ffcc66";
  ctx.beginPath();
  ctx.arc(cx - r * 0.15, cy - r * 0.4 + bob, s * 0.035, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cx + r * 0.15, cy - r * 0.4 + bob, s * 0.035, 0, Math.PI * 2);
  ctx.fill();
}

function drawDecomposer(ctx: CanvasRenderingContext2D, s: number, frame: number) {
  const cx = s / 2, cy = s / 2, r = s * 0.3;
  const shift = frame * 0.03 * s;

  // Amorphous overlapping translucent blobs
  const blobs = [
    { x: cx - r * 0.2 + shift, y: cy - r * 0.1, rx: r * 0.7, ry: r * 0.6, color: "rgba(70, 100, 60, 0.7)" },
    { x: cx + r * 0.25 - shift, y: cy + r * 0.15, rx: r * 0.6, ry: r * 0.7, color: "rgba(60, 90, 50, 0.6)" },
    { x: cx - r * 0.05, y: cy - r * 0.2 + shift, rx: r * 0.55, ry: r * 0.5, color: "rgba(80, 110, 70, 0.5)" },
    { x: cx + r * 0.1, y: cy + r * 0.3, rx: r * 0.4, ry: r * 0.35, color: "rgba(50, 80, 45, 0.65)" },
  ];

  for (const b of blobs) {
    ctx.fillStyle = b.color;
    ctx.beginPath();
    ctx.ellipse(b.x, b.y, b.rx, b.ry, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  // Subtle speckles
  ctx.fillStyle = "rgba(130, 160, 110, 0.4)";
  for (let i = 0; i < 6; i++) {
    const angle = (i / 6) * Math.PI * 2 + frame * 0.3;
    const d = r * 0.4;
    ctx.beginPath();
    ctx.arc(cx + Math.cos(angle) * d, cy + Math.sin(angle) * d, s * 0.02, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawSessileAutotroph(ctx: CanvasRenderingContext2D, s: number, frame: number) {
  const cx = s / 2, cy = s / 2, r = s * 0.38;
  const glow = 0.6 + Math.sin(frame * Math.PI * 2 / ANIM_FRAMES) * 0.2;

  // Trunk/base
  ctx.fillStyle = "#1a8866";
  ctx.beginPath();
  ctx.moveTo(cx - s * 0.04, cy + r * 0.6);
  ctx.lineTo(cx + s * 0.04, cy + r * 0.6);
  ctx.lineTo(cx + s * 0.03, cy - r * 0.1);
  ctx.lineTo(cx - s * 0.03, cy - r * 0.1);
  ctx.closePath();
  ctx.fill();

  // Branching coral/tree structure
  const branches = [
    { angle: -Math.PI / 2, len: r * 0.8 },
    { angle: -Math.PI / 2 - 0.5, len: r * 0.6 },
    { angle: -Math.PI / 2 + 0.5, len: r * 0.6 },
    { angle: -Math.PI / 2 - 0.9, len: r * 0.4 },
    { angle: -Math.PI / 2 + 0.9, len: r * 0.4 },
  ];

  ctx.strokeStyle = "#22aa88";
  ctx.lineWidth = s * 0.03;
  ctx.lineCap = "round";
  for (const b of branches) {
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.1);
    ctx.lineTo(cx + Math.cos(b.angle) * b.len, cy + Math.sin(b.angle) * b.len);
    ctx.stroke();
  }

  // Luminescent cyan tips
  ctx.fillStyle = `rgba(80, 255, 220, ${glow})`;
  for (const b of branches) {
    const tx = cx + Math.cos(b.angle) * b.len;
    const ty = cy + Math.sin(b.angle) * b.len;
    ctx.beginPath();
    ctx.arc(tx, ty, s * 0.04, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawMobileAutotroph(ctx: CanvasRenderingContext2D, s: number, frame: number) {
  const cx = s / 2, cy = s / 2, r = s * 0.34;
  const sway = Math.sin(frame * Math.PI * 2 / ANIM_FRAMES) * s * 0.02;

  // Translucent dome body
  ctx.fillStyle = "rgba(60, 180, 100, 0.6)";
  ctx.beginPath();
  ctx.arc(cx, cy - r * 0.1, r * 0.8, Math.PI, 0);
  ctx.quadraticCurveTo(cx + r * 0.8, cy + r * 0.5, cx, cy + r * 0.4);
  ctx.quadraticCurveTo(cx - r * 0.8, cy + r * 0.5, cx - r * 0.8, cy - r * 0.1);
  ctx.fill();

  // Internal organelle dots
  ctx.fillStyle = "rgba(100, 220, 120, 0.7)";
  const organelles = [
    { x: -0.15, y: -0.1 }, { x: 0.2, y: 0.0 },
    { x: -0.05, y: 0.15 }, { x: 0.1, y: -0.2 },
  ];
  for (const o of organelles) {
    ctx.beginPath();
    ctx.arc(cx + r * o.x, cy + r * o.y, s * 0.035, 0, Math.PI * 2);
    ctx.fill();
  }

  // Trailing flagella
  ctx.strokeStyle = "rgba(100, 200, 130, 0.5)";
  ctx.lineWidth = s * 0.015;
  for (let i = 0; i < 3; i++) {
    const startX = cx - r * 0.2 + i * r * 0.2;
    ctx.beginPath();
    ctx.moveTo(startX, cy + r * 0.4);
    ctx.quadraticCurveTo(
      startX + sway, cy + r * 0.7,
      startX + sway * 2, cy + r * 0.95
    );
    ctx.stroke();
  }

  // Green glow halo
  const grad = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r);
  grad.addColorStop(0, "rgba(80, 255, 120, 0.15)");
  grad.addColorStop(1, "rgba(80, 255, 120, 0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();
}

function drawFilterFeeder(ctx: CanvasRenderingContext2D, s: number, frame: number) {
  const cx = s / 2, cy = s / 2, r = s * 0.38;
  const wave = Math.sin(frame * Math.PI * 2 / ANIM_FRAMES) * 0.08;

  // Fan-shaped radiating appendages
  const armCount = 7;
  for (let i = 0; i < armCount; i++) {
    const angle = -Math.PI / 2 + (i - (armCount - 1) / 2) * 0.35;
    const len = r * (0.7 + Math.sin(i * 1.5 + frame) * 0.1);

    // Feathery edge gradient
    const grad = ctx.createLinearGradient(
      cx, cy,
      cx + Math.cos(angle + wave) * len,
      cy + Math.sin(angle + wave) * len
    );
    grad.addColorStop(0, "#4466aa");
    grad.addColorStop(0.6, "#6688cc");
    grad.addColorStop(1, "rgba(130, 150, 220, 0.3)");

    ctx.strokeStyle = grad;
    ctx.lineWidth = s * 0.04;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle + wave) * len, cy + Math.sin(angle + wave) * len);
    ctx.stroke();

    // Feathery side strands
    ctx.strokeStyle = "rgba(130, 160, 220, 0.3)";
    ctx.lineWidth = s * 0.015;
    const midX = cx + Math.cos(angle + wave) * len * 0.6;
    const midY = cy + Math.sin(angle + wave) * len * 0.6;
    const perpAngle = angle + Math.PI / 2;
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(midX, midY);
      ctx.lineTo(midX + Math.cos(perpAngle) * side * s * 0.06, midY + Math.sin(perpAngle) * side * s * 0.06);
      ctx.stroke();
    }
  }

  // Central body
  ctx.fillStyle = "#3355aa";
  ctx.beginPath();
  ctx.arc(cx, cy + r * 0.1, r * 0.22, 0, Math.PI * 2);
  ctx.fill();
}

function drawParasite(ctx: CanvasRenderingContext2D, s: number, frame: number) {
  const cx = s / 2, cy = s / 2, r = s * 0.3;
  const pulse = 1 + Math.sin(frame * Math.PI * 2 / ANIM_FRAMES) * 0.06;

  // Spiky star shape
  const points = 5;
  ctx.fillStyle = "#442233";
  ctx.beginPath();
  for (let i = 0; i < points * 2; i++) {
    const angle = (i * Math.PI) / points - Math.PI / 2;
    const dist = i % 2 === 0 ? r * pulse : r * 0.45 * pulse;
    const px = cx + Math.cos(angle) * dist;
    const py = cy + Math.sin(angle) * dist;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fill();

  // Darker outline
  ctx.strokeStyle = "#663355";
  ctx.lineWidth = s * 0.02;
  ctx.stroke();

  // Red accent spots
  ctx.fillStyle = "#cc3355";
  const spots = [
    { x: 0, y: -0.15 }, { x: 0.15, y: 0.1 },
    { x: -0.15, y: 0.1 }, { x: 0.08, y: -0.05 },
  ];
  for (const sp of spots) {
    ctx.beginPath();
    ctx.arc(cx + r * sp.x, cy + r * sp.y, s * 0.025 * pulse, 0, Math.PI * 2);
    ctx.fill();
  }
}

const DRAW_FUNCTIONS: Record<string, (ctx: CanvasRenderingContext2D, s: number, frame: number) => void> = {
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

// --- Texture cache ---

const textureCache = new Map<string, SpriteTextures>();

function generateTextures(bodyPlan: string): SpriteTextures {
  const cached = textureCache.get(bodyPlan);
  if (cached) return cached;

  const drawFn = DRAW_FUNCTIONS[bodyPlan] || drawGrazer;
  const frames: Texture[] = [];

  for (let f = 0; f < ANIM_FRAMES; f++) {
    const canvas = document.createElement("canvas");
    canvas.width = SPRITE_SIZE;
    canvas.height = SPRITE_SIZE;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
    drawFn(ctx, SPRITE_SIZE, f);
    frames.push(Texture.from(canvas));
  }

  // Icon variant (32x32)
  const iconCanvas = document.createElement("canvas");
  iconCanvas.width = ICON_SIZE;
  iconCanvas.height = ICON_SIZE;
  const iconCtx = iconCanvas.getContext("2d")!;
  iconCtx.clearRect(0, 0, ICON_SIZE, ICON_SIZE);
  drawFn(iconCtx, ICON_SIZE, 0);
  const icon = Texture.from(iconCanvas);

  const result: SpriteTextures = { frames, icon };
  textureCache.set(bodyPlan, result);
  return result;
}

/** Get a data URL for a creature icon (for use in React components). */
export function getCreatureIconDataURL(bodyPlan: string): string {
  const drawFn = DRAW_FUNCTIONS[bodyPlan] || drawGrazer;
  const canvas = document.createElement("canvas");
  canvas.width = ICON_SIZE;
  canvas.height = ICON_SIZE;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, ICON_SIZE, ICON_SIZE);
  drawFn(ctx, ICON_SIZE, 0);
  return canvas.toDataURL();
}

// --- Creature Renderer (manages sprite pool) ---

const INITIAL_POOL_SIZE = 500;

const COUNT_LABEL_STYLE = new TextStyle({
  fontFamily: "Rajdhani, monospace",
  fontSize: 13,
  fontWeight: "700",
  fill: "#ffffff",
  stroke: { color: "#000000", width: 3 },
});

export class CreatureRenderer {
  private container: Container;
  private pool: PooledAgent[] = [];
  private activeCount = 0;
  private worldSize: number;
  private speciesMap = new Map<number, SpeciesInfo>();

  constructor(container: Container, worldSize: number) {
    this.container = container;
    this.worldSize = worldSize;
    this.growPool(INITIAL_POOL_SIZE);
  }

  private growPool(count: number) {
    for (let i = 0; i < count; i++) {
      const sprite = new Sprite();
      sprite.anchor.set(0.5);
      sprite.visible = false;

      const highlight = new Graphics();
      highlight.visible = false;

      const energyBar = new Graphics();
      energyBar.visible = false;

      const stateRing = new Graphics();
      stateRing.visible = false;

      const countLabel = new Text({ text: "", style: COUNT_LABEL_STYLE });
      countLabel.anchor.set(0.5);
      countLabel.visible = false;

      this.container.addChild(stateRing);
      this.container.addChild(sprite);
      this.container.addChild(highlight);
      this.container.addChild(energyBar);
      this.container.addChild(countLabel);

      this.pool.push({ sprite, highlight, energyBar, stateRing, countLabel, inUse: false });
    }
  }

  private acquire(): PooledAgent {
    for (let i = this.activeCount; i < this.pool.length; i++) {
      if (!this.pool[i].inUse) {
        this.pool[i].inUse = true;
        return this.pool[i];
      }
    }
    // Need to grow
    this.growPool(100);
    const entry = this.pool[this.pool.length - 1];
    entry.inUse = true;
    return entry;
  }

  update(agents: AgentSnapshot[], species: SpeciesInfo[], hoveredUid?: string | null) {
    this.speciesMap = new Map(species.map((s) => [s.sid, s]));

    // Release all
    for (let i = 0; i < this.pool.length; i++) {
      const p = this.pool[i];
      p.inUse = false;
      p.sprite.visible = false;
      p.highlight.visible = false;
      p.energyBar.visible = false;
      p.stateRing.visible = false;
      p.countLabel.visible = false;
    }
    this.activeCount = 0;

    for (const agent of agents) {
      if (agent.dead) continue;

      const entry = this.acquire();
      this.activeCount++;

      const sp = this.speciesMap.get(agent.species_sid);
      const bodyPlan = sp?.plan || "grazer";
      const textures = generateTextures(bodyPlan);

      // Animation frame based on time stored as data
      const frameIndex = (entry.sprite as any).__animFrame || 0;
      entry.sprite.texture = textures.frames[frameIndex % ANIM_FRAMES];
      (entry.sprite as any).__bodyPlan = bodyPlan;
      (entry.sprite as any).__aid = agent.uid;

      // Scale based on count
      const baseScale = (0.4 + Math.sqrt(agent.count || 1) * 0.15);
      entry.sprite.scale.set(baseScale);
      entry.sprite.x = agent.x;
      entry.sprite.y = agent.y;
      entry.sprite.visible = true;

      // State tint
      const stateTint = STATE_TINTS[agent.state];
      if (stateTint) {
        entry.stateRing.visible = true;
        entry.stateRing.clear();
        const ringR = SPRITE_SIZE * baseScale * 0.55;
        entry.stateRing.circle(agent.x, agent.y, ringR);
        entry.stateRing.stroke({ color: stateTint, alpha: 0.6, width: 2 });
      }

      // Hover highlight
      if (agent.uid === hoveredUid) {
        entry.highlight.visible = true;
        entry.highlight.clear();
        const hlR = SPRITE_SIZE * baseScale * 0.6;
        entry.highlight.circle(agent.x, agent.y, hlR);
        entry.highlight.stroke({ color: 0xffffff, alpha: 0.7, width: 2 });
      }

      // Energy bar
      if (agent.energy < 20) {
        entry.energyBar.visible = true;
        entry.energyBar.clear();
        const barW = 10;
        const barH = 2;
        const barX = agent.x - barW / 2;
        const barY = agent.y - SPRITE_SIZE * baseScale * 0.4 - 4;

        entry.energyBar.rect(barX, barY, barW, barH);
        entry.energyBar.fill({ color: 0x333333, alpha: 0.5 });
        entry.energyBar.rect(barX, barY, (agent.energy / 100) * barW, barH);
        entry.energyBar.fill({ color: 0xffff00, alpha: 0.8 });
      }

      // Count badge for clusters
      if (agent.count > 1) {
        entry.countLabel.visible = true;
        entry.countLabel.text = String(agent.count);
        entry.countLabel.x = agent.x;
        entry.countLabel.y = agent.y - SPRITE_SIZE * baseScale * 0.5 - 6;
      }
    }
  }

  updateAnimations(now: number) {
    const phase = (now % ANIM_PERIOD) / ANIM_PERIOD;
    const frameIndex = Math.floor(phase * ANIM_FRAMES);

    for (let i = 0; i < this.pool.length; i++) {
      const p = this.pool[i];
      if (!p.inUse) continue;

      const prev = (p.sprite as any).__animFrame;
      if (prev !== frameIndex) {
        (p.sprite as any).__animFrame = frameIndex;
        const bodyPlan = (p.sprite as any).__bodyPlan || "grazer";
        const textures = generateTextures(bodyPlan);
        p.sprite.texture = textures.frames[frameIndex];
      }
    }
  }

  destroy() {
    for (const entry of textureCache.values()) {
      for (const frame of entry.frames) frame.destroy(true);
      entry.icon.destroy(true);
    }
    textureCache.clear();
  }
}
