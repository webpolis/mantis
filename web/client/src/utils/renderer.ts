import type { AgentSnapshot, SpeciesInfo, BiomeData, VegetationPatchData } from "../types/simulation";

export interface Camera {
  zoom: number;
  offsetX: number;
  offsetY: number;
}

export const DEFAULT_CAMERA: Camera = { zoom: 1, offsetX: 0, offsetY: 0 };

// --- Biome tints (subtle rgba overlays on the dark background) ---
const BIOME_TINTS: Record<string, [number, number, number, number]> = {
  shallows:       [100, 180, 220, 0.15],
  reef:           [60, 200, 180, 0.12],
  deep_ocean:     [20, 40, 100, 0.18],
  tidal_pools:    [80, 160, 200, 0.10],
  mangrove:       [60, 120, 60, 0.14],
  savanna:        [180, 160, 80, 0.12],
  forest:         [30, 90, 30, 0.16],
  rainforest:     [20, 100, 40, 0.18],
  desert:         [200, 170, 100, 0.14],
  tundra:         [160, 180, 200, 0.12],
  cave:           [50, 40, 60, 0.20],
  volcanic_vent:  [180, 60, 20, 0.14],
  meadow:         [100, 180, 80, 0.12],
  swamp:          [60, 80, 40, 0.16],
  alpine:         [140, 160, 190, 0.12],
};

/** Seeded PRNG (mulberry32) for deterministic biome centers. */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Compute deterministic center for each biome using lid as seed. */
function biomeCenters(biomes: BiomeData[], size: number): Map<number, [number, number]> {
  const centers = new Map<number, [number, number]>();
  for (const b of biomes) {
    const rng = mulberry32(b.lid * 7919 + 31);
    const margin = size * 0.15;
    const x = margin + rng() * (size - 2 * margin);
    const y = margin + rng() * (size - 2 * margin);
    centers.set(b.lid, [x, y]);
  }
  return centers;
}

/**
 * Build an offscreen canvas with Voronoi biome regions.
 * Renders at a low resolution (GRID_RES) and upscales for speed.
 */
const GRID_RES = 200;

export function buildBiomeTexture(
  biomes: BiomeData[],
  canvasSize: number
): HTMLCanvasElement | null {
  if (biomes.length === 0) return null;

  const offscreen = document.createElement("canvas");
  offscreen.width = canvasSize;
  offscreen.height = canvasSize;
  const ctx = offscreen.getContext("2d");
  if (!ctx) return null;

  // Build low-res Voronoi grid
  const centers = biomeCenters(biomes, GRID_RES);
  const biomeMap = new Map(biomes.map((b) => [b.lid, b]));

  const small = document.createElement("canvas");
  small.width = GRID_RES;
  small.height = GRID_RES;
  const sctx = small.getContext("2d");
  if (!sctx) return null;

  const imgData = sctx.createImageData(GRID_RES, GRID_RES);
  const data = imgData.data;

  const centerList = Array.from(centers.entries());

  for (let py = 0; py < GRID_RES; py++) {
    for (let px = 0; px < GRID_RES; px++) {
      let bestLid = centerList[0][0];
      let bestDist = Infinity;
      for (const [lid, [cx, cy]] of centerList) {
        const dx = px - cx;
        const dy = py - cy;
        const d = dx * dx + dy * dy;
        if (d < bestDist) {
          bestDist = d;
          bestLid = lid;
        }
      }

      const biome = biomeMap.get(bestLid);
      const tint = biome ? (BIOME_TINTS[biome.name] || [80, 80, 80, 0.1]) : [80, 80, 80, 0.1];
      const idx = (py * GRID_RES + px) * 4;
      data[idx] = tint[0];
      data[idx + 1] = tint[1];
      data[idx + 2] = tint[2];
      data[idx + 3] = Math.round(tint[3] * 255);
    }
  }

  sctx.putImageData(imgData, 0, 0);

  // Upscale with smoothing
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(small, 0, 0, canvasSize, canvasSize);

  return offscreen;
}

/** Render vegetation patches as soft green radial gradient splotches. */
export function renderVegetation(
  ctx: CanvasRenderingContext2D,
  biomes: BiomeData[],
  worldSize: number
) {
  const scale = ctx.canvas.width / worldSize;
  for (const biome of biomes) {
    for (const patch of biome.patches) {
      if (patch.density < 0.01) continue;
      const cx = patch.x * scale;
      const cy = patch.y * scale;
      const r = patch.radius * scale;
      const alpha = Math.min(0.35, patch.density * 0.4);
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      grad.addColorStop(0, `rgba(40, 180, 60, ${alpha})`);
      grad.addColorStop(0.6, `rgba(40, 180, 60, ${alpha * 0.4})`);
      grad.addColorStop(1, "rgba(40, 180, 60, 0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

/** Render the cached biome background texture. */
export function renderBiomeBackground(
  ctx: CanvasRenderingContext2D,
  texture: HTMLCanvasElement
) {
  ctx.drawImage(texture, 0, 0);
}

/** Draw a body-plan-specific shape at (x, y) with given radius and color. */
function drawAgentShape(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  bodyPlan: string,
  color: string
) {
  ctx.fillStyle = color;
  ctx.beginPath();

  switch (bodyPlan) {
    case "predator": {
      // Triangle pointing up
      ctx.moveTo(x, y - r);
      ctx.lineTo(x - r, y + r * 0.8);
      ctx.lineTo(x + r, y + r * 0.8);
      ctx.closePath();
      break;
    }
    case "omnivore": {
      // Diamond (rotated square)
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r, y);
      ctx.lineTo(x, y + r);
      ctx.lineTo(x - r, y);
      ctx.closePath();
      break;
    }
    case "scavenger": {
      // Pentagon
      for (let i = 0; i < 5; i++) {
        const angle = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
        const px = x + r * Math.cos(angle);
        const py = y + r * Math.sin(angle);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      break;
    }
    case "decomposer": {
      // Square
      ctx.rect(x - r * 0.75, y - r * 0.75, r * 1.5, r * 1.5);
      break;
    }
    case "sessile_autotroph": {
      // 6-pointed star
      for (let i = 0; i < 12; i++) {
        const angle = -Math.PI / 2 + (i * Math.PI) / 6;
        const dist = i % 2 === 0 ? r : r * 0.5;
        const px = x + dist * Math.cos(angle);
        const py = y + dist * Math.sin(angle);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      break;
    }
    case "mobile_autotroph": {
      // Teardrop / leaf
      ctx.moveTo(x, y - r);
      ctx.bezierCurveTo(x + r, y - r * 0.3, x + r * 0.6, y + r, x, y + r);
      ctx.bezierCurveTo(x - r * 0.6, y + r, x - r, y - r * 0.3, x, y - r);
      ctx.closePath();
      break;
    }
    case "filter_feeder": {
      // Cross / plus
      const arm = r * 0.35;
      ctx.moveTo(x - arm, y - r);
      ctx.lineTo(x + arm, y - r);
      ctx.lineTo(x + arm, y - arm);
      ctx.lineTo(x + r, y - arm);
      ctx.lineTo(x + r, y + arm);
      ctx.lineTo(x + arm, y + arm);
      ctx.lineTo(x + arm, y + r);
      ctx.lineTo(x - arm, y + r);
      ctx.lineTo(x - arm, y + arm);
      ctx.lineTo(x - r, y + arm);
      ctx.lineTo(x - r, y - arm);
      ctx.lineTo(x - arm, y - arm);
      ctx.closePath();
      break;
    }
    case "parasite": {
      // Small inverted triangle
      ctx.moveTo(x, y + r);
      ctx.lineTo(x - r * 0.8, y - r * 0.6);
      ctx.lineTo(x + r * 0.8, y - r * 0.6);
      ctx.closePath();
      break;
    }
    default: {
      // Grazer / fallback: circle
      ctx.arc(x, y, r, 0, Math.PI * 2);
      break;
    }
  }

  ctx.fill();
}

const BODY_PLAN_COLORS: Record<string, string> = {
  predator: "#ff4444",
  grazer: "#88cc88",
  omnivore: "#cc88ff",
  scavenger: "#ccaa66",
  decomposer: "#666666",
  sessile_autotroph: "#44ff44",
  mobile_autotroph: "#66ff66",
  filter_feeder: "#6688ff",
  parasite: "#ff88ff",
};

const STATE_BORDER_COLORS: Record<string, string> = {
  hunt: "#ff0000",
  flee: "#ff8800",
  mate: "#ff44ff",
  flock: "#44aaff",
};

function agentRadius(count: number): number {
  return Math.max(2, Math.min(8, Math.sqrt(count || 1) * 2));
}

/** Apply camera transform, call fn, then restore. */
export function withCamera(
  ctx: CanvasRenderingContext2D,
  cam: Camera,
  fn: () => void
) {
  ctx.save();
  ctx.translate(cam.offsetX, cam.offsetY);
  ctx.scale(cam.zoom, cam.zoom);
  fn();
  ctx.restore();
}

/** Convert screen-space canvas coords to base (pre-camera) canvas coords. */
export function screenToBase(
  screenX: number,
  screenY: number,
  cam: Camera
): [number, number] {
  return [
    (screenX - cam.offsetX) / cam.zoom,
    (screenY - cam.offsetY) / cam.zoom,
  ];
}

export function renderAgents(
  ctx: CanvasRenderingContext2D,
  agents: AgentSnapshot[],
  species: SpeciesInfo[],
  worldSize: number,
  hoveredAid?: number | null
) {
  const scale = ctx.canvas.width / worldSize;
  const speciesMap = new Map(species.map((s) => [s.sid, s]));

  for (const agent of agents) {
    if (agent.dead) continue;

    const sp = speciesMap.get(agent.species_sid);
    const bodyPlan = sp?.plan || "grazer";

    const x = agent.x * scale;
    const y = agent.y * scale;

    const color = BODY_PLAN_COLORS[bodyPlan] || "#aaaaaa";
    const radius = agentRadius(agent.count);

    // State indicator border
    const borderColor = STATE_BORDER_COLORS[agent.state];
    if (borderColor) {
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Hover highlight
    if (agent.aid === hoveredAid) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw agent shape based on body plan
    drawAgentShape(ctx, x, y, radius, bodyPlan, color);

    // Low energy indicator
    if (agent.energy < 20) {
      ctx.fillStyle = "#ffff00";
      const barWidth = 8;
      const barHeight = 2;
      ctx.fillRect(
        x - barWidth / 2,
        y - radius - 5,
        (agent.energy / 100) * barWidth,
        barHeight
      );
    }
  }
}

export function findAgentAt(
  canvasX: number,
  canvasY: number,
  agents: AgentSnapshot[],
  worldSize: number,
  canvasSize: number
): AgentSnapshot | null {
  const scale = canvasSize / worldSize;
  const hitSlop = 4;
  let closest: AgentSnapshot | null = null;
  let closestDist = Infinity;

  for (const agent of agents) {
    if (agent.dead) continue;
    const ax = agent.x * scale;
    const ay = agent.y * scale;
    const radius = agentRadius(agent.count);
    const dx = canvasX - ax;
    const dy = canvasY - ay;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist <= radius + hitSlop && dist < closestDist) {
      closest = agent;
      closestDist = dist;
    }
  }
  return closest;
}

export function findVegetationAt(
  canvasX: number,
  canvasY: number,
  biomes: BiomeData[],
  worldSize: number,
  canvasSize: number
): { patch: VegetationPatchData; biome: BiomeData } | null {
  const scale = canvasSize / worldSize;
  let closest: { patch: VegetationPatchData; biome: BiomeData } | null = null;
  let closestDist = Infinity;

  for (const biome of biomes) {
    for (const patch of biome.patches) {
      if (patch.density < 0.01) continue;
      const px = patch.x * scale;
      const py = patch.y * scale;
      const r = patch.radius * scale;
      const dx = canvasX - px;
      const dy = canvasY - py;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist <= r * 0.6 && dist < closestDist) {
        closest = { patch, biome };
        closestDist = dist;
      }
    }
  }
  return closest;
}

export function findBiomeAt(
  canvasX: number,
  canvasY: number,
  biomes: BiomeData[],
  canvasSize: number
): BiomeData | null {
  if (biomes.length === 0) return null;
  const centers = biomeCenters(biomes, canvasSize);
  let bestLid = biomes[0].lid;
  let bestDist = Infinity;
  for (const [lid, [cx, cy]] of centers.entries()) {
    const dx = canvasX - cx;
    const dy = canvasY - cy;
    const d = dx * dx + dy * dy;
    if (d < bestDist) {
      bestDist = d;
      bestLid = lid;
    }
  }
  return biomes.find((b) => b.lid === bestLid) ?? null;
}

export function renderGrid(
  ctx: CanvasRenderingContext2D,
  worldSize: number,
  cellSize: number = 100
) {
  const scale = ctx.canvas.width / worldSize;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 0.5;

  for (let x = 0; x <= worldSize; x += cellSize) {
    ctx.beginPath();
    ctx.moveTo(x * scale, 0);
    ctx.lineTo(x * scale, ctx.canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= worldSize; y += cellSize) {
    ctx.beginPath();
    ctx.moveTo(0, y * scale);
    ctx.lineTo(ctx.canvas.width, y * scale);
    ctx.stroke();
  }
}

const EPOCH_NAMES: Record<number, string> = {
  1: "Primordial",
  2: "Cambrian",
  3: "Ecosystem",
  4: "Intelligence",
};

export function renderHUD(
  ctx: CanvasRenderingContext2D,
  tick: number,
  epoch: number,
  agentCount: number,
  speciesCount: number,
  zoom?: number
) {
  const showZoom = zoom != null && zoom !== 1;
  const height = showZoom ? 96 : 80;

  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(5, 5, 200, height);

  ctx.fillStyle = "#fff";
  ctx.font = "12px monospace";
  ctx.fillText(`Tick: ${tick}`, 12, 22);
  ctx.fillText(`Epoch: ${EPOCH_NAMES[epoch] || epoch}`, 12, 38);
  ctx.fillText(`Agents: ${agentCount}`, 12, 54);
  ctx.fillText(`Species: ${speciesCount}`, 12, 70);
  if (showZoom) {
    ctx.fillStyle = "#8cf";
    ctx.fillText(`Zoom: ${zoom!.toFixed(1)}x`, 12, 86);
  }
}
