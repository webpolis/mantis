import type { AgentSnapshot, SpeciesInfo } from "../types/simulation";

export interface Camera {
  zoom: number;
  offsetX: number;
  offsetY: number;
}

export const DEFAULT_CAMERA: Camera = { zoom: 1, offsetX: 0, offsetY: 0 };

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

    // Draw agent
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

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
