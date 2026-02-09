import type { AgentSnapshot, SpeciesInfo } from "../types/simulation";

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

export function renderAgents(
  ctx: CanvasRenderingContext2D,
  agents: AgentSnapshot[],
  species: SpeciesInfo[],
  worldSize: number
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
    const radius = Math.max(2, 4);

    // State indicator border
    const borderColor = STATE_BORDER_COLORS[agent.state];
    if (borderColor) {
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
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
  speciesCount: number
) {
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(5, 5, 200, 80);

  ctx.fillStyle = "#fff";
  ctx.font = "12px monospace";
  ctx.fillText(`Tick: ${tick}`, 12, 22);
  ctx.fillText(`Epoch: ${EPOCH_NAMES[epoch] || epoch}`, 12, 38);
  ctx.fillText(`Agents: ${agentCount}`, 12, 54);
  ctx.fillText(`Species: ${speciesCount}`, 12, 70);
}
