import { useEffect, useMemo, useRef } from "react";
import type { AgentSnapshot, SpeciesInfo } from "../types/simulation";
import { getCreatureIconDataURL, BODY_PLAN_COLORS } from "../pixi/creatures";

interface Props {
  species: SpeciesInfo[];
  agents: AgentSnapshot[];
  populationHistory: Map<number, number[]>;
}

// Cache creature icon data URLs
const iconCache = new Map<string, string>();
function getIcon(plan: string): string {
  if (!iconCache.has(plan)) {
    iconCache.set(plan, getCreatureIconDataURL(plan));
  }
  return iconCache.get(plan)!;
}

export function SpeciesPanel({ species, agents, populationHistory }: Props) {
  // Count visible agents per species (respecting count field for aggregated agents)
  const agentCounts = useMemo(() => {
    const counts = new Map<number, number>();
    for (const a of agents) {
      if (a.dead) continue;
      counts.set(a.species_sid, (counts.get(a.species_sid) || 0) + (a.count || 1));
    }
    return counts;
  }, [agents]);

  return (
    <div style={{ minWidth: 240 }}>
      <h3 style={{ marginBottom: 8, fontSize: "15px", fontWeight: 700, color: "#e94560", letterSpacing: "1px", textTransform: "uppercase" }}>
        Species ({species.length})
      </h3>
      {species.map((sp) => {
        const visibleCount = agentCounts.get(sp.sid) || 0;
        const history = populationHistory.get(sp.sid) || [];
        const prevPop = history.length >= 2 ? history[history.length - 2] : sp.population;
        const delta = sp.population - prevPop;
        const color = BODY_PLAN_COLORS[sp.plan] || "#aaa";

        return (
          <div
            key={sp.sid}
            style={{
              marginBottom: 6,
              padding: "7px 10px",
              background: "rgba(255, 255, 255, 0.06)",
              borderRadius: "6px",
              borderLeft: `3px solid ${color}`,
              display: "flex",
              alignItems: "center",
              gap: 8,
              transition: "background 0.2s ease",
            }}
          >
            <img
              src={getIcon(sp.plan)}
              width={28}
              height={28}
              style={{ imageRendering: "pixelated", flexShrink: 0 }}
              alt={sp.plan}
            />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: "14px", fontWeight: 600 }}>S{sp.sid}</span>
                <span style={{ fontSize: "13px", color: "#999" }}>{sp.plan}</span>
                {sp.age > 0 && (
                  <span style={{ fontSize: "11px", color: "#777" }}>t{sp.age}</span>
                )}
                {delta !== 0 && (
                  <span style={{
                    fontSize: "12px",
                    fontWeight: 700,
                    color: delta > 0 ? "#4ade80" : "#f87171",
                    marginLeft: "auto",
                  }}>
                    {delta > 0 ? "+" : ""}{delta}
                  </span>
                )}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 2 }}>
                <span style={{ fontSize: "13px", color: "#bbb" }}>
                  {visibleCount > 0 && visibleCount !== sp.population
                    ? <>{visibleCount.toLocaleString()} <span style={{ color: "#666", fontSize: "12px" }}>/ {sp.population.toLocaleString()}</span></>
                    : sp.population.toLocaleString()
                  }
                </span>
                {history.length > 2 && (
                  <Sparkline data={history} color={color} />
                )}
              </div>
            </div>
          </div>
        );
      })}
      {species.length === 0 && (
        <div style={{ color: "#666", fontSize: "13px" }}>
          No species data yet. Start a simulation.
        </div>
      )}
    </div>
  );
}

function Sparkline({ data, color }: { data: number[]; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const w = 60, h = 16;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length < 2) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, w, h);

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((data[i] - min) / range) * (h - 2) - 1;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [data, color]);

  return <canvas ref={canvasRef} width={w} height={h} style={{ display: "block", opacity: 0.7 }} />;
}
