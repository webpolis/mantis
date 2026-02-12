import { useEffect, useMemo, useRef, useState } from "react";
import type { AgentSnapshot, SpeciesInfo } from "../types/simulation";
import { getCreatureIconDataURL, BODY_PLAN_COLORS } from "../pixi/creatures";

interface Props {
  species: SpeciesInfo[];
  agents: AgentSnapshot[];
  populationHistory: Map<number, number[]>;
}

const iconCache = new Map<string, string>();
function getIcon(plan: string): string {
  if (!iconCache.has(plan)) {
    iconCache.set(plan, getCreatureIconDataURL(plan));
  }
  return iconCache.get(plan)!;
}

// Trait tier colors for the bars
const TRAIT_TIER: Record<string, { tier: number; color: string }> = {
  size: { tier: 0, color: "#66cc88" }, speed: { tier: 0, color: "#66cc88" },
  armor: { tier: 0, color: "#66cc88" }, metab: { tier: 0, color: "#66cc88" },
  sense: { tier: 0, color: "#66cc88" }, camo: { tier: 0, color: "#66cc88" },
  repro: { tier: 0, color: "#66cc88" }, regen: { tier: 0, color: "#66cc88" },
  venom: { tier: 0, color: "#66cc88" }, photosynth: { tier: 0, color: "#66cc88" },
  mouth: { tier: 0, color: "#66cc88" }, endurance: { tier: 0, color: "#66cc88" },
  chem_digest: { tier: 0, color: "#66cc88" }, toxin_resist: { tier: 0, color: "#66cc88" },
  toxin: { tier: 0, color: "#66cc88" },
  social: { tier: 1, color: "#55aadd" }, aggression: { tier: 1, color: "#55aadd" },
  curiosity: { tier: 1, color: "#55aadd" }, patience: { tier: 1, color: "#55aadd" },
  nocturnal: { tier: 1, color: "#55aadd" },
  intel: { tier: 2, color: "#aa77dd" }, memory: { tier: 2, color: "#aa77dd" },
  learning: { tier: 2, color: "#aa77dd" }, planning: { tier: 2, color: "#aa77dd" },
  deception: { tier: 2, color: "#aa77dd" },
  language: { tier: 3, color: "#dd7744" }, tooluse: { tier: 3, color: "#dd7744" },
  ritual: { tier: 3, color: "#dd7744" }, teaching: { tier: 3, color: "#dd7744" },
  trade: { tier: 3, color: "#dd7744" },
  subconscious: { tier: 4, color: "#ee5577" }, theory_of_mind: { tier: 4, color: "#ee5577" },
  creativity: { tier: 4, color: "#ee5577" }, abstraction: { tier: 4, color: "#ee5577" },
  ethics: { tier: 4, color: "#ee5577" },
};

const DIET_COLORS: Record<string, string> = {
  detritus: "#bb9944", det: "#bb9944",
  plant: "#44bb66", plt: "#44bb66",
  solar: "#eedd44", sol: "#eedd44",
  chemical: "#77aacc",
  meat: "#cc3344",
  none: "#666",
};

export function SpeciesPanel({ species, agents, populationHistory }: Props) {
  const [expandedSid, setExpandedSid] = useState<number | null>(null);

  const agentCounts = useMemo(() => {
    const counts = new Map<number, number>();
    for (const a of agents) {
      if (a.dead) continue;
      counts.set(a.species_sid, (counts.get(a.species_sid) || 0) + (a.count || 1));
    }
    return counts;
  }, [agents]);

  return (
    <div style={{ minWidth: 250 }}>
      <h3 style={{ marginBottom: 8, fontSize: "18px", fontWeight: 700, color: "#e94560", letterSpacing: "1px", textTransform: "uppercase" }}>
        Species ({species.length})
      </h3>
      {species.map((sp) => {
        const history = populationHistory.get(sp.sid) || [];
        const prevPop = history.length >= 2 ? history[history.length - 2] : sp.population;
        const delta = sp.population - prevPop;
        const color = BODY_PLAN_COLORS[sp.plan] || "#aaa";
        const isExpanded = expandedSid === sp.sid;
        const hasTraits = Object.keys(sp.traits ?? {}).length > 0;
        const hasDiet = Object.keys(sp.diet ?? {}).length > 0;

        return (
          <div
            key={sp.sid}
            onClick={() => setExpandedSid(isExpanded ? null : sp.sid)}
            style={{
              marginBottom: 5,
              padding: "6px 8px",
              background: isExpanded ? "rgba(255, 255, 255, 0.08)" : "rgba(255, 255, 255, 0.04)",
              borderRadius: "6px",
              borderLeft: `3px solid ${color}`,
              cursor: "pointer",
              transition: "background 0.15s ease",
            }}
          >
            {/* Header row */}
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <img
                src={getIcon(sp.plan)}
                width={24} height={24}
                style={{ imageRendering: "pixelated", flexShrink: 0 }}
                alt={sp.plan}
              />
              <span style={{ fontSize: "17px", fontWeight: 700, color: "#ddd" }}>S{sp.sid}</span>
              <span style={{ fontSize: "16px", color: "#888" }}>{sp.plan}</span>
              {sp.repro_strategy && (
                <span style={{
                  fontSize: "13px", fontWeight: 700,
                  color: sp.repro_strategy === "r" ? "#4ade80" : "#60a5fa",
                  background: sp.repro_strategy === "r" ? "rgba(74,222,128,0.12)" : "rgba(96,165,250,0.12)",
                  padding: "1px 4px", borderRadius: "3px",
                }}>{sp.repro_strategy}</span>
              )}
              {sp.age > 0 && (
                <span style={{ fontSize: "13px", color: "#666" }}>t{sp.age}</span>
              )}
              <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 4 }}>
                {delta !== 0 && (
                  <span style={{
                    fontSize: "14px", fontWeight: 700,
                    color: delta > 0 ? "#4ade80" : "#f87171",
                  }}>{delta > 0 ? "+" : ""}{delta}</span>
                )}
                <span style={{ fontSize: "16px", color: "#bbb", fontWeight: 600 }}>
                  {sp.population.toLocaleString()}
                </span>
              </span>
            </div>

            {/* Compact stat bar row (always visible) */}
            <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 4 }}>
              {/* Mini energy gauge */}
              <EnergyMiniBar energyIn={sp.energy_in} energyOut={sp.energy_out} />
              {/* Diet mini dots */}
              {hasDiet && <DietMiniBar diet={sp.diet} />}
              {/* Sparkline */}
              {history.length > 2 && (
                <span style={{ marginLeft: "auto" }}>
                  <Sparkline data={history} color={color} />
                </span>
              )}
            </div>

            {/* Expanded details */}
            {isExpanded && (
              <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid rgba(255,255,255,0.06)" }}
                onClick={(e) => e.stopPropagation()}>
                {/* Trait bars */}
                {hasTraits && <TraitBars traits={sp.traits} />}
                {/* Diet breakdown */}
                {hasDiet && <DietChart diet={sp.diet} />}
                {/* Energy details */}
                {(sp.energy_in > 0 || sp.energy_out > 0) && (
                  <EnergyDetail
                    energyIn={sp.energy_in} energyOut={sp.energy_out}
                    energyStore={sp.energy_store}
                  />
                )}
              </div>
            )}
          </div>
        );
      })}
      {species.length === 0 && (
        <div style={{ color: "#666", fontSize: "17px" }}>
          No species data yet. Start a simulation.
        </div>
      )}
    </div>
  );
}

/* ----------- Sub-components ----------- */

function TraitBars({ traits }: { traits: Record<string, number> }) {
  const sorted = Object.entries(traits).sort((a, b) => {
    const ta = TRAIT_TIER[a[0]]?.tier ?? 5;
    const tb = TRAIT_TIER[b[0]]?.tier ?? 5;
    return ta !== tb ? ta - tb : b[1] - a[1];
  });

  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: "13px", color: "#666", textTransform: "uppercase", marginBottom: 3, letterSpacing: "0.5px" }}>
        Traits
      </div>
      {sorted.map(([name, val]) => {
        const info = TRAIT_TIER[name] ?? TRAIT_TIER[name.replace("*", "")] ?? { tier: 5, color: "#888" };
        const pct = Math.min(100, (val / 10) * 100);
        return (
          <div key={name} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
            <span style={{ fontSize: "13px", color: "#999", width: 62, textAlign: "right", flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {name.replace("*", "")}
            </span>
            <div style={{
              flex: 1, height: 6, background: "rgba(255,255,255,0.06)",
              borderRadius: 3, overflow: "hidden",
            }}>
              <div style={{
                width: `${pct}%`, height: "100%",
                background: `linear-gradient(90deg, ${info.color}88, ${info.color})`,
                borderRadius: 3,
                transition: "width 0.3s ease",
              }} />
            </div>
            <span style={{ fontSize: "13px", color: "#888", width: 24, textAlign: "right", flexShrink: 0 }}>
              {val.toFixed(1)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function DietMiniBar({ diet }: { diet: Record<string, number> }) {
  const entries = Object.entries(diet).filter(([, v]) => v >= 0.01).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return null;

  return (
    <div style={{
      display: "flex", height: 5, borderRadius: 3, overflow: "hidden",
      width: 50, flexShrink: 0,
    }}>
      {entries.map(([src, pct]) => (
        <div key={src} style={{
          width: `${pct * 100}%`,
          background: DIET_COLORS[src] || "#666",
        }} />
      ))}
    </div>
  );
}

function DietChart({ diet }: { diet: Record<string, number> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const entries = Object.entries(diet).filter(([, v]) => v >= 0.01).sort((a, b) => b[1] - a[1]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || entries.length === 0) return;
    const ctx = canvas.getContext("2d")!;
    const size = 40;
    const cx = size / 2, cy = size / 2, r = 16;
    ctx.clearRect(0, 0, size, size);

    let angle = -Math.PI / 2;
    for (const [src, pct] of entries) {
      const sweep = pct * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r, angle, angle + sweep);
      ctx.closePath();
      ctx.fillStyle = DIET_COLORS[src] || "#666";
      ctx.fill();
      angle += sweep;
    }
    // Center hole for donut effect
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#1a1a2e";
    ctx.fill();
  }, [entries]);

  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: "13px", color: "#666", textTransform: "uppercase", marginBottom: 3, letterSpacing: "0.5px" }}>
        Diet
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <canvas ref={canvasRef} width={40} height={40} style={{ flexShrink: 0 }} />
        <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
          {entries.map(([src, pct]) => (
            <div key={src} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: "13px" }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: DIET_COLORS[src] || "#666", flexShrink: 0 }} />
              <span style={{ color: "#aaa" }}>{src}</span>
              <span style={{ color: "#777" }}>{Math.round(pct * 100)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function EnergyMiniBar({ energyIn, energyOut }: { energyIn: number; energyOut: number }) {
  if (energyIn <= 0 && energyOut <= 0) return null;
  const max = Math.max(energyIn, energyOut, 1);
  const ratio = energyIn / max;
  const deficit = energyOut > energyIn;

  return (
    <div style={{
      width: 30, height: 5, borderRadius: 3, overflow: "hidden",
      background: "rgba(255,255,255,0.06)", flexShrink: 0,
      position: "relative",
    }}>
      <div style={{
        width: `${ratio * 100}%`, height: "100%",
        background: deficit ? "#f87171" : "#4ade80",
        borderRadius: 3,
        transition: "width 0.3s ease",
      }} />
    </div>
  );
}

function EnergyDetail({ energyIn, energyOut, energyStore }: {
  energyIn: number; energyOut: number; energyStore: number;
}) {
  const net = energyIn - energyOut;
  const max = Math.max(energyIn, energyOut, 1);

  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ fontSize: "13px", color: "#666", textTransform: "uppercase", marginBottom: 3, letterSpacing: "0.5px" }}>
        Energy
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {/* Income bar */}
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ fontSize: "13px", color: "#4ade80", width: 14, textAlign: "right" }}>+</span>
          <div style={{ flex: 1, height: 5, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ width: `${(energyIn / max) * 100}%`, height: "100%", background: "#4ade80", borderRadius: 3 }} />
          </div>
          <span style={{ fontSize: "12px", color: "#888", width: 32, textAlign: "right" }}>{fmtNum(energyIn)}</span>
        </div>
        {/* Cost bar */}
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ fontSize: "13px", color: "#f87171", width: 14, textAlign: "right" }}>-</span>
          <div style={{ flex: 1, height: 5, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ width: `${(energyOut / max) * 100}%`, height: "100%", background: "#f87171", borderRadius: 3 }} />
          </div>
          <span style={{ fontSize: "12px", color: "#888", width: 32, textAlign: "right" }}>{fmtNum(energyOut)}</span>
        </div>
        {/* Net + Store */}
        <div style={{ display: "flex", gap: 8, fontSize: "13px", marginTop: 1 }}>
          <span style={{ color: net >= 0 ? "#4ade80" : "#f87171" }}>
            Net: {net >= 0 ? "+" : ""}{fmtNum(net)}
          </span>
          {energyStore > 0 && (
            <span style={{ color: "#eebb44" }}>Store: {fmtNum(energyStore)}</span>
          )}
        </div>
      </div>
    </div>
  );
}

function fmtNum(n: number): string {
  if (Math.abs(n) >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return Math.round(n).toString();
}

function Sparkline({ data, color }: { data: number[]; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const w = 50, h = 14;

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
