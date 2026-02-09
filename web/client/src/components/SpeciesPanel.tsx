import type { SpeciesInfo } from "../types/simulation";

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

interface Props {
  species: SpeciesInfo[];
}

export function SpeciesPanel({ species }: Props) {
  return (
    <div
      style={{
        width: "260px",
        background: "#0f3460",
        borderRadius: "4px",
        padding: "12px",
        overflowY: "auto",
        maxHeight: "820px",
      }}
    >
      <h3 style={{ marginBottom: "12px", fontSize: "14px", color: "#e94560" }}>
        Species ({species.length})
      </h3>
      {species.map((sp) => (
        <div
          key={sp.sid}
          style={{
            marginBottom: "8px",
            padding: "8px",
            background: "#1a1a2e",
            borderRadius: "4px",
            borderLeft: `3px solid ${BODY_PLAN_COLORS[sp.plan] || "#aaa"}`,
          }}
        >
          <div style={{ fontSize: "13px", fontWeight: "bold" }}>
            S{sp.sid}: {sp.plan}
          </div>
          <div style={{ fontSize: "12px", color: "#aaa" }}>
            Pop: {sp.population.toLocaleString()}
          </div>
          <div style={{ fontSize: "11px", color: "#777" }}>
            {sp.locations.join(", ")}
          </div>
        </div>
      ))}
      {species.length === 0 && (
        <div style={{ color: "#666", fontSize: "12px" }}>
          No species data yet. Start a simulation.
        </div>
      )}
    </div>
  );
}
