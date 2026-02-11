import { useState } from "react";
import type { DatasetFile, ModelFile } from "../types/simulation";

interface Props {
  onPlay: (mode: "file" | "live" | "model", file?: string, worldIndex?: number) => void;
  onPause: () => void;
  onResume: () => void;
  onSpeed: (speed: number) => void;
  isPlaying: boolean;
  datasets: DatasetFile[];
  selectedFile: string | null;
  worldCount: number;
  worldsWithAgents: number[];
  worldsWithSpotlights: number[];
  selectedWorld: number;
  onSelectFile: (name: string) => void;
  onSelectWorld: (index: number) => void;
  models: ModelFile[];
  selectedModel: string | null;
  onSelectModel: (name: string) => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}K`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}M`;
}

const SPEEDS = [0.5, 1, 2, 5, 10];

export function Controls({
  onPlay,
  onPause,
  onResume,
  onSpeed,
  isPlaying,
  datasets,
  selectedFile,
  worldCount,
  worldsWithAgents,
  worldsWithSpotlights,
  selectedWorld,
  onSelectFile,
  onSelectWorld,
  models,
  selectedModel,
  onSelectModel,
}: Props) {
  const [speed, setLocalSpeed] = useState(1);
  const [filterAgents, setFilterAgents] = useState(false);
  const [filterSpotlights, setFilterSpotlights] = useState(false);
  const [showDataset, setShowDataset] = useState(false);

  const computeFiltered = (agents: boolean, spotlights: boolean): number[] | null => {
    if (!agents && !spotlights) return null;
    const all = Array.from({ length: worldCount }, (_, i) => i);
    return all.filter(
      (i) => (!agents || worldsWithAgents.includes(i)) && (!spotlights || worldsWithSpotlights.includes(i))
    );
  };

  const filteredWorlds = computeFiltered(filterAgents, filterSpotlights);

  const handleFilterToggle = (kind: "agents" | "spotlights") => {
    const nextAgents = kind === "agents" ? !filterAgents : filterAgents;
    const nextSpotlights = kind === "spotlights" ? !filterSpotlights : filterSpotlights;
    if (kind === "agents") setFilterAgents(nextAgents);
    else setFilterSpotlights(nextSpotlights);

    const next = computeFiltered(nextAgents, nextSpotlights);
    if (next && next.length > 0 && !next.includes(selectedWorld)) {
      onSelectWorld(next[0]);
    }
  };

  const handleWorldChange = (value: number) => {
    if (filteredWorlds) {
      const clamped = Math.max(0, Math.min(value, worldCount - 1));
      if (filteredWorlds.includes(clamped)) {
        onSelectWorld(clamped);
      } else {
        const direction = value >= selectedWorld ? 1 : -1;
        const next = direction === 1
          ? filteredWorlds.find((i) => i >= clamped)
          : [...filteredWorlds].reverse().find((i) => i <= clamped);
        if (next !== undefined) onSelectWorld(next);
      }
    } else {
      onSelectWorld(value);
    }
  };

  const handleSpeed = (s: number) => {
    setLocalSpeed(s);
    onSpeed(s);
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
      {!isPlaying ? (
        <>
          <button onClick={() => onPlay("live")} style={btnPrimary}>
            Live Simulation
          </button>

          <div style={divider} />

          <button
            onClick={() => setShowDataset(!showDataset)}
            style={{ ...btnGhost, color: selectedFile ? "#dde" : "#777" }}
          >
            {selectedFile || "Dataset..."}
          </button>

          {showDataset && (
            <select
              value={selectedFile ?? ""}
              onChange={(e) => { e.target.value && onSelectFile(e.target.value); setShowDataset(false); }}
              style={selectStyle}
              autoFocus
              onBlur={() => setShowDataset(false)}
            >
              <option value="">Select...</option>
              {datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} ({formatSize(d.size)})
                </option>
              ))}
            </select>
          )}

          {worldCount > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "14px" }}>
              <span style={{ color: "#999" }}>W:</span>
              <input
                type="number"
                min={0}
                max={worldCount - 1}
                value={selectedWorld}
                onChange={(e) => handleWorldChange(Number(e.target.value))}
                style={{ ...selectStyle, width: "52px", textAlign: "center" }}
              />
              <span style={{ color: "#666", fontSize: "13px" }}>
                /{filteredWorlds ? filteredWorlds.length : worldCount}
              </span>
              <label style={{ display: "flex", alignItems: "center", gap: 3, cursor: "pointer" }}>
                <input type="checkbox" checked={filterAgents} onChange={() => handleFilterToggle("agents")} style={{ accentColor: "#e94560" }} />
                <span style={{ fontSize: "13px", color: "#999" }}>A</span>
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 3, cursor: "pointer" }}>
                <input type="checkbox" checked={filterSpotlights} onChange={() => handleFilterToggle("spotlights")} style={{ accentColor: "#e94560" }} />
                <span style={{ fontSize: "13px", color: "#999" }}>S</span>
              </label>
            </div>
          )}

          <button
            onClick={() => onPlay("file", selectedFile ?? undefined, selectedWorld)}
            disabled={!selectedFile}
            style={selectedFile ? btnSecondary : { ...btnSecondary, opacity: 0.35, cursor: "not-allowed" }}
          >
            Play
          </button>

          {models.length > 0 && (
            <>
              <div style={divider} />
              <select
                value={selectedModel ?? ""}
                onChange={(e) => e.target.value && onSelectModel(e.target.value)}
                style={selectStyle}
              >
                <option value="">Model...</option>
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name} ({formatSize(m.size)})
                  </option>
                ))}
              </select>
              <button
                onClick={() => onPlay("model")}
                disabled={!selectedModel}
                style={selectedModel ? btnAccent : { ...btnAccent, opacity: 0.35, cursor: "not-allowed" }}
              >
                Infer
              </button>
            </>
          )}
        </>
      ) : (
        <button onClick={onPause} style={btnPrimary}>
          Pause
        </button>
      )}

      {!isPlaying && (
        <button onClick={onResume} style={btnGhost}>
          Resume
        </button>
      )}

      <div style={divider} />

      {/* Speed pill group */}
      <div style={{ display: "flex", borderRadius: "6px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.1)" }}>
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => handleSpeed(s)}
            style={{
              padding: "5px 12px",
              background: speed === s ? "rgba(233, 69, 96, 0.6)" : "transparent",
              color: speed === s ? "#fff" : "#999",
              border: "none",
              cursor: "pointer",
              fontSize: "14px",
              fontWeight: speed === s ? 700 : 400,
              fontFamily: "inherit",
              transition: "all 0.15s ease",
            }}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  );
}

const btnPrimary: React.CSSProperties = {
  padding: "7px 18px",
  background: "linear-gradient(135deg, #e94560, #c73450)",
  color: "#fff",
  border: "none",
  borderRadius: "6px",
  cursor: "pointer",
  fontWeight: 700,
  fontSize: "14px",
  fontFamily: "inherit",
  letterSpacing: "0.5px",
  boxShadow: "0 0 12px rgba(233, 69, 96, 0.3)",
  transition: "box-shadow 0.2s ease",
};

const btnSecondary: React.CSSProperties = {
  padding: "7px 16px",
  background: "rgba(255, 255, 255, 0.08)",
  color: "#dde",
  border: "1px solid rgba(255, 255, 255, 0.14)",
  borderRadius: "6px",
  cursor: "pointer",
  fontWeight: 600,
  fontSize: "14px",
  fontFamily: "inherit",
  transition: "all 0.15s ease",
};

const btnAccent: React.CSSProperties = {
  padding: "7px 16px",
  background: "linear-gradient(135deg, #16a34a, #128a3e)",
  color: "#fff",
  border: "none",
  borderRadius: "6px",
  cursor: "pointer",
  fontWeight: 700,
  fontSize: "14px",
  fontFamily: "inherit",
  boxShadow: "0 0 12px rgba(22, 163, 74, 0.3)",
};

const btnGhost: React.CSSProperties = {
  padding: "7px 14px",
  background: "transparent",
  color: "#aaa",
  border: "1px solid rgba(255, 255, 255, 0.1)",
  borderRadius: "6px",
  cursor: "pointer",
  fontWeight: 500,
  fontSize: "14px",
  fontFamily: "inherit",
};

const selectStyle: React.CSSProperties = {
  padding: "5px 10px",
  background: "rgba(10, 10, 20, 0.85)",
  color: "#dde",
  border: "1px solid rgba(255, 255, 255, 0.14)",
  borderRadius: "4px",
  fontSize: "14px",
  fontFamily: "inherit",
};

const divider: React.CSSProperties = {
  width: "1px",
  height: "24px",
  background: "rgba(255, 255, 255, 0.1)",
};
