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
  selectedWorld,
  onSelectFile,
  onSelectWorld,
  models,
  selectedModel,
  onSelectModel,
}: Props) {
  const [filterAgents, setFilterAgents] = useState(false);

  const handleFilterToggle = () => {
    const next = !filterAgents;
    setFilterAgents(next);
    if (next && worldsWithAgents.length > 0 && !worldsWithAgents.includes(selectedWorld)) {
      onSelectWorld(worldsWithAgents[0]);
    }
  };

  const handleWorldChange = (value: number) => {
    if (filterAgents) {
      const clamped = Math.max(0, Math.min(value, worldCount - 1));
      // Snap to nearest world with agents
      if (worldsWithAgents.includes(clamped)) {
        onSelectWorld(clamped);
      } else {
        const direction = value >= selectedWorld ? 1 : -1;
        const next = direction === 1
          ? worldsWithAgents.find((i) => i >= clamped)
          : [...worldsWithAgents].reverse().find((i) => i <= clamped);
        if (next !== undefined) onSelectWorld(next);
      }
    } else {
      onSelectWorld(value);
    }
  };
  return (
    <div
      style={{
        padding: "12px 16px",
        background: "#0f3460",
        borderRadius: "4px",
        marginBottom: "8px",
        display: "flex",
        alignItems: "center",
        gap: "16px",
      }}
    >
      {!isPlaying ? (
        <>
          <button onClick={() => onPlay("live")} style={btnStyle}>
            Live Simulation
          </button>

          <div style={dividerStyle} />

          <select
            value={selectedFile ?? ""}
            onChange={(e) => e.target.value && onSelectFile(e.target.value)}
            style={selectStyle}
          >
            <option value="">Dataset...</option>
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} ({formatSize(d.size)})
              </option>
            ))}
          </select>

          {worldCount > 0 && (
            <label style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              World:
              <input
                type="number"
                min={0}
                max={worldCount - 1}
                value={selectedWorld}
                onChange={(e) => handleWorldChange(Number(e.target.value))}
                style={{ ...selectStyle, width: "60px" }}
              />
              <span style={{ color: "#888", fontSize: "12px" }}>
                / {filterAgents ? worldsWithAgents.length : worldCount}
              </span>
              <label
                style={{ display: "flex", alignItems: "center", gap: "4px", cursor: "pointer" }}
                title="Only show worlds that contain agents"
              >
                <input
                  type="checkbox"
                  checked={filterAgents}
                  onChange={handleFilterToggle}
                  style={{ accentColor: "#e94560" }}
                />
                <span style={{ fontSize: "12px", color: "#aaa" }}>Has agents</span>
              </label>
            </label>
          )}

          <button
            onClick={() => onPlay("file", selectedFile ?? undefined, selectedWorld)}
            disabled={!selectedFile}
            style={{
              ...btnStyle,
              background: selectedFile ? "#444" : "#333",
              opacity: selectedFile ? 1 : 0.5,
              cursor: selectedFile ? "pointer" : "not-allowed",
            }}
          >
            Play World
          </button>

          {models.length > 0 && (
            <>
              <div style={dividerStyle} />

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
                style={{
                  ...btnStyle,
                  background: selectedModel ? "#16a34a" : "#333",
                  opacity: selectedModel ? 1 : 0.5,
                  cursor: selectedModel ? "pointer" : "not-allowed",
                }}
              >
                Model Inference
              </button>
            </>
          )}
        </>
      ) : (
        <button onClick={onPause} style={btnStyle}>
          Pause
        </button>
      )}

      {!isPlaying && (
        <button onClick={onResume} style={{ ...btnStyle, background: "#555" }}>
          Resume
        </button>
      )}

      <label style={{ display: "flex", alignItems: "center", gap: "6px" }}>
        Speed:
        <select
          onChange={(e) => onSpeed(Number(e.target.value))}
          defaultValue="1"
          style={selectStyle}
        >
          <option value="0.5">0.5x</option>
          <option value="1">1x</option>
          <option value="2">2x</option>
          <option value="5">5x</option>
          <option value="10">10x</option>
        </select>
      </label>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: "8px 16px",
  background: "#e94560",
  color: "#fff",
  border: "none",
  borderRadius: "4px",
  cursor: "pointer",
  fontWeight: "bold",
  fontSize: "14px",
};

const selectStyle: React.CSSProperties = {
  padding: "4px 8px",
  background: "#1a1a2e",
  color: "#eee",
  border: "1px solid #555",
  borderRadius: "4px",
};

const dividerStyle: React.CSSProperties = {
  width: "1px",
  height: "28px",
  background: "#555",
};
