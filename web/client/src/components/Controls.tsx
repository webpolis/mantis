interface Props {
  onPlay: (mode: "file" | "live") => void;
  onPause: () => void;
  onResume: () => void;
  onSpeed: (speed: number) => void;
  isPlaying: boolean;
}

export function Controls({ onPlay, onPause, onResume, onSpeed, isPlaying }: Props) {
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
          <button onClick={() => onPlay("file")} style={{ ...btnStyle, background: "#444" }}>
            From File
          </button>
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
