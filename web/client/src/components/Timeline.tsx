interface EpochSegment {
  epoch: number;
  startIndex: number;
  endIndex: number;
}

interface Props {
  historyLength: number;
  viewIndex: number | null;
  currentEpoch: number;
  isFollowing: boolean;
  epochs: EpochSegment[];
  onSeek: (index: number) => void;
  onFollowLatest: () => void;
}

const EPOCH_NAMES: Record<number, string> = {
  1: "Primordial",
  2: "Cambrian",
  3: "Ecosystem",
  4: "Intelligence",
};

const EPOCH_COLORS: Record<number, string> = {
  1: "#ff6633",
  2: "#3399ff",
  3: "#33cc66",
  4: "#ffcc33",
};

export function Timeline({
  historyLength,
  viewIndex,
  currentEpoch,
  isFollowing,
  epochs,
  onSeek,
  onFollowLatest,
}: Props) {
  if (historyLength === 0) return null;

  const displayIndex = viewIndex ?? historyLength - 1;
  const max = historyLength - 1;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      onSeek(Math.max(0, displayIndex - 1));
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      if (displayIndex >= max) onFollowLatest();
      else onSeek(Math.min(max, displayIndex + 1));
    } else if (e.key === "End") {
      e.preventDefault();
      onFollowLatest();
    } else if (e.key === "Home") {
      e.preventDefault();
      onSeek(0);
    }
  };

  // Build gradient for epoch-colored slider track
  const trackGradient = epochs.length > 1 && max > 0
    ? epochs.map((seg) => {
        const color = EPOCH_COLORS[seg.epoch] || "#555";
        const start = (seg.startIndex / max) * 100;
        const end = (seg.endIndex / max) * 100;
        return `${color} ${start}%, ${color} ${end}%`;
      }).join(", ")
    : undefined;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
        <span style={{ fontFamily: "monospace", fontSize: "17px", color: "#999" }}>
          {displayIndex + 1} / {historyLength}
        </span>

        {isFollowing && (
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{
              width: 6, height: 6,
              borderRadius: "50%",
              background: "#4ade80",
              boxShadow: "0 0 6px #4ade80",
              display: "inline-block",
              animation: "pulse 2s ease infinite",
            }} />
            <span style={{ fontSize: "16px", color: "#4ade80", fontWeight: 600, letterSpacing: "0.5px" }}>
              LIVE
            </span>
          </div>
        )}

        {!isFollowing && (
          <>
            <span style={{
              fontSize: "14px",
              fontWeight: 700,
              color: "#f59e0b",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
            }}>
              Reviewing
            </span>
            <button onClick={onFollowLatest} style={followBtn}>
              Follow Live
            </button>
          </>
        )}

        {/* Epoch labels */}
        {epochs.length > 0 && (
          <div style={{ marginLeft: "auto", display: "flex", gap: 8, fontSize: "14px" }}>
            {epochs.map((seg) => (
              <span
                key={`${seg.epoch}-${seg.startIndex}`}
                style={{
                  color: EPOCH_COLORS[seg.epoch] || "#666",
                  fontWeight: seg.epoch === currentEpoch ? 700 : 400,
                  opacity: seg.epoch === currentEpoch ? 1 : 0.5,
                }}
              >
                {EPOCH_NAMES[seg.epoch] || `E${seg.epoch}`}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Epoch color bar */}
      {epochs.length > 1 && max > 0 && (
        <div style={{ display: "flex", height: 3, borderRadius: 2, overflow: "hidden", marginBottom: 4 }}>
          {epochs.map((seg) => (
            <div
              key={`${seg.epoch}-${seg.startIndex}`}
              style={{
                flex: seg.endIndex - seg.startIndex + 1,
                background: EPOCH_COLORS[seg.epoch] || "#555",
                opacity: 0.5,
              }}
            />
          ))}
        </div>
      )}

      <input
        type="range"
        min={0}
        max={max}
        value={displayIndex}
        onChange={(e) => onSeek(Number(e.target.value))}
        onKeyDown={handleKeyDown}
        style={{
          width: "100%",
          accentColor: EPOCH_COLORS[currentEpoch] || "#e94560",
          cursor: "pointer",
          height: "4px",
        }}
      />

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
}

const followBtn: React.CSSProperties = {
  padding: "3px 12px",
  background: "#f59e0b",
  color: "#000",
  border: "none",
  borderRadius: "4px",
  cursor: "pointer",
  fontWeight: 700,
  fontSize: "16px",
  fontFamily: "inherit",
};
