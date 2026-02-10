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

const EPOCH_COLORS = [
  "#e94560",
  "#16a34a",
  "#3b82f6",
  "#f59e0b",
  "#8b5cf6",
  "#06b6d4",
  "#ec4899",
  "#84cc16",
];

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
      if (displayIndex >= max) {
        onFollowLatest();
      } else {
        onSeek(Math.min(max, displayIndex + 1));
      }
    } else if (e.key === "End") {
      e.preventDefault();
      onFollowLatest();
    } else if (e.key === "Home") {
      e.preventDefault();
      onSeek(0);
    }
  };

  return (
    <div
      style={{
        padding: "8px 16px",
        background: "#0f3460",
        borderRadius: "4px",
        marginBottom: "8px",
        border: isFollowing ? "1px solid transparent" : "1px solid #f59e0b",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          marginBottom: "4px",
        }}
      >
        <span style={{ fontFamily: "monospace", fontSize: "13px", color: "#ccc", minWidth: "120px" }}>
          Frame {displayIndex + 1} / {historyLength}
        </span>
        <span style={{ fontFamily: "monospace", fontSize: "13px", color: EPOCH_COLORS[(currentEpoch - 1) % EPOCH_COLORS.length] }}>
          Epoch {currentEpoch}
        </span>
        {!isFollowing && (
          <>
            <span
              style={{
                fontSize: "11px",
                fontWeight: "bold",
                color: "#f59e0b",
                textTransform: "uppercase",
                letterSpacing: "0.5px",
              }}
            >
              Reviewing
            </span>
            <button onClick={onFollowLatest} style={followBtnStyle}>
              Follow Live
            </button>
          </>
        )}
      </div>

      {/* Epoch color bar */}
      {epochs.length > 1 && max > 0 && (
        <div
          style={{
            display: "flex",
            height: "4px",
            borderRadius: "2px",
            overflow: "hidden",
            marginBottom: "4px",
          }}
        >
          {epochs.map((seg) => (
            <div
              key={`${seg.epoch}-${seg.startIndex}`}
              style={{
                flex: seg.endIndex - seg.startIndex + 1,
                background: EPOCH_COLORS[(seg.epoch - 1) % EPOCH_COLORS.length],
                opacity: 0.6,
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
          accentColor: "#e94560",
          cursor: "pointer",
        }}
      />
    </div>
  );
}

const followBtnStyle: React.CSSProperties = {
  padding: "3px 10px",
  background: "#f59e0b",
  color: "#000",
  border: "none",
  borderRadius: "3px",
  cursor: "pointer",
  fontWeight: "bold",
  fontSize: "12px",
  marginLeft: "auto",
};
