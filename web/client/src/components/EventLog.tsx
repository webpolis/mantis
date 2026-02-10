import type { EventLogEntry } from "../hooks/useWebSocket";

const EVENT_COLORS: Record<string, string> = {
  catastrophe: "#ff6644",
  catastrophe_end: "#88aa44",
  disease: "#ffcc44",
  symbiogenesis: "#44dddd",
  evo_trap: "#cc88ff",
  extinction: "#ff4444",
  speciation: "#44ff88",
  body_plan: "#88ccff",
};

interface Props {
  eventLog: EventLogEntry[];
}

export function EventLog({ eventLog }: Props) {
  const recent = eventLog.slice(-50).reverse();

  return (
    <div
      style={{
        width: "260px",
        background: "#0f3460",
        borderRadius: "4px",
        padding: "12px",
        overflowY: "auto",
        maxHeight: "300px",
        marginTop: "8px",
      }}
    >
      <h3 style={{ marginBottom: "8px", fontSize: "14px", color: "#e94560" }}>
        Events ({eventLog.length})
      </h3>
      {recent.map((entry, i) => {
        const color = EVENT_COLORS[entry.event.event_type] ?? "#999";
        return (
          <div
            key={`${entry.tick}-${entry.event.event_type}-${i}`}
            style={{
              fontSize: "11px",
              fontFamily: "monospace",
              marginBottom: "3px",
              lineHeight: "1.4",
            }}
          >
            <span style={{ color: "#666" }}>[T{entry.tick}]</span>{" "}
            <span style={{ color: "#888" }}>{entry.event.target}</span>{" "}
            <span style={{ color, fontWeight: "bold" }}>{entry.event.event_type}</span>
            {entry.event.detail && (
              <span style={{ color: "#aaa" }}>: {entry.event.detail}</span>
            )}
          </div>
        );
      })}
      {eventLog.length === 0 && (
        <div style={{ color: "#666", fontSize: "12px" }}>
          No events yet.
        </div>
      )}
    </div>
  );
}
