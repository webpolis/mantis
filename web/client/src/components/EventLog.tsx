import { useEffect, useRef, useState } from "react";
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

const EVENT_ICONS: Record<string, string> = {
  catastrophe: "\u26a0",
  catastrophe_end: "\u2714",
  disease: "\u2620",
  symbiogenesis: "\u2696",
  evo_trap: "\u26d4",
  extinction: "\u2620",
  speciation: "\u2728",
  body_plan: "\ud83e\udde0",
};

interface Props {
  eventLog: EventLogEntry[];
  onSeekToTick: (tick: number) => void;
}

interface Toast {
  id: number;
  entry: EventLogEntry;
  exiting: boolean;
}

let toastId = 0;

export function EventLog({ eventLog, onSeekToTick }: Props) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [expanded, setExpanded] = useState(false);
  const prevLenRef = useRef(0);

  useEffect(() => {
    if (eventLog.length <= prevLenRef.current) {
      prevLenRef.current = eventLog.length;
      return;
    }

    const newEntries = eventLog.slice(prevLenRef.current);
    prevLenRef.current = eventLog.length;

    const newToasts = newEntries.map((entry) => ({
      id: ++toastId,
      entry,
      exiting: false,
    }));

    setToasts((prev) => [...prev, ...newToasts].slice(-6));

    // Auto dismiss after 4s (collect timers for cleanup)
    const timers: number[] = [];
    for (const t of newToasts) {
      const outer = window.setTimeout(() => {
        setToasts((prev) => prev.map((p) => p.id === t.id ? { ...p, exiting: true } : p));
        const inner = window.setTimeout(() => {
          setToasts((prev) => prev.filter((p) => p.id !== t.id));
        }, 300);
        timers.push(inner);
      }, t.entry.event.event_type === "catastrophe" ? 6000 : 4000);
      timers.push(outer);
    }

    return () => timers.forEach(clearTimeout);
  }, [eventLog]);

  return (
    <div style={{ minWidth: 240 }}>
      {/* Toast stack */}
      {toasts.length > 0 && !expanded && (
        <div style={{ marginBottom: 8, display: "flex", flexDirection: "column", gap: 4 }}>
          {toasts.map((t) => {
            const isCatastrophe = t.entry.event.event_type === "catastrophe";
            const color = EVENT_COLORS[t.entry.event.event_type] ?? "#999";
            return (
              <div
                key={t.id}
                onClick={() => onSeekToTick(t.entry.tick)}
                style={{
                  padding: "7px 10px",
                  background: isCatastrophe ? "rgba(255, 60, 30, 0.18)" : "rgba(255, 255, 255, 0.06)",
                  borderRadius: "6px",
                  border: isCatastrophe ? `1px solid ${color}55` : "1px solid rgba(255,255,255,0.06)",
                  boxShadow: isCatastrophe ? `0 0 12px ${color}33` : "none",
                  fontSize: "13px",
                  fontFamily: "monospace",
                  opacity: t.exiting ? 0 : 1,
                  transform: t.exiting ? "translateX(20px)" : "translateX(0)",
                  transition: "opacity 0.3s ease, transform 0.3s ease",
                  cursor: "pointer",
                }}
              >
                <span style={{ color: "#666" }}>[T{t.entry.tick}]</span>{" "}
                <span style={{ color }}>{EVENT_ICONS[t.entry.event.event_type] || "\u25cf"}</span>{" "}
                <span style={{ color, fontWeight: isCatastrophe ? 700 : 600 }}>{t.entry.event.event_type}</span>
                {t.entry.event.detail && (
                  <span style={{ color: "#999" }}>: {t.entry.event.detail.slice(0, 40)}</span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Header + expand toggle */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", marginBottom: 6 }}
      >
        <h3 style={{ fontSize: "15px", fontWeight: 700, color: "#e94560", letterSpacing: "1px", textTransform: "uppercase" }}>
          Events
        </h3>
        <span style={{ fontSize: "13px", color: "#777" }}>({eventLog.length})</span>
        <span style={{ fontSize: "12px", color: "#666", marginLeft: "auto" }}>
          {expanded ? "\u25b2" : "\u25bc"}
        </span>
      </div>

      {/* Full log */}
      {expanded && (
        <div style={{ maxHeight: "240px", overflowY: "auto" }}>
          {eventLog.slice(-50).reverse().map((entry, i) => {
            const color = EVENT_COLORS[entry.event.event_type] ?? "#999";
            return (
              <div
                key={`${entry.tick}-${entry.event.event_type}-${i}`}
                onClick={() => onSeekToTick(entry.tick)}
                style={{
                  fontSize: "13px",
                  fontFamily: "monospace",
                  marginBottom: 3,
                  lineHeight: 1.5,
                  cursor: "pointer",
                  padding: "2px 4px",
                  borderRadius: "3px",
                  transition: "background 0.15s ease",
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.06)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
              >
                <span style={{ color: "#666" }}>[T{entry.tick}]</span>{" "}
                <span style={{ color: "#888" }}>{entry.event.target}</span>{" "}
                <span style={{ color, fontWeight: 600 }}>{entry.event.event_type}</span>
                {entry.event.detail && (
                  <span style={{ color: "#999" }}>: {entry.event.detail}</span>
                )}
              </div>
            );
          })}
          {eventLog.length === 0 && (
            <div style={{ color: "#666", fontSize: "13px" }}>No events yet.</div>
          )}
        </div>
      )}
    </div>
  );
}
