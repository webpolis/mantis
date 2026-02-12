import { useEffect, useMemo, useRef, useState } from "react";
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
  // Mutations
  Mpoint: "#77bbdd",
  Mdrift: "#8899aa",
  Mleap: "#ff9944",
  "M-": "#ee6677",
  "M+": "#66ddaa",
  Mfuse: "#dd77ee",
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
  // Mutations
  Mpoint: "\u2022",
  Mdrift: "\u223c",
  Mleap: "\u26a1",
  "M-": "\u2212",
  "M+": "\u002b",
  Mfuse: "\u2726",
};

// Mutation events are high-frequency — group them in the expanded log
const MUTATION_TYPES = new Set(["Mpoint", "Mdrift", "Mleap", "M-", "M+", "Mfuse"]);
const isMutation = (t: string) => MUTATION_TYPES.has(t);

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

    // Mutations are frequent — only toast non-mutation events + Mleap/M+/Mfuse
    const toastWorthy = newEntries.filter((e) =>
      !isMutation(e.event.event_type) ||
      e.event.event_type === "Mleap" ||
      e.event.event_type === "M+" ||
      e.event.event_type === "Mfuse"
    );
    const newToasts = toastWorthy.map((entry) => ({
      id: ++toastId,
      entry,
      exiting: false,
    }));

    setToasts((prev) => [...prev, ...newToasts].slice(-6));

    // Auto dismiss (collect timers for cleanup)
    const timers: number[] = [];
    for (const t of newToasts) {
      const evtType = t.entry.event.event_type;
      const duration = evtType === "catastrophe" ? 6000
        : evtType === "evo_trap" ? 5000
        : isMutation(evtType) ? 2500
        : 4000;
      const outer = window.setTimeout(() => {
        setToasts((prev) => prev.map((p) => p.id === t.id ? { ...p, exiting: true } : p));
        const inner = window.setTimeout(() => {
          setToasts((prev) => prev.filter((p) => p.id !== t.id));
        }, 300);
        timers.push(inner);
      }, duration);
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
            const evtType = t.entry.event.event_type;
            const isCatastrophe = evtType === "catastrophe";
            const isEvoTrap = evtType === "evo_trap";
            const isMut = isMutation(evtType);
            const isHighlight = isCatastrophe || isEvoTrap;
            const color = EVENT_COLORS[evtType] ?? "#999";
            return (
              <div
                key={t.id}
                onClick={() => onSeekToTick(t.entry.tick)}
                style={{
                  padding: isMut ? "4px 8px" : "7px 10px",
                  background: isCatastrophe ? "rgba(255, 60, 30, 0.18)"
                    : isEvoTrap ? "rgba(160, 80, 255, 0.14)"
                    : isMut ? "rgba(255, 255, 255, 0.03)"
                    : "rgba(255, 255, 255, 0.06)",
                  borderRadius: "6px",
                  border: isHighlight ? `1px solid ${color}55` : "1px solid rgba(255,255,255,0.06)",
                  boxShadow: isHighlight ? `0 0 12px ${color}33` : "none",
                  fontSize: isMut ? "12px" : "14px",
                  fontFamily: "monospace",
                  opacity: t.exiting ? 0 : 1,
                  transform: t.exiting ? "translateX(20px)" : "translateX(0)",
                  transition: "opacity 0.3s ease, transform 0.3s ease",
                  cursor: "pointer",
                }}
              >
                <span style={{ color: "#666" }}>[T{t.entry.tick}]</span>{" "}
                <span style={{ color }}>{EVENT_ICONS[evtType] || "\u25cf"}</span>{" "}
                {isMut ? (
                  <>
                    <span style={{ color: "#888" }}>{t.entry.event.target}</span>{" "}
                    <MutationLabel type={evtType} detail={t.entry.event.detail} />
                  </>
                ) : (
                  <>
                    <span style={{ color, fontWeight: isHighlight ? 700 : 600 }}>{evtType}</span>
                    {t.entry.event.detail && (
                      <span style={{ color: "#999" }}>: {t.entry.event.detail.slice(0, 40)}</span>
                    )}
                  </>
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
        <h3 style={{ fontSize: "16px", fontWeight: 700, color: "#e94560", letterSpacing: "1px", textTransform: "uppercase" }}>
          Events
        </h3>
        <span style={{ fontSize: "14px", color: "#777" }}>({eventLog.length})</span>
        <span style={{ fontSize: "13px", color: "#666", marginLeft: "auto" }}>
          {expanded ? "\u25b2" : "\u25bc"}
        </span>
      </div>

      {/* Full log */}
      {expanded && (
        <div style={{ maxHeight: "240px", overflowY: "auto" }}>
          {/* Mutation summary bar */}
          <MutationSummary eventLog={eventLog} />
          {eventLog.slice(-50).reverse().map((entry, i) => {
            const evtType = entry.event.event_type;
            const isMut = isMutation(evtType);
            const color = EVENT_COLORS[evtType] ?? "#999";
            return (
              <div
                key={`${entry.tick}-${evtType}-${i}`}
                onClick={() => onSeekToTick(entry.tick)}
                style={{
                  fontSize: isMut ? "12px" : "14px",
                  fontFamily: "monospace",
                  marginBottom: isMut ? 1 : 3,
                  lineHeight: 1.5,
                  cursor: "pointer",
                  padding: isMut ? "1px 4px" : "2px 4px",
                  borderRadius: "3px",
                  borderLeft: isMut ? `2px solid ${color}44` : "none",
                  transition: "background 0.15s ease",
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.06)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
              >
                <span style={{ color: "#555" }}>[T{entry.tick}]</span>{" "}
                <span style={{ color: "#888" }}>{entry.event.target}</span>{" "}
                {isMut ? (
                  <MutationLabel type={evtType} detail={entry.event.detail} />
                ) : (
                  <>
                    <span style={{ color, fontWeight: 600 }}>{evtType}</span>
                    {entry.event.detail && (
                      <span style={{ color: "#999" }}>: {entry.event.detail}</span>
                    )}
                  </>
                )}
              </div>
            );
          })}
          {eventLog.length === 0 && (
            <div style={{ color: "#666", fontSize: "14px" }}>No events yet.</div>
          )}
        </div>
      )}
    </div>
  );
}

/* ----------- Sub-components ----------- */

const MUTATION_LABELS: Record<string, string> = {
  Mpoint: "point",
  Mdrift: "drift",
  Mleap: "LEAP",
  "M-": "lost",
  "M+": "gained",
  Mfuse: "fused",
};

function MutationLabel({ type, detail }: { type: string; detail: string }) {
  const color = EVENT_COLORS[type] ?? "#888";
  const label = MUTATION_LABELS[type] ?? type;
  return (
    <span>
      <span style={{
        color, fontWeight: type === "Mleap" || type === "M+" || type === "Mfuse" ? 700 : 500,
        fontSize: "inherit",
      }}>
        {label}
      </span>
      {detail && <span style={{ color: "#777" }}> {detail}</span>}
    </span>
  );
}

function MutationSummary({ eventLog }: { eventLog: EventLogEntry[] }) {
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const e of eventLog) {
      if (isMutation(e.event.event_type)) {
        c[e.event.event_type] = (c[e.event.event_type] || 0) + 1;
      }
    }
    return c;
  }, [eventLog]);

  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (total === 0) return null;

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6,
      padding: "3px 6px", marginBottom: 4,
      background: "rgba(255,255,255,0.03)", borderRadius: 4,
      fontSize: "11px", fontFamily: "monospace",
    }}>
      <span style={{ color: "#666", textTransform: "uppercase", letterSpacing: "0.5px" }}>
        Mutations
      </span>
      {/* Stacked bar */}
      <div style={{
        flex: 1, height: 4, borderRadius: 2, overflow: "hidden",
        display: "flex", background: "rgba(255,255,255,0.04)",
      }}>
        {Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([type, cnt]) => (
          <div key={type} style={{
            width: `${(cnt / total) * 100}%`,
            height: "100%",
            background: EVENT_COLORS[type] ?? "#666",
          }} />
        ))}
      </div>
      <span style={{ color: "#888" }}>{total}</span>
    </div>
  );
}
