import { useEffect, useRef, useState } from "react";

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

interface Props {
  epoch: number;
  tick: number;
  agentCount: number;
  speciesCount: number;
}

export function EpochIndicator({ epoch, tick, agentCount, speciesCount }: Props) {
  const [animate, setAnimate] = useState(false);
  const prevEpochRef = useRef(epoch);

  useEffect(() => {
    if (epoch !== prevEpochRef.current) {
      prevEpochRef.current = epoch;
      setAnimate(true);
      const t = setTimeout(() => setAnimate(false), 1200);
      return () => clearTimeout(t);
    }
  }, [epoch]);

  const color = EPOCH_COLORS[epoch] || "#aaa";
  const name = EPOCH_NAMES[epoch] || `Epoch ${epoch}`;

  return (
    <div style={{
      background: "rgba(10, 10, 20, 0.7)",
      backdropFilter: "blur(12px)",
      WebkitBackdropFilter: "blur(12px)",
      border: "1px solid rgba(255, 255, 255, 0.08)",
      borderRadius: "8px",
      padding: "12px 18px",
    }}>
      <div style={{
        fontSize: animate ? "28px" : "22px",
        fontWeight: 700,
        color,
        textShadow: `0 0 20px ${color}66, 0 0 40px ${color}33`,
        transition: "font-size 0.4s ease, text-shadow 0.4s ease",
        letterSpacing: "1px",
        lineHeight: 1.1,
      }}>
        {name}
      </div>
      <div style={{ display: "flex", gap: 16, marginTop: 6 }}>
        <Stat label="Tick" value={tick} />
        <Stat label="Agents" value={agentCount} />
        <Stat label="Species" value={speciesCount} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div style={{ fontSize: "12px", letterSpacing: "0.5px" }}>
      <span style={{ color: "#777", textTransform: "uppercase", fontSize: "10px" }}>{label}</span>
      <br />
      <span style={{ color: "#ccc", fontWeight: 600, fontFamily: "monospace" }}>
        {value.toLocaleString()}
      </span>
    </div>
  );
}
