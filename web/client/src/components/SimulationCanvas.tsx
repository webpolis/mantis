import { useEffect, useRef, useCallback, useState } from "react";
import type { AgentSnapshot, SpeciesInfo } from "../types/simulation";
import { useInterpolation } from "../hooks/useInterpolation";
import { renderAgents, renderGrid, renderHUD, findAgentAt } from "../utils/renderer";

interface Props {
  agents: AgentSnapshot[];
  species: SpeciesInfo[];
  worldSize: number;
  tick: number;
  epoch: number;
  interpolateDuration: number;
  isPlaying: boolean;
}

const CANVAS_SIZE = 800;

export function SimulationCanvas({
  agents,
  species,
  worldSize,
  tick,
  epoch,
  interpolateDuration,
  isPlaying,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const { updateSnapshot, getInterpolated } = useInterpolation();
  const [hovered, setHovered] = useState<AgentSnapshot | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const hoveredRef = useRef<AgentSnapshot | null>(null);

  // Keep ref in sync for use in render loop
  hoveredRef.current = hovered;

  // Update interpolation snapshot when new agents arrive
  useEffect(() => {
    if (agents.length > 0) {
      updateSnapshot(agents, interpolateDuration);
    }
  }, [agents, interpolateDuration, updateSnapshot]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top) * scaleY;

      const currentAgents = isPlaying ? getInterpolated() : agents;
      const agent = findAgentAt(cx, cy, currentAgents, worldSize, CANVAS_SIZE);
      setHovered(agent);
      setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    },
    [agents, worldSize, isPlaying, getInterpolated]
  );

  const handleMouseLeave = useCallback(() => {
    setHovered(null);
  }, []);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#16213e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    renderGrid(ctx, worldSize);

    const interpolated = getInterpolated();
    renderAgents(ctx, interpolated, species, worldSize, hoveredRef.current?.aid);

    renderHUD(ctx, tick, epoch, interpolated.length, species.length);

    if (isPlaying) {
      rafRef.current = requestAnimationFrame(render);
    }
  }, [getInterpolated, species, worldSize, tick, epoch, isPlaying]);

  useEffect(() => {
    if (isPlaying) {
      rafRef.current = requestAnimationFrame(render);
    }
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [isPlaying, render]);

  // Render once even when not playing (static frame)
  useEffect(() => {
    if (!isPlaying) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.fillStyle = "#16213e";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      renderGrid(ctx, worldSize);
      renderAgents(ctx, agents, species, worldSize, hoveredRef.current?.aid);
      renderHUD(ctx, tick, epoch, agents.length, species.length);
    }
  }, [agents, species, worldSize, tick, epoch, isPlaying]);

  const speciesMap = new Map(species.map((s) => [s.sid, s]));
  const hoveredSpecies = hovered ? speciesMap.get(hovered.species_sid) : null;

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{
          border: "2px solid #333",
          borderRadius: "4px",
          display: "block",
        }}
      />
      {hovered && (
        <div
          style={{
            position: "absolute",
            left: tooltipPos.x + 14,
            top: tooltipPos.y - 10,
            background: "rgba(0, 0, 0, 0.85)",
            color: "#eee",
            padding: "8px 10px",
            borderRadius: "4px",
            fontSize: "12px",
            fontFamily: "monospace",
            lineHeight: "1.5",
            pointerEvents: "none",
            whiteSpace: "nowrap",
            border: "1px solid #555",
            zIndex: 10,
          }}
        >
          <div style={{ fontWeight: "bold", color: "#fff", marginBottom: "2px" }}>
            Agent #{hovered.aid}
          </div>
          <div>
            <span style={{ color: "#999" }}>Species:</span>{" "}
            S{hovered.species_sid}
            {hoveredSpecies && (
              <span style={{ color: "#aaa" }}> ({hoveredSpecies.plan})</span>
            )}
          </div>
          <div>
            <span style={{ color: "#999" }}>Energy:</span>{" "}
            <span style={{ color: hovered.energy < 20 ? "#ff4" : "#eee" }}>
              {Math.round(hovered.energy)}
            </span>
          </div>
          <div>
            <span style={{ color: "#999" }}>Age:</span> {hovered.age}
          </div>
          <div>
            <span style={{ color: "#999" }}>State:</span>{" "}
            <span style={{ color: "#8cf" }}>{hovered.state}</span>
          </div>
          {hovered.target_aid != null && (
            <div>
              <span style={{ color: "#999" }}>Target:</span> #{hovered.target_aid}
            </div>
          )}
          <div>
            <span style={{ color: "#999" }}>Pos:</span>{" "}
            ({Math.round(hovered.x)}, {Math.round(hovered.y)})
          </div>
          {hovered.count > 1 && (
            <div>
              <span style={{ color: "#999" }}>Count:</span> {hovered.count}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
