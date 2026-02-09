import { useEffect, useRef, useCallback } from "react";
import type { AgentSnapshot, SpeciesInfo } from "../types/simulation";
import { useInterpolation } from "../hooks/useInterpolation";
import { renderAgents, renderGrid, renderHUD } from "../utils/renderer";

interface Props {
  agents: AgentSnapshot[];
  species: SpeciesInfo[];
  worldSize: number;
  tick: number;
  epoch: number;
  interpolateDuration: number;
  isPlaying: boolean;
}

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

  // Update interpolation snapshot when new agents arrive
  useEffect(() => {
    if (agents.length > 0) {
      updateSnapshot(agents, interpolateDuration);
    }
  }, [agents, interpolateDuration, updateSnapshot]);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear
    ctx.fillStyle = "#16213e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid
    renderGrid(ctx, worldSize);

    // Get interpolated agent positions
    const interpolated = getInterpolated();

    // Render agents
    renderAgents(ctx, interpolated, species, worldSize);

    // HUD
    renderHUD(ctx, tick, epoch, interpolated.length, species.length);

    // Continue animation loop
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
      renderAgents(ctx, agents, species, worldSize);
      renderHUD(ctx, tick, epoch, agents.length, species.length);
    }
  }, [agents, species, worldSize, tick, epoch, isPlaying]);

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={800}
      style={{
        border: "2px solid #333",
        borderRadius: "4px",
        display: "block",
      }}
    />
  );
}
