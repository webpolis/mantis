import { useEffect, useRef, useCallback, useState } from "react";
import type { AgentSnapshot, SpeciesInfo, BiomeData } from "../types/simulation";
import { useInterpolation } from "../hooks/useInterpolation";
import {
  renderAgents,
  renderGrid,
  renderHUD,
  findAgentAt,
  withCamera,
  screenToBase,
  DEFAULT_CAMERA,
  buildBiomeTexture,
  renderBiomeBackground,
  renderVegetation,
} from "../utils/renderer";
import type { Camera } from "../utils/renderer";

interface Props {
  agents: AgentSnapshot[];
  species: SpeciesInfo[];
  worldSize: number;
  tick: number;
  epoch: number;
  interpolateDuration: number;
  isPlaying: boolean;
  biomes: BiomeData[];
}

const CANVAS_SIZE = 800;
const MIN_ZOOM = 1;
const MAX_ZOOM = 20;
const ZOOM_SPEED = 0.001;

export function SimulationCanvas({
  agents,
  species,
  worldSize,
  tick,
  epoch,
  interpolateDuration,
  isPlaying,
  biomes,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const { updateSnapshot, getInterpolated } = useInterpolation();
  const [hovered, setHovered] = useState<AgentSnapshot | null>(null);

  // Cached biome texture
  const biomeTextureRef = useRef<HTMLCanvasElement | null>(null);
  const biomeKeyRef = useRef<string>("");
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const hoveredRef = useRef<AgentSnapshot | null>(null);

  // Camera state as ref to avoid re-creating render callback on every zoom/pan
  const camRef = useRef<Camera>({ ...DEFAULT_CAMERA });
  const [camSnapshot, setCamSnapshot] = useState<Camera>({ ...DEFAULT_CAMERA });

  // Pan dragging state
  const dragRef = useRef<{ active: boolean; lastX: number; lastY: number }>({
    active: false,
    lastX: 0,
    lastY: 0,
  });

  hoveredRef.current = hovered;

  useEffect(() => {
    if (agents.length > 0) {
      updateSnapshot(agents, interpolateDuration);
    }
  }, [agents, interpolateDuration, updateSnapshot]);

  // Rebuild biome texture when biome set changes
  useEffect(() => {
    const key = biomes.map((b) => `${b.lid}:${b.name}`).join(",");
    if (key !== biomeKeyRef.current) {
      biomeKeyRef.current = key;
      biomeTextureRef.current = buildBiomeTexture(biomes, CANVAS_SIZE);
    }
  }, [biomes]);

  // --- Zoom (wheel) ---
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const cam = camRef.current;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;

      const oldZoom = cam.zoom;
      const delta = -e.deltaY * ZOOM_SPEED * oldZoom;
      const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, oldZoom + delta));

      // Zoom toward cursor: keep the world point under cursor fixed
      cam.offsetX = mx - (mx - cam.offsetX) * (newZoom / oldZoom);
      cam.offsetY = my - (my - cam.offsetY) * (newZoom / oldZoom);
      cam.zoom = newZoom;

      // Clamp at zoom=1 to reset offset
      if (newZoom === 1) {
        cam.offsetX = 0;
        cam.offsetY = 0;
      }

      setCamSnapshot({ ...cam });
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  // --- Pan (drag) ---
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (camRef.current.zoom <= 1) return;
      dragRef.current = { active: true, lastX: e.clientX, lastY: e.clientY };
    },
    []
  );

  const handleMouseUp = useCallback(() => {
    dragRef.current.active = false;
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();

      // Handle panning
      if (dragRef.current.active) {
        const cam = camRef.current;
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        cam.offsetX += (e.clientX - dragRef.current.lastX) * scaleX;
        cam.offsetY += (e.clientY - dragRef.current.lastY) * scaleY;
        dragRef.current.lastX = e.clientX;
        dragRef.current.lastY = e.clientY;
        setCamSnapshot({ ...cam });
      }

      // Hit detection
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top) * scaleY;

      // Convert screen coords to base (pre-camera) coords
      const [bx, by] = screenToBase(cx, cy, camRef.current);

      const currentAgents = isPlaying ? getInterpolated() : agents;
      const agent = findAgentAt(bx, by, currentAgents, worldSize, CANVAS_SIZE);
      setHovered(agent);
      setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    },
    [agents, worldSize, isPlaying, getInterpolated]
  );

  const handleMouseLeave = useCallback(() => {
    setHovered(null);
    dragRef.current.active = false;
  }, []);

  const handleDoubleClick = useCallback(() => {
    camRef.current = { ...DEFAULT_CAMERA };
    setCamSnapshot({ ...DEFAULT_CAMERA });
  }, []);

  // --- Render loop ---
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cam = camRef.current;
    const interpolated = getInterpolated();

    withCamera(ctx, cam, () => {
      if (biomeTextureRef.current) {
        renderBiomeBackground(ctx, biomeTextureRef.current);
      }
      renderGrid(ctx, worldSize);
      renderVegetation(ctx, biomes, worldSize);
      renderAgents(ctx, interpolated, species, worldSize, hoveredRef.current?.aid);
    });

    renderHUD(ctx, tick, epoch, interpolated.length, species.length, cam.zoom);

    if (isPlaying) {
      rafRef.current = requestAnimationFrame(render);
    }
  }, [getInterpolated, species, worldSize, tick, epoch, isPlaying, biomes]);

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

      ctx.fillStyle = "#0d1117";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const cam = camRef.current;

      withCamera(ctx, cam, () => {
        if (biomeTextureRef.current) {
          renderBiomeBackground(ctx, biomeTextureRef.current);
        }
        renderGrid(ctx, worldSize);
        renderVegetation(ctx, biomes, worldSize);
        renderAgents(ctx, agents, species, worldSize, hoveredRef.current?.aid);
      });

      renderHUD(ctx, tick, epoch, agents.length, species.length, cam.zoom);
    }
  }, [agents, species, worldSize, tick, epoch, isPlaying, camSnapshot, biomes]);

  const speciesMap = new Map(species.map((s) => [s.sid, s]));
  const hoveredSpecies = hovered ? speciesMap.get(hovered.species_sid) : null;
  const isZoomed = camSnapshot.zoom > 1;

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onDoubleClick={handleDoubleClick}
        style={{
          border: "2px solid #333",
          borderRadius: "4px",
          display: "block",
          cursor: isZoomed
            ? dragRef.current.active
              ? "grabbing"
              : "grab"
            : "default",
        }}
      />
      {hovered && !dragRef.current.active && (
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
