import { useEffect, useRef, useCallback, useState } from "react";
import type { AgentSnapshot, SpeciesInfo, BiomeData, VegetationPatchData, SimulationEvent } from "../types/simulation";
import { useInterpolation } from "../hooks/useInterpolation";
import {
  renderAgents,
  renderGrid,
  renderHUD,
  findAgentAt,
  findVegetationAt,
  findBiomeAt,
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
  events: SimulationEvent[];
}

type HoveredEntity =
  | { type: "agent"; agent: AgentSnapshot; species?: SpeciesInfo }
  | { type: "vegetation"; patch: VegetationPatchData; biome: BiomeData }
  | { type: "biome"; biome: BiomeData };

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
  events,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const { updateSnapshot, getInterpolated } = useInterpolation();
  const [hovered, setHovered] = useState<HoveredEntity | null>(null);

  // Cached biome texture
  const biomeTextureRef = useRef<HTMLCanvasElement | null>(null);
  const biomeKeyRef = useRef<string>("");
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const hoveredRef = useRef<HoveredEntity | null>(null);

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

  // Rebuild biome texture when biome identity or vegetation levels change
  useEffect(() => {
    const key = biomes.map((b) => `${b.lid}:${b.name}:${b.vegetation.toFixed(2)}`).join(",");
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
      const speciesMap = new Map(species.map((s) => [s.sid, s]));

      // Priority: agent > vegetation > biome
      const agent = findAgentAt(bx, by, currentAgents, worldSize, CANVAS_SIZE);
      if (agent) {
        setHovered({ type: "agent", agent, species: speciesMap.get(agent.species_sid) });
      } else {
        const veg = findVegetationAt(bx, by, biomes, worldSize, CANVAS_SIZE);
        if (veg) {
          setHovered({ type: "vegetation", patch: veg.patch, biome: veg.biome });
        } else {
          const biome = findBiomeAt(bx, by, biomes, CANVAS_SIZE);
          if (biome) {
            setHovered({ type: "biome", biome });
          } else {
            setHovered(null);
          }
        }
      }
      setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    },
    [agents, species, biomes, worldSize, isPlaying, getInterpolated]
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
      const hovAid = hoveredRef.current?.type === "agent" ? hoveredRef.current.agent.aid : undefined;
      renderAgents(ctx, interpolated, species, worldSize, hovAid);
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
        const hovAid = hoveredRef.current?.type === "agent" ? hoveredRef.current.agent.aid : undefined;
        renderAgents(ctx, agents, species, worldSize, hovAid);
      });

      renderHUD(ctx, tick, epoch, agents.length, species.length, cam.zoom);
    }
  }, [agents, species, worldSize, tick, epoch, isPlaying, camSnapshot, biomes]);

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
      <CatastropheBanner events={events} />
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
          {hovered.type === "agent" && <AgentTooltip entity={hovered} />}
          {hovered.type === "vegetation" && <VegetationTooltip entity={hovered} />}
          {hovered.type === "biome" && <BiomeTooltip entity={hovered} />}
        </div>
      )}
    </div>
  );
}

function AgentTooltip({ entity }: { entity: Extract<HoveredEntity, { type: "agent" }> }) {
  const { agent, species: sp } = entity;
  return (
    <>
      <div style={{ fontWeight: "bold", color: "#fff", marginBottom: "2px" }}>
        Agent #{agent.aid}
      </div>
      <div>
        <span style={{ color: "#999" }}>Species:</span>{" "}
        S{agent.species_sid}
        {sp && <span style={{ color: "#aaa" }}> ({sp.plan})</span>}
      </div>
      <div>
        <span style={{ color: "#999" }}>Energy:</span>{" "}
        <span style={{ color: agent.energy < 20 ? "#ff4" : "#eee" }}>
          {Math.round(agent.energy)}
        </span>
      </div>
      <div>
        <span style={{ color: "#999" }}>Age:</span> {agent.age}
      </div>
      <div>
        <span style={{ color: "#999" }}>State:</span>{" "}
        <span style={{ color: "#8cf" }}>{agent.state}</span>
      </div>
      {agent.target_aid != null && (
        <div>
          <span style={{ color: "#999" }}>Target:</span> #{agent.target_aid}
        </div>
      )}
      <div>
        <span style={{ color: "#999" }}>Pos:</span>{" "}
        ({Math.round(agent.x)}, {Math.round(agent.y)})
      </div>
      {agent.count > 1 && (
        <div>
          <span style={{ color: "#999" }}>Count:</span> {agent.count}
        </div>
      )}
    </>
  );
}

function VegetationTooltip({ entity }: { entity: Extract<HoveredEntity, { type: "vegetation" }> }) {
  const { patch, biome } = entity;
  return (
    <>
      <div style={{ fontWeight: "bold", color: "#6c6", marginBottom: "2px" }}>
        Vegetation Patch
      </div>
      <div>
        <span style={{ color: "#999" }}>Biome:</span>{" "}
        {biome.name} (L{biome.lid})
      </div>
      <div>
        <span style={{ color: "#999" }}>Density:</span>{" "}
        {(patch.density * 100).toFixed(0)}%
      </div>
      <div>
        <span style={{ color: "#999" }}>Radius:</span> {Math.round(patch.radius)}
      </div>
      <div>
        <span style={{ color: "#999" }}>Pos:</span>{" "}
        ({Math.round(patch.x)}, {Math.round(patch.y)})
      </div>
    </>
  );
}

function BiomeTooltip({ entity }: { entity: Extract<HoveredEntity, { type: "biome" }> }) {
  const { biome } = entity;
  return (
    <>
      <div style={{ fontWeight: "bold", color: "#8cf", marginBottom: "2px" }}>
        {biome.name}
      </div>
      <div>
        <span style={{ color: "#999" }}>ID:</span> L{biome.lid}
      </div>
      <div>
        <span style={{ color: "#999" }}>Vegetation:</span>{" "}
        {(biome.vegetation * 100).toFixed(0)}%
      </div>
      <div>
        <span style={{ color: "#999" }}>Detritus:</span>{" "}
        {Math.round(biome.detritus)}
      </div>
      <div>
        <span style={{ color: "#999" }}>Nitrogen:</span>{" "}
        <span style={{ color: biome.nitrogen < 0.2 ? "#ff4" : "#eee" }}>
          {((biome.nitrogen ?? 0.5) * 100).toFixed(0)}%
        </span>
      </div>
      <div>
        <span style={{ color: "#999" }}>Phosphorus:</span>{" "}
        <span style={{ color: biome.phosphorus < 0.1 ? "#ff4" : "#eee" }}>
          {((biome.phosphorus ?? 0.3) * 100).toFixed(0)}%
        </span>
      </div>
    </>
  );
}

const CATASTROPHE_COLORS: Record<string, string> = {
  volcanic_winter: "rgba(255, 100, 30, 0.25)",
  meteor_impact: "rgba(255, 40, 40, 0.25)",
  ice_age: "rgba(60, 120, 255, 0.25)",
};

function CatastropheBanner({ events }: { events: SimulationEvent[] }) {
  const active = events.find((e) => e.event_type === "catastrophe");
  if (!active) return null;

  const kind = active.detail.split("|")[0];
  const bg = CATASTROPHE_COLORS[kind] ?? "rgba(255, 100, 30, 0.2)";

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        padding: "6px 12px",
        background: bg,
        color: "#fff",
        fontFamily: "monospace",
        fontSize: "13px",
        fontWeight: "bold",
        textAlign: "center",
        pointerEvents: "none",
        borderRadius: "4px 4px 0 0",
        textShadow: "0 1px 3px rgba(0,0,0,0.6)",
      }}
    >
      {kind.replace(/_/g, " ").toUpperCase()}
    </div>
  );
}
