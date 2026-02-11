import { useEffect, useRef, useCallback, useState, useMemo } from "react";
import type { AgentSnapshot, SpeciesInfo, BiomeData, VegetationPatchData, SimulationEvent } from "../types/simulation";
import { useInterpolation } from "../hooks/useInterpolation";
import { PixiApp } from "../pixi/PixiApp";
import { biomeCenters } from "../pixi/biomes";

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

const HIT_SLOP = 8; // world-space pixels

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
  const containerRef = useRef<HTMLDivElement>(null);
  const pixiRef = useRef<PixiApp | null>(null);
  const { updateSnapshot, getInterpolated } = useInterpolation();
  const [hovered, setHovered] = useState<HoveredEntity | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const hoveredRef = useRef<HoveredEntity | null>(null);
  const prevEpochRef = useRef(epoch);
  const biomeCentersMap = useMemo(() => biomeCenters(biomes, worldSize), [biomes, worldSize]);

  // Track dragging
  const dragRef = useRef<{ active: boolean; lastX: number; lastY: number }>({
    active: false, lastX: 0, lastY: 0,
  });

  hoveredRef.current = hovered;

  // Init PixiJS on mount
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const pixi = new PixiApp();
    pixiRef.current = pixi;

    // Register wheel synchronously so cleanup never misses it
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      pixi.camera?.zoomAt(e.clientX, e.clientY, e.deltaY);
    };
    container.addEventListener("wheel", onWheel, { passive: false });

    pixi.init(container, worldSize);

    return () => {
      container.removeEventListener("wheel", onWheel);
      pixi.destroy();
      pixiRef.current = null;
    };
  }, [worldSize]);

  // Feed interpolation
  useEffect(() => {
    updateSnapshot(agents, interpolateDuration);
  }, [agents, interpolateDuration, updateSnapshot]);

  // Update biomes
  useEffect(() => {
    pixiRef.current?.updateBiomes(biomes);
  }, [biomes]);

  // Update events + epoch transitions
  useEffect(() => {
    pixiRef.current?.updateEvents(events);
  }, [events]);

  useEffect(() => {
    if (epoch !== prevEpochRef.current) {
      prevEpochRef.current = epoch;
      pixiRef.current?.particleSystem?.emitEpochTransition(worldSize / 2, worldSize / 2);
    }
  }, [epoch, worldSize]);

  // Update agents on every frame
  useEffect(() => {
    const pixi = pixiRef.current;
    if (!pixi) return;

    let raf = 0;
    const loop = () => {
      const currentAgents = isPlaying ? getInterpolated() : agents;
      const hovUid = hoveredRef.current?.type === "agent" ? hoveredRef.current.agent.uid : undefined;
      pixi.updateAgents(currentAgents, species, hovUid);
      if (isPlaying) raf = requestAnimationFrame(loop);
    };

    if (isPlaying) {
      raf = requestAnimationFrame(loop);
    } else {
      // Static update
      const hovUid = hoveredRef.current?.type === "agent" ? hoveredRef.current.agent.uid : undefined;
      pixi.updateAgents(agents, species, hovUid);
    }

    return () => { if (raf) cancelAnimationFrame(raf); };
  }, [agents, species, isPlaying, getInterpolated]);

  // Mouse handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const pixi = pixiRef.current;
    if (!pixi) return;

    // Minimap click — pan camera to that world position
    if (pixi.hitTestMinimap(e.clientX, e.clientY)) {
      pixi.isMinimapDragging = true;
      pixi.minimapPanTo(e.clientX, e.clientY);
      return;
    }

    if (!pixi.camera.isZoomed) return;
    dragRef.current = { active: true, lastX: e.clientX, lastY: e.clientY };
  }, []);

  const handleMouseUp = useCallback(() => {
    dragRef.current.active = false;
    if (pixiRef.current) pixiRef.current.isMinimapDragging = false;
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const pixi = pixiRef.current;
    if (!pixi) return;

    // Minimap drag
    if (pixi.isMinimapDragging) {
      pixi.minimapPanTo(e.clientX, e.clientY);
      return;
    }

    // Pan
    if (dragRef.current.active) {
      pixi.camera.panBy(
        e.clientX - dragRef.current.lastX,
        e.clientY - dragRef.current.lastY,
      );
      dragRef.current.lastX = e.clientX;
      dragRef.current.lastY = e.clientY;
    }

    // Hit detection in world space
    const world = pixi.camera.screenToWorld(e.clientX, e.clientY);
    const currentAgents = isPlaying ? getInterpolated() : agents;
    const speciesMap = new Map(species.map((s) => [s.sid, s]));

    // Agent hit
    let closestAgent: AgentSnapshot | null = null;
    let closestDist = Infinity;
    for (const a of currentAgents) {
      if (a.dead) continue;
      const dx = world.x - a.x;
      const dy = world.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      const r = Math.max(4, Math.sqrt(a.count || 1) * 3) + HIT_SLOP / pixi.camera.state.zoom;
      if (d < r && d < closestDist) {
        closestAgent = a;
        closestDist = d;
      }
    }

    if (closestAgent) {
      setHovered({ type: "agent", agent: closestAgent, species: speciesMap.get(closestAgent.species_sid) });
    } else {
      // Vegetation hit
      let closestVeg: { patch: VegetationPatchData; biome: BiomeData } | null = null;
      let vegDist = Infinity;
      for (const biome of biomes) {
        for (const patch of biome.patches) {
          if (patch.density < 0.01) continue;
          const dx = world.x - patch.x;
          const dy = world.y - patch.y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d <= patch.radius * 0.6 && d < vegDist) {
            closestVeg = { patch, biome };
            vegDist = d;
          }
        }
      }

      if (closestVeg) {
        setHovered({ type: "vegetation", patch: closestVeg.patch, biome: closestVeg.biome });
      } else if (biomes.length > 0 && biomeCentersMap.size > 0) {
        // Biome hit via nearest Voronoi center
        let bestBiome: BiomeData | null = null;
        let bestDist = Infinity;
        for (const b of biomes) {
          const center = biomeCentersMap.get(b.lid);
          if (!center) continue;
          const dx = world.x - center[0];
          const dy = world.y - center[1];
          const d = dx * dx + dy * dy;
          if (d < bestDist) {
            bestDist = d;
            bestBiome = b;
          }
        }
        setHovered(bestBiome ? { type: "biome", biome: bestBiome } : null);
      } else {
        setHovered(null);
      }
    }

    setTooltipPos({ x: e.clientX, y: e.clientY });
  }, [agents, species, biomes, biomeCentersMap, isPlaying, getInterpolated]);

  const handleMouseLeave = useCallback(() => {
    setHovered(null);
    dragRef.current.active = false;
    if (pixiRef.current) pixiRef.current.isMinimapDragging = false;
  }, []);

  const handleDoubleClick = useCallback(() => {
    pixiRef.current?.camera.reset();
  }, []);

  const isZoomed = pixiRef.current?.camera.isZoomed ?? false;

  return (
    <div
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onDoubleClick={handleDoubleClick}
      style={{
        position: "absolute",
        inset: 0,
        cursor: isZoomed
          ? dragRef.current.active ? "grabbing" : "grab"
          : "default",
      }}
    >
      {/* Catastrophe banner overlay */}
      <CatastropheBanner events={events} />

      {/* Tooltip */}
      {hovered && !dragRef.current.active && (
        <div
          style={{
            position: "fixed",
            left: tooltipPos.x + 14,
            top: tooltipPos.y - 10,
            background: "rgba(8, 8, 16, 0.92)",
            backdropFilter: "blur(10px)",
            color: "#dde",
            padding: "9px 12px",
            borderRadius: "6px",
            fontSize: "13px",
            fontFamily: "'Rajdhani', monospace",
            lineHeight: "1.5",
            pointerEvents: "none",
            whiteSpace: "nowrap",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            zIndex: 20,
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
      <div style={{ fontWeight: 700, color: "#fff", marginBottom: 2, fontSize: "14px" }}>
        Agent #{agent.aid}
      </div>
      <div>
        <span style={{ color: "#666" }}>Species:</span>{" "}
        S{agent.species_sid}
        {sp && <span style={{ color: "#888" }}> ({sp.plan})</span>}
      </div>
      <div>
        <span style={{ color: "#666" }}>Energy:</span>{" "}
        <span style={{ color: agent.energy < 20 ? "#ff4" : "#ccc" }}>
          {Math.round(agent.energy)}
        </span>
      </div>
      <div>
        <span style={{ color: "#666" }}>Age:</span> {agent.age}
      </div>
      <div>
        <span style={{ color: "#666" }}>State:</span>{" "}
        <span style={{ color: "#8cf" }}>{agent.state}</span>
      </div>
      {agent.target_aid != null && (
        <div>
          <span style={{ color: "#666" }}>Target:</span> #{agent.target_aid}
        </div>
      )}
      <div>
        <span style={{ color: "#666" }}>Pos:</span>{" "}
        ({Math.round(agent.x)}, {Math.round(agent.y)})
      </div>
      {agent.count > 1 && (
        <div>
          <span style={{ color: "#666" }}>Count:</span> {agent.count}
        </div>
      )}
    </>
  );
}

function VegetationTooltip({ entity }: { entity: Extract<HoveredEntity, { type: "vegetation" }> }) {
  const { patch, biome } = entity;
  return (
    <>
      <div style={{ fontWeight: 700, color: "#6c6", marginBottom: 2, fontSize: "14px" }}>
        Vegetation Patch
      </div>
      <div>
        <span style={{ color: "#666" }}>Biome:</span>{" "}
        {biome.name} (L{biome.lid})
      </div>
      <div>
        <span style={{ color: "#666" }}>Density:</span>{" "}
        {(patch.density * 100).toFixed(0)}%
      </div>
      <div>
        <span style={{ color: "#666" }}>Radius:</span> {Math.round(patch.radius)}
      </div>
    </>
  );
}

function BiomeTooltip({ entity }: { entity: Extract<HoveredEntity, { type: "biome" }> }) {
  const { biome } = entity;
  return (
    <>
      <div style={{ fontWeight: 700, color: "#8cf", marginBottom: 2, fontSize: "14px" }}>
        {biome.name} <span style={{ color: "#666", fontWeight: 400 }}>L{biome.lid}</span>
      </div>
      <div>
        <span style={{ color: "#666" }}>Vegetation:</span>{" "}
        {(biome.vegetation * 100).toFixed(0)}%
      </div>
      <div>
        <span style={{ color: "#666" }}>Detritus:</span>{" "}
        {Math.round(biome.detritus)}
      </div>
      <div>
        <span style={{ color: "#666" }}>Nitrogen:</span>{" "}
        {(biome.nitrogen * 100).toFixed(0)}%
      </div>
      <div>
        <span style={{ color: "#666" }}>Phosphorus:</span>{" "}
        {(biome.phosphorus * 100).toFixed(0)}%
      </div>
    </>
  );
}

const CATASTROPHE_COLORS: Record<string, string> = {
  volcanic_winter: "rgba(255, 100, 30, 0.2)",
  meteor_impact: "rgba(255, 40, 40, 0.25)",
  ice_age: "rgba(60, 120, 255, 0.2)",
  tsunami: "rgba(40, 140, 255, 0.2)",
  drought: "rgba(200, 150, 50, 0.15)",
  plague: "rgba(150, 200, 50, 0.15)",
};

function CatastropheBanner({ events }: { events: SimulationEvent[] }) {
  const active = events.find((e) => e.event_type === "catastrophe");
  if (!active) return null;

  const kind = active.detail.split("|")[0];
  const bg = CATASTROPHE_COLORS[kind] ?? "rgba(255, 100, 30, 0.15)";

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        padding: "8px 20px",
        background: bg,
        backdropFilter: "blur(4px)",
        color: "#fff",
        fontSize: "16px",
        fontWeight: 700,
        textAlign: "center",
        pointerEvents: "none",
        textShadow: "0 1px 4px rgba(0,0,0,0.6)",
        letterSpacing: "2px",
        textTransform: "uppercase",
        zIndex: 15,
      }}
    >
      {kind.replace(/_/g, " ")}
    </div>
  );
}
