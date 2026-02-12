import { useEffect, useRef, useCallback, useState, useMemo } from "react";
import type { AgentSnapshot, SpeciesInfo, BiomeData, VegetationPatchData, SimulationEvent } from "../types/simulation";
import { useInterpolation } from "../hooks/useInterpolation";
import { PixiApp } from "../pixi/PixiApp";
import { clusterAgents } from "../pixi/clustering";
import { biomeCenters } from "../pixi/biomes";
import { getCreatureIconDataURL, BODY_PLAN_COLORS } from "../pixi/creatures";

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

interface InspectionData {
  entities: Array<{ agent: AgentSnapshot; species?: SpeciesInfo }>;
  totalCount: number;
  screenX: number;
  screenY: number;
}

const INSPECTION_HIT_RADIUS = 30;
const MAX_INSPECTION_ENTITIES = 12;
const POPUP_WIDTH = 340;

const STATE_BADGE_COLORS: Record<string, string> = {
  hunt: "#ff3333",
  flee: "#ffaa33",
  mate: "#ff66cc",
  flock: "#44aaff",
  rest: "#777777",
  forage: "#66cc66",
};

const TRAIT_DISPLAY_ORDER = ["speed", "size", "armor", "sense", "intel", "camouflage", "cooperation", "fecundity"];

function energyColor(energy: number): string {
  if (energy > 60) return "#44cc66";
  if (energy > 20) return "#cccc44";
  return "#cc4444";
}

// Inject popup animation keyframes
if (typeof document !== "undefined") {
  const id = "inspection-popup-style";
  if (!document.getElementById(id)) {
    const style = document.createElement("style");
    style.id = id;
    style.textContent = "@keyframes inspection-appear{from{opacity:0;transform:scale(.96) translateY(6px)}to{opacity:1;transform:scale(1) translateY(0)}}";
    document.head.appendChild(style);
  }
}

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
  const [inspection, setInspection] = useState<InspectionData | null>(null);
  const hoveredRef = useRef<HoveredEntity | null>(null);
  const prevEpochRef = useRef(epoch);
  const biomeCentersMap = useMemo(() => biomeCenters(biomes, worldSize), [biomes, worldSize]);

  // Cached clustered agents from the RAF loop — reused by mouse hit-testing
  const clusteredRef = useRef<AgentSnapshot[]>([]);
  const rawAgentsRef = useRef<AgentSnapshot[]>([]);

  // Track dragging
  const dragRef = useRef<{ active: boolean; lastX: number; lastY: number; startX: number; startY: number; didDrag: boolean }>({
    active: false, lastX: 0, lastY: 0, startX: 0, startY: 0, didDrag: false,
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
      if ((e.target as Element)?.closest?.("[data-inspection-popup]")) return;
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

  // Close inspection popup on ESC
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setInspection(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Update agents on every frame
  useEffect(() => {
    const pixi = pixiRef.current;
    if (!pixi) return;

    let raf = 0;
    const loop = () => {
      const currentAgents = isPlaying ? getInterpolated() : agents;
      const zoom = pixi.camera?.state.zoom ?? 1;
      rawAgentsRef.current = currentAgents;
      const clustered = clusterAgents(currentAgents, zoom);
      clusteredRef.current = clustered;
      const hovUid = hoveredRef.current?.type === "agent" ? hoveredRef.current.agent.uid : undefined;
      pixi.updateAgents(clustered, species, hovUid, currentAgents);
      raf = requestAnimationFrame(loop);
    };

    // Always run the RAF loop so clusters update when zooming while paused
    raf = requestAnimationFrame(loop);

    return () => { if (raf) cancelAnimationFrame(raf); };
  }, [agents, species, isPlaying, getInterpolated]);

  // Mouse handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const pixi = pixiRef.current;
    if (!pixi) return;

    dragRef.current.startX = e.clientX;
    dragRef.current.startY = e.clientY;
    dragRef.current.didDrag = false;

    // Minimap click — pan camera to that world position
    if (pixi.hitTestMinimap(e.clientX, e.clientY)) {
      pixi.isMinimapDragging = true;
      pixi.minimapPanTo(e.clientX, e.clientY);
      dragRef.current.didDrag = true;
      return;
    }

    if (!pixi.camera.isZoomed) return;
    dragRef.current.active = true;
    dragRef.current.lastX = e.clientX;
    dragRef.current.lastY = e.clientY;
  }, []);

  const handleMouseUp = useCallback(() => {
    dragRef.current.active = false;
    if (pixiRef.current) pixiRef.current.isMinimapDragging = false;
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const pixi = pixiRef.current;
    if (!pixi) return;

    // Track mouse movement for click-vs-drag detection (only during active canvas drag)
    if (e.buttons > 0 && dragRef.current.active) {
      const mdx = e.clientX - dragRef.current.startX;
      const mdy = e.clientY - dragRef.current.startY;
      if (mdx * mdx + mdy * mdy > 16) {
        if (!dragRef.current.didDrag) setInspection(null);
        dragRef.current.didDrag = true;
      }
    }

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

    // Hit detection against the last rendered clustered snapshot
    const world = pixi.camera.screenToWorld(e.clientX, e.clientY);
    const currentAgents = clusteredRef.current;
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
  }, [species, biomes, biomeCentersMap]);

  const handleMouseLeave = useCallback(() => {
    setHovered(null);
    dragRef.current.active = false;
    if (pixiRef.current) pixiRef.current.isMinimapDragging = false;
  }, []);

  const handleDoubleClick = useCallback(() => {
    setInspection(null);
    pixiRef.current?.camera.reset();
  }, []);

  const handleClick = useCallback((e: React.MouseEvent) => {
    if (dragRef.current.didDrag) return;
    const pixi = pixiRef.current;
    if (!pixi) return;
    if (pixi.hitTestMinimap(e.clientX, e.clientY)) return;

    const world = pixi.camera.screenToWorld(e.clientX, e.clientY);
    const raw = rawAgentsRef.current;
    const speciesMap = new Map(species.map(s => [s.sid, s]));

    const found: InspectionData["entities"] = [];
    for (const a of raw) {
      if (a.dead) continue;
      const dx = world.x - a.x;
      const dy = world.y - a.y;
      if (dx * dx + dy * dy < INSPECTION_HIT_RADIUS * INSPECTION_HIT_RADIUS) {
        found.push({ agent: a, species: speciesMap.get(a.species_sid) });
      }
    }

    if (found.length >= 1) {
      found.sort((a, b) => {
        const da = (world.x - a.agent.x) ** 2 + (world.y - a.agent.y) ** 2;
        const db = (world.x - b.agent.x) ** 2 + (world.y - b.agent.y) ** 2;
        return da - db;
      });
      setInspection({
        entities: found.slice(0, MAX_INSPECTION_ENTITIES),
        totalCount: found.length,
        screenX: e.clientX,
        screenY: e.clientY,
      });
    } else {
      setInspection(null);
    }
  }, [species]);

  const isZoomed = pixiRef.current?.camera.isZoomed ?? false;

  return (
    <div
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onDoubleClick={handleDoubleClick}
      onClick={handleClick}
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
      {hovered && !dragRef.current.active && !inspection && (
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
            fontSize: "16px",
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

      {/* Inspection popup */}
      {inspection && (
        <InspectionPopup data={inspection} onClose={() => setInspection(null)} />
      )}
    </div>
  );
}

function AgentTooltip({ entity }: { entity: Extract<HoveredEntity, { type: "agent" }> }) {
  const { agent, species: sp } = entity;
  const isCluster = agent.uid.startsWith("cluster_");
  const isProxy = agent.uid.startsWith("proxy_");

  if (isProxy) {
    return (
      <>
        <div style={{ fontWeight: 700, color: "#fff", marginBottom: 2, fontSize: "17px" }}>
          S{agent.species_sid}
          {sp && <span style={{ color: "#888" }}> ({sp.plan})</span>}
        </div>
        <div>
          <span style={{ color: "#666" }}>Pop:</span> {sp?.population ?? "?"}
        </div>
        <div>
          <span style={{ color: "#666" }}>Species Age:</span> {agent.age}
        </div>
        {agent.count > 1 && (
          <div>
            <span style={{ color: "#666" }}>Group:</span> ~{agent.count}
          </div>
        )}
      </>
    );
  }

  if (isCluster) {
    return (
      <>
        <div style={{ fontWeight: 700, color: "#fff", marginBottom: 2, fontSize: "17px" }}>
          Cluster ({agent.count} agents)
        </div>
        <div>
          <span style={{ color: "#666" }}>Species:</span>{" "}
          S{agent.species_sid}
          {sp && <span style={{ color: "#888" }}> ({sp.plan})</span>}
        </div>
        <div>
          <span style={{ color: "#666" }}>Avg Energy:</span>{" "}
          <span style={{ color: agent.energy < 20 ? "#ff4" : "#ccc" }}>
            {Math.round(agent.energy)}
          </span>
        </div>
        <div>
          <span style={{ color: "#666" }}>Dominant State:</span>{" "}
          <span style={{ color: "#8cf" }}>{agent.state}</span>
        </div>
        <div>
          <span style={{ color: "#666" }}>Pos:</span>{" "}
          ({Math.round(agent.x)}, {Math.round(agent.y)})
        </div>
      </>
    );
  }

  return (
    <>
      <div style={{ fontWeight: 700, color: "#fff", marginBottom: 2, fontSize: "17px" }}>
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
    </>
  );
}

function VegetationTooltip({ entity }: { entity: Extract<HoveredEntity, { type: "vegetation" }> }) {
  const { patch, biome } = entity;
  return (
    <>
      <div style={{ fontWeight: 700, color: "#6c6", marginBottom: 2, fontSize: "17px" }}>
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
      <div style={{ fontWeight: 700, color: "#8cf", marginBottom: 2, fontSize: "17px" }}>
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
        fontSize: "19px",
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

function InspectionPopup({ data, onClose }: { data: InspectionData; onClose: () => void }) {
  const [pos, setPos] = useState(() => ({
    x: Math.min(data.screenX + 20, window.innerWidth - POPUP_WIDTH - 20),
    y: Math.max(20, Math.min(data.screenY - 40, window.innerHeight * 0.7 - 60)),
  }));
  const drag = useRef({ active: false, startX: 0, startY: 0, originX: 0, originY: 0 });

  // Reset position when click location changes
  useEffect(() => {
    setPos({
      x: Math.min(data.screenX + 20, window.innerWidth - POPUP_WIDTH - 20),
      y: Math.max(20, Math.min(data.screenY - 40, window.innerHeight * 0.7 - 60)),
    });
  }, [data.screenX, data.screenY]);

  // Pointer capture drag — all pointer events route to the header once captured
  const onHeaderPointerDown = useCallback((e: React.PointerEvent) => {
    e.stopPropagation();
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { active: true, startX: e.clientX, startY: e.clientY, originX: pos.x, originY: pos.y };
  }, [pos.x, pos.y]);

  const onHeaderPointerMove = useCallback((e: React.PointerEvent) => {
    if (!drag.current.active) return;
    e.stopPropagation();
    setPos({
      x: drag.current.originX + (e.clientX - drag.current.startX),
      y: drag.current.originY + (e.clientY - drag.current.startY),
    });
  }, []);

  const onHeaderPointerUp = useCallback((e: React.PointerEvent) => {
    drag.current.active = false;
    e.stopPropagation();
  }, []);

  const iconCache = useMemo(() => {
    const cache = new Map<string, string>();
    for (const { species } of data.entities) {
      if (species && !cache.has(species.plan)) {
        cache.set(species.plan, getCreatureIconDataURL(species.plan));
      }
    }
    return cache;
  }, [data.entities]);

  return (
    <div
      data-inspection-popup
      onClick={e => e.stopPropagation()}
      onMouseDown={e => e.stopPropagation()}
      onMouseMove={e => e.stopPropagation()}
      style={{
        position: "fixed",
        left: pos.x,
        top: pos.y,
        width: POPUP_WIDTH,
        maxHeight: "70vh",
        background: "rgba(8, 10, 20, 0.96)",
        border: "1px solid rgba(80, 160, 255, 0.25)",
        borderRadius: "8px",
        boxShadow: "0 0 24px rgba(40, 100, 255, 0.12), 0 8px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
        backdropFilter: "blur(16px)",
        zIndex: 30,
        fontFamily: "'Rajdhani', monospace",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column" as const,
        animation: "inspection-appear 0.15s ease-out",
      }}
    >
      {/* Scanline overlay */}
      <div style={{
        position: "absolute",
        inset: 0,
        background: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0, 0, 0, 0.03) 2px, rgba(0, 0, 0, 0.03) 4px)",
        pointerEvents: "none",
        borderRadius: "8px",
        zIndex: 1,
      }} />

      {/* Header — drag handle */}
      <div
        onPointerDown={onHeaderPointerDown}
        onPointerMove={onHeaderPointerMove}
        onPointerUp={onHeaderPointerUp}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 14px",
          borderBottom: "1px solid rgba(80, 160, 255, 0.12)",
          background: "rgba(30, 50, 80, 0.3)",
          position: "relative",
          zIndex: 2,
          cursor: "grab",
          userSelect: "none",
          touchAction: "none",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{
            color: "#6ab0ff",
            fontSize: "14px",
            fontWeight: 700,
            letterSpacing: "2px",
            textTransform: "uppercase" as const,
          }}>INSPECT</span>
          <span style={{
            background: "rgba(80, 160, 255, 0.15)",
            color: "#8ac4ff",
            padding: "2px 8px",
            borderRadius: "4px",
            fontSize: "15px",
            fontWeight: 600,
          }}>
            {data.totalCount} {data.totalCount === 1 ? "entity" : "entities"}
          </span>
        </div>
        <button
          onClick={onClose}
          onPointerDown={e => e.stopPropagation()}
          style={{
            background: "none",
            border: "none",
            color: "#556",
            cursor: "pointer",
            fontSize: "20px",
            padding: "0 4px",
            lineHeight: "1",
          }}
          onMouseEnter={e => { e.currentTarget.style.color = "#aab"; }}
          onMouseLeave={e => { e.currentTarget.style.color = "#556"; }}
        >
          ✕
        </button>
      </div>

      {/* Entity list */}
      <div style={{
        overflowY: "auto",
        flex: 1,
        padding: "4px 0",
        position: "relative",
        zIndex: 2,
      }}>
        {data.entities.map((entry, i) => (
          <EntityCard
            key={entry.agent.uid}
            agent={entry.agent}
            species={entry.species}
            iconUrl={entry.species ? iconCache.get(entry.species.plan) : undefined}
            isLast={i === data.entities.length - 1}
          />
        ))}
        {data.totalCount > MAX_INSPECTION_ENTITIES && (
          <div style={{
            padding: "8px 14px",
            color: "#556",
            fontSize: "14px",
            textAlign: "center",
            fontStyle: "italic",
          }}>
            +{data.totalCount - MAX_INSPECTION_ENTITIES} more
          </div>
        )}
      </div>
    </div>
  );
}

function EntityCard({ agent, species, iconUrl, isLast }: {
  agent: AgentSnapshot;
  species?: SpeciesInfo;
  iconUrl?: string;
  isLast: boolean;
}) {
  const bodyPlan = species?.plan || "grazer";
  const planColor = BODY_PLAN_COLORS[bodyPlan] || "#888";
  const isCluster = agent.uid.startsWith("cluster_");
  const isProxy = agent.uid.startsWith("proxy_");
  const stateColor = STATE_BADGE_COLORS[agent.state] || "#777";

  const traits = species?.traits || {};
  const displayTraits = TRAIT_DISPLAY_ORDER
    .filter(t => traits[t] != null)
    .slice(0, 4)
    .map(t => ({ name: t, value: traits[t] }));

  return (
    <div style={{
      padding: "10px 14px",
      borderBottom: isLast ? "none" : "1px solid rgba(255, 255, 255, 0.04)",
      borderLeft: `3px solid ${planColor}`,
      marginLeft: "4px",
      display: "flex",
      gap: "10px",
    }}>
      {/* Creature icon */}
      {iconUrl && (
        <div style={{
          width: 36,
          height: 36,
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: `radial-gradient(circle, ${planColor}22 0%, transparent 70%)`,
          borderRadius: "6px",
        }}>
          <img src={iconUrl} width={32} height={32} style={{ imageRendering: "pixelated" }} alt={bodyPlan} />
        </div>
      )}

      {/* Info block */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Name + plan badge */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginBottom: "4px",
        }}>
          <span style={{ color: "#fff", fontWeight: 700, fontSize: "17px" }}>
            {isCluster ? `Cluster (${agent.count})` : isProxy ? `S${agent.species_sid}` : `#${agent.aid}`}
          </span>
          <span style={{
            background: `${planColor}33`,
            color: planColor,
            padding: "1px 6px",
            borderRadius: "3px",
            fontSize: "12px",
            fontWeight: 700,
            letterSpacing: "0.5px",
            textTransform: "uppercase" as const,
          }}>
            {bodyPlan.replace(/_/g, " ")}
          </span>
        </div>

        {/* Energy bar */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          marginBottom: "4px",
        }}>
          <div style={{
            flex: 1,
            height: 6,
            background: "rgba(255, 255, 255, 0.06)",
            borderRadius: 3,
            overflow: "hidden",
          }}>
            <div style={{
              width: `${Math.min(100, Math.max(0, agent.energy))}%`,
              height: "100%",
              background: `linear-gradient(90deg, ${energyColor(agent.energy)}, ${energyColor(agent.energy)}cc)`,
              borderRadius: 3,
            }} />
          </div>
          <span style={{
            color: energyColor(agent.energy),
            fontSize: "14px",
            fontWeight: 600,
            minWidth: "24px",
            textAlign: "right" as const,
          }}>
            {Math.round(agent.energy)}
          </span>
        </div>

        {/* Stats: age, state, target */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          fontSize: "14px",
          flexWrap: "wrap" as const,
        }}>
          <span style={{ color: "#667" }}>Age {agent.age}</span>
          <span style={{
            background: `${stateColor}22`,
            color: stateColor,
            padding: "1px 5px",
            borderRadius: "3px",
            fontSize: "12px",
            fontWeight: 700,
            textTransform: "uppercase" as const,
          }}>
            {agent.state}
          </span>
          {agent.target_aid != null && (
            <span style={{ color: "#667" }}>→ #{agent.target_aid}</span>
          )}
        </div>

        {/* Trait bars */}
        {displayTraits.length > 0 && (
          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "2px 10px",
            marginTop: "6px",
          }}>
            {displayTraits.map(t => (
              <div key={t.name} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <span style={{
                  color: "#556",
                  fontSize: "11px",
                  fontWeight: 600,
                  textTransform: "uppercase" as const,
                  width: "30px",
                  flexShrink: 0,
                }}>
                  {t.name.slice(0, 3)}
                </span>
                <div style={{
                  flex: 1,
                  height: 3,
                  background: "rgba(255, 255, 255, 0.04)",
                  borderRadius: 2,
                  overflow: "hidden",
                }}>
                  <div style={{
                    width: `${Math.min(100, Math.max(0, t.value * 100))}%`,
                    height: "100%",
                    background: `${planColor}88`,
                    borderRadius: 2,
                  }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
