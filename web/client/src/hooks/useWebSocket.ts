import { useCallback, useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";
import type { AgentSnapshot, SpeciesInfo, TickData, SimulationInfo, DatasetFile, WorldList, ModelFile, BiomeData, VegetationPatchData, HistoryFrame, SimulationEvent } from "../types/simulation";

export interface EventLogEntry {
  tick: number;
  event: SimulationEvent;
}

/** Mulberry32 seeded PRNG for deterministic vegetation patches. */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateDeterministicPatches(lid: number, vegetation: number): VegetationPatchData[] {
  const rng = mulberry32(lid * 7919 + 137);
  const count = 5 + Math.floor(rng() * 4); // 5-8 patches
  const patches: VegetationPatchData[] = [];
  for (let i = 0; i < count; i++) {
    patches.push({
      x: rng() * 1000,
      y: rng() * 1000,
      radius: 80 + rng() * 40,
      density: vegetation * (0.5 + rng() * 0.5),
    });
  }
  return patches;
}

export function useWebSocket() {
  const socketRef = useRef<Socket | null>(null);
  const [tick, setTick] = useState(0);
  const [epoch, setEpoch] = useState(1);
  const [species, setSpecies] = useState<SpeciesInfo[]>([]);
  const [agents, setAgents] = useState<AgentSnapshot[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [info, setInfo] = useState<SimulationInfo | null>(null);
  const [interpolateDuration, setInterpolateDuration] = useState(66.7);
  const [datasets, setDatasets] = useState<DatasetFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [worldCount, setWorldCount] = useState(0);
  const [worldsWithAgents, setWorldsWithAgents] = useState<number[]>([]);
  const [worldsWithSpotlights, setWorldsWithSpotlights] = useState<number[]>([]);
  const [selectedWorld, setSelectedWorld] = useState(0);
  const [models, setModels] = useState<ModelFile[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [biomes, setBiomes] = useState<BiomeData[]>([]);
  const [events, setEvents] = useState<SimulationEvent[]>([]);
  const eventLogRef = useRef<EventLogEntry[]>([]);
  const [eventLog, setEventLog] = useState<EventLogEntry[]>([]);
  const historyRef = useRef<HistoryFrame[]>([]);
  const [historyLength, setHistoryLength] = useState(0);
  const [viewIndex, setViewIndex] = useState<number | null>(null);
  const viewIndexRef = useRef<number | null>(null);
  const biomesRef = useRef<BiomeData[]>([]);

  useEffect(() => {
    const socket = io(window.location.origin, {
      transports: ["websocket", "polling"],
    });
    socketRef.current = socket;

    socket.on("tick_update", (data: TickData) => {
      const prevSpecies = historyRef.current.length > 0
        ? historyRef.current[historyRef.current.length - 1].species
        : [];

      // Merge species: carry forward ALL previous species, overlay current tick's updates
      let mergedSpecies: SpeciesInfo[];
      if (data.species.length > 0) {
        const resultMap = new Map(prevSpecies.map((s) => [s.sid, { ...s }]));
        for (const s of data.species) {
          const prev = resultMap.get(s.sid);
          resultMap.set(s.sid, {
            ...s,
            plan: s.plan || prev?.plan || "",
            locations: s.locations.length > 0 ? s.locations : prev?.locations ?? [],
          });
        }
        mergedSpecies = Array.from(resultMap.values());
      } else {
        mergedSpecies = prevSpecies;
      }

      const tickEvents = data.events ?? [];

      // Append to event log
      if (tickEvents.length > 0) {
        const newEntries = tickEvents.map((e) => ({ tick: data.tick, event: e }));
        eventLogRef.current = [...eventLogRef.current, ...newEntries].slice(-200);
        setEventLog([...eventLogRef.current]);
      }

      // Update biome stats when tick includes @BIO data (keyframe ticks)
      if (data.biomes && data.biomes.length > 0) {
        const updated = biomesRef.current.map((b) => {
          const fresh = data.biomes!.find((fb) => fb.lid === b.lid);
          if (!fresh) return b;
          // Scale patch densities proportionally to vegetation change
          const vegRatio = b.vegetation > 0.001 ? fresh.vegetation / b.vegetation : 1;
          const patches = vegRatio !== 1
            ? b.patches.map((p) => ({ ...p, density: Math.min(1, p.density * vegRatio) }))
            : b.patches;
          return {
            ...b,
            vegetation: fresh.vegetation,
            detritus: fresh.detritus,
            nitrogen: fresh.nitrogen ?? b.nitrogen,
            phosphorus: fresh.phosphorus ?? b.phosphorus,
            patches,
          };
        });
        biomesRef.current = updated;
        if (viewIndexRef.current === null) setBiomes(updated);
      }

      const frame: HistoryFrame = {
        tick: data.tick,
        epoch: data.epoch,
        species: mergedSpecies,
        agents: data.agents,
        biomes: biomesRef.current.map((b) => ({ ...b, patches: [...b.patches] })),
        events: tickEvents,
      };
      historyRef.current.push(frame);
      setHistoryLength(historyRef.current.length);

      // Only update displayed state when following live
      if (viewIndexRef.current === null) {
        setTick(data.tick);
        setEpoch(data.epoch);
        setSpecies(mergedSpecies);
        setAgents(data.agents);
        setEvents(tickEvents);
        setInterpolateDuration(data.interpolate_duration);
      }
    });

    socket.on("simulation_info", (data: SimulationInfo) => {
      setInfo(data);
    });

    socket.on("simulation_complete", () => {
      setIsPlaying(false);
    });

    socket.on("error", (data: { message: string }) => {
      console.error("Server error:", data.message);
      setIsPlaying(false);
    });

    socket.on("dataset_list", (data: DatasetFile[]) => {
      setDatasets(data);
    });

    socket.on("world_list", (data: WorldList) => {
      setWorldCount(data.world_count);
      setWorldsWithAgents(data.worlds_with_agents ?? []);
      setWorldsWithSpotlights(data.worlds_with_spotlights ?? []);
      setSelectedWorld(0);
    });

    socket.on("model_list", (data: ModelFile[]) => {
      setModels(data);
    });

    socket.on("environment_init", (data: { biomes: BiomeData[] }) => {
      const enriched = data.biomes.map((b) => ({
        ...b,
        nitrogen: b.nitrogen ?? 0.5,
        phosphorus: b.phosphorus ?? 0.3,
        patches: b.patches.length > 0 ? b.patches : generateDeterministicPatches(b.lid, b.vegetation),
      }));
      biomesRef.current = enriched;
      setBiomes(enriched);
    });

    socket.on("vegetation_update", (data: {
      patches: Array<{ lid: number; x: number; y: number; density: number }>;
      biome_stats?: Array<{ lid: number; vegetation: number; detritus: number; nitrogen: number; phosphorus: number }>;
    }) => {
      // Always update the ref (latest biome state for future frames)
      const updated = biomesRef.current.map((b) => ({ ...b, patches: [...b.patches] }));
      for (const upd of data.patches) {
        const biome = updated.find((b) => b.lid === upd.lid);
        if (!biome) continue;
        const patch = biome.patches.find((p) => Math.abs(p.x - upd.x) < 1 && Math.abs(p.y - upd.y) < 1);
        if (patch) patch.density = upd.density;
      }
      // Merge biome-level stats (vegetation, detritus, N, P)
      if (data.biome_stats) {
        for (const stat of data.biome_stats) {
          const biome = updated.find((b) => b.lid === stat.lid);
          if (!biome) continue;
          biome.vegetation = stat.vegetation;
          biome.detritus = stat.detritus;
          biome.nitrogen = stat.nitrogen ?? biome.nitrogen;
          biome.phosphorus = stat.phosphorus ?? biome.phosphorus;
        }
      }
      biomesRef.current = updated;
      // Update the latest history frame's biome snapshot
      if (historyRef.current.length > 0) {
        historyRef.current[historyRef.current.length - 1].biomes = updated.map((b) => ({ ...b, patches: [...b.patches] }));
      }
      // Only update displayed biomes when following live
      if (viewIndexRef.current === null) {
        setBiomes(updated);
      }
    });

    socket.on("connect", () => {
      socket.emit("list_datasets");
      socket.emit("list_models");
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const selectFile = useCallback((name: string) => {
    setSelectedFile(name);
    setWorldCount(0);
    setWorldsWithAgents([]);
    setWorldsWithSpotlights([]);
    setSelectedWorld(0);
    socketRef.current?.emit("list_worlds", { file: name });
  }, []);

  const selectWorld = useCallback((index: number) => {
    setSelectedWorld(index);
    setAgents([]);
    setSpecies([]);
    setBiomes([]);
    setTick(0);
    setEpoch(1);
    setInfo(null);
  }, []);

  const selectModel = useCallback((name: string) => {
    setSelectedModel(name);
  }, []);

  const play = useCallback((mode: "file" | "live" | "model" = "live", file?: string, worldIndex?: number) => {
    const socket = socketRef.current;
    if (!socket) return;
    historyRef.current = [];
    setHistoryLength(0);
    eventLogRef.current = [];
    setEventLog([]);
    setEvents([]);
    viewIndexRef.current = null;
    setViewIndex(null);
    setIsPlaying(true);
    setBiomes([]);
    biomesRef.current = [];
    if (mode === "model") {
      socket.emit("start_model", {
        model: selectedModel,
        temperature: 0.8,
        max_tokens: 4096,
      });
    } else if (mode === "live") {
      socket.emit("start_live", {
        max_generations: 200,
        seed: Math.floor(Math.random() * 2147483647),
        enable_agents: true,
        agent_epoch: "ECOSYSTEM",
      });
    } else {
      socket.emit("start_simulation", {
        file: file ?? undefined,
        world_index: worldIndex ?? 0,
      });
    }
  }, [selectedModel]);

  const pause = useCallback(() => {
    socketRef.current?.emit("pause");
    setIsPlaying(false);
  }, []);

  const resume = useCallback(() => {
    socketRef.current?.emit("resume");
    setIsPlaying(true);
  }, []);

  const setSpeed = useCallback((speed: number) => {
    socketRef.current?.emit("set_speed", { speed });
  }, []);

  const seekTo = useCallback((index: number) => {
    const frame = historyRef.current[index];
    if (!frame) return;
    viewIndexRef.current = index;
    setViewIndex(index);
    setTick(frame.tick);
    setEpoch(frame.epoch);
    setSpecies(frame.species);
    setAgents(frame.agents);
    setBiomes(frame.biomes);
    setEvents(frame.events);
  }, []);

  const followLatest = useCallback(() => {
    viewIndexRef.current = null;
    setViewIndex(null);
    const last = historyRef.current[historyRef.current.length - 1];
    if (last) {
      setTick(last.tick);
      setEpoch(last.epoch);
      setSpecies(last.species);
      setAgents(last.agents);
      setBiomes(last.biomes);
      setEvents(last.events);
    }
  }, []);

  const isFollowing = viewIndex === null;

  // Build epoch segments for the timeline
  const epochs: Array<{ epoch: number; startIndex: number; endIndex: number }> = [];
  if (historyRef.current.length > 0) {
    let current = historyRef.current[0].epoch;
    let start = 0;
    for (let i = 1; i < historyRef.current.length; i++) {
      if (historyRef.current[i].epoch !== current) {
        epochs.push({ epoch: current, startIndex: start, endIndex: i - 1 });
        current = historyRef.current[i].epoch;
        start = i;
      }
    }
    epochs.push({ epoch: current, startIndex: start, endIndex: historyRef.current.length - 1 });
  }

  return {
    tick,
    epoch,
    species,
    agents,
    isPlaying,
    info,
    interpolateDuration,
    play,
    pause,
    resume,
    setSpeed,
    datasets,
    selectedFile,
    worldCount,
    worldsWithAgents,
    worldsWithSpotlights,
    selectedWorld,
    selectFile,
    selectWorld,
    models,
    selectedModel,
    selectModel,
    biomes,
    events,
    eventLog,
    historyLength,
    viewIndex,
    isFollowing,
    seekTo,
    followLatest,
    epochs,
  };
}
