import { useCallback, useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";
import type { AgentSnapshot, SpeciesInfo, TickData, SimulationInfo, DatasetFile, WorldList, ModelFile, BiomeData, VegetationPatchData } from "../types/simulation";

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
  const [selectedWorld, setSelectedWorld] = useState(0);
  const [models, setModels] = useState<ModelFile[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [biomes, setBiomes] = useState<BiomeData[]>([]);

  useEffect(() => {
    const socket = io(window.location.origin, {
      transports: ["websocket", "polling"],
    });
    socketRef.current = socket;

    socket.on("tick_update", (data: TickData) => {
      setTick(data.tick);
      setEpoch(data.epoch);
      if (data.species.length > 0) setSpecies(data.species);
      setAgents(data.agents);
      setInterpolateDuration(data.interpolate_duration);
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
      setSelectedWorld(0);
    });

    socket.on("model_list", (data: ModelFile[]) => {
      setModels(data);
    });

    socket.on("environment_init", (data: { biomes: BiomeData[] }) => {
      const enriched = data.biomes.map((b) => ({
        ...b,
        patches: b.patches.length > 0 ? b.patches : generateDeterministicPatches(b.lid, b.vegetation),
      }));
      setBiomes(enriched);
    });

    socket.on("vegetation_update", (data: { patches: Array<{ lid: number; x: number; y: number; density: number }> }) => {
      setBiomes((prev) => {
        const updated = prev.map((b) => ({ ...b, patches: [...b.patches] }));
        for (const upd of data.patches) {
          const biome = updated.find((b) => b.lid === upd.lid);
          if (!biome) continue;
          const patch = biome.patches.find((p) => Math.abs(p.x - upd.x) < 1 && Math.abs(p.y - upd.y) < 1);
          if (patch) patch.density = upd.density;
        }
        return updated;
      });
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
    setSelectedWorld(0);
    socketRef.current?.emit("list_worlds", { file: name });
  }, []);

  const selectWorld = useCallback((index: number) => {
    setSelectedWorld(index);
  }, []);

  const selectModel = useCallback((name: string) => {
    setSelectedModel(name);
  }, []);

  const play = useCallback((mode: "file" | "live" | "model" = "live", file?: string, worldIndex?: number) => {
    const socket = socketRef.current;
    if (!socket) return;
    setIsPlaying(true);
    setBiomes([]);
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
    selectedWorld,
    selectFile,
    selectWorld,
    models,
    selectedModel,
    selectModel,
    biomes,
  };
}
