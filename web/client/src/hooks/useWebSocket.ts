import { useCallback, useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";
import type { AgentSnapshot, SpeciesInfo, TickData, SimulationInfo } from "../types/simulation";

export function useWebSocket() {
  const socketRef = useRef<Socket | null>(null);
  const [tick, setTick] = useState(0);
  const [epoch, setEpoch] = useState(1);
  const [species, setSpecies] = useState<SpeciesInfo[]>([]);
  const [agents, setAgents] = useState<AgentSnapshot[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [info, setInfo] = useState<SimulationInfo | null>(null);
  const [interpolateDuration, setInterpolateDuration] = useState(66.7);

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

    return () => {
      socket.disconnect();
    };
  }, []);

  const play = useCallback((mode: "file" | "live" = "live") => {
    const socket = socketRef.current;
    if (!socket) return;
    setIsPlaying(true);
    if (mode === "live") {
      socket.emit("start_live", {
        max_generations: 200,
        seed: Math.floor(Math.random() * 2147483647),
        enable_agents: true,
        agent_epoch: "ECOSYSTEM",
      });
    } else {
      socket.emit("start_simulation", {});
    }
  }, []);

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
  };
}
