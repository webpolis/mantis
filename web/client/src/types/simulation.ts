export interface AgentSnapshot {
  uid: string;
  aid: number;
  species_sid: number;
  x: number;
  y: number;
  energy: number;
  age: number;
  state: string;
  target_aid: number | null;
  dead: boolean;
  count: number;
}

export interface SpeciesInfo {
  sid: number;
  plan: string;
  population: number;
  locations: string[];
}

export interface TickData {
  tick: number;
  epoch: number;
  species: SpeciesInfo[];
  agents: AgentSnapshot[];
  interpolate_duration: number;
  events?: SimulationEvent[];
  biomes?: BiomeData[];
}

export interface VegetationPatchData {
  x: number;
  y: number;
  radius: number;
  density: number;
}

export interface BiomeData {
  lid: number;
  name: string;
  vegetation: number;
  detritus: number;
  nitrogen: number;
  phosphorus: number;
  patches: VegetationPatchData[];
}

export interface SimulationEvent {
  target: string;
  event_type: string;
  detail: string;
}

export interface HistoryFrame {
  tick: number;
  epoch: number;
  species: SpeciesInfo[];
  agents: AgentSnapshot[];
  biomes: BiomeData[];
  events: SimulationEvent[];
}

export interface SimulationInfo {
  total_ticks: number;
  file?: string;
  mode?: string;
  world_index?: number;
  world_count?: number;
}

export interface DatasetFile {
  name: string;
  size: number;
}

export interface WorldList {
  file: string;
  world_count: number;
  worlds_with_agents: number[];
  worlds_with_spotlights: number[];
}

export interface ModelFile {
  name: string;
  size: number;
}
