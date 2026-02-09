export interface AgentSnapshot {
  aid: number;
  species_sid: number;
  x: number;
  y: number;
  energy: number;
  age: number;
  state: string;
  target_aid: number | null;
  dead: boolean;
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
}
