/**
 * Color palettes per body plan, visual-state modifiers, and biome tints.
 */
import type { BodyPlan } from "./types";
import { VisualState } from "./types";

export interface BodyPalette {
  primary: string;
  secondary: string;
  accent: string;
  eye: string;
  belly: string;
  glow: string;
}

export const PALETTES: Record<BodyPlan, BodyPalette> = {
  predator: {
    primary: "#cc2233",
    secondary: "#991a28",
    accent: "#ff6666",
    eye: "#ff2200",
    belly: "#771122",
    glow: "#ff4444",
  },
  grazer: {
    primary: "#66bb77",
    secondary: "#449955",
    accent: "#88ddaa",
    eye: "#223322",
    belly: "#88cc99",
    glow: "#aaffbb",
  },
  omnivore: {
    primary: "#8855aa",
    secondary: "#6b3d8a",
    accent: "#aa77cc",
    eye: "#eecc88",
    belly: "#aa77cc",
    glow: "#cc99ee",
  },
  scavenger: {
    primary: "#887744",
    secondary: "#665533",
    accent: "#ee8833",
    eye: "#ffcc66",
    belly: "#aa9955",
    glow: "#ffaa44",
  },
  decomposer: {
    primary: "rgba(70, 100, 60, 0.85)",
    secondary: "rgba(60, 90, 50, 0.75)",
    accent: "rgba(130, 160, 110, 0.5)",
    eye: "",
    belly: "rgba(80, 110, 70, 0.6)",
    glow: "rgba(120, 180, 100, 0.3)",
  },
  sessile_autotroph: {
    primary: "#1a8866",
    secondary: "#22aa88",
    accent: "#50ffdc",
    eye: "",
    belly: "#166b50",
    glow: "rgba(80, 255, 220, 0.7)",
  },
  mobile_autotroph: {
    primary: "rgba(60, 180, 100, 0.7)",
    secondary: "rgba(100, 220, 120, 0.7)",
    accent: "rgba(100, 200, 130, 0.5)",
    eye: "",
    belly: "rgba(80, 160, 90, 0.5)",
    glow: "rgba(80, 255, 120, 0.15)",
  },
  filter_feeder: {
    primary: "#4466aa",
    secondary: "#3355aa",
    accent: "#6688cc",
    eye: "",
    belly: "#3355aa",
    glow: "rgba(130, 150, 220, 0.3)",
  },
  parasite: {
    primary: "#442233",
    secondary: "#663355",
    accent: "#cc3355",
    eye: "",
    belly: "#331122",
    glow: "#ff4466",
  },
};

/** Hex string used by React components (SpeciesPanel, SimulationCanvas). */
export const BODY_PLAN_COLORS: Record<string, string> = {
  predator: "#cc2233",
  grazer: "#66bb77",
  omnivore: "#9966cc",
  scavenger: "#bb9944",
  decomposer: "#557755",
  sessile_autotroph: "#22ccaa",
  mobile_autotroph: "#44bb66",
  filter_feeder: "#5577cc",
  parasite: "#cc4488",
};

export interface StateModifier {
  saturation: number;   // multiplier (1 = normal)
  brightness: number;   // multiplier
  scale: number;        // multiplier on sprite scale
}

export const STATE_MODIFIERS: Record<VisualState, StateModifier> = {
  [VisualState.IDLE]: { saturation: 1.0, brightness: 1.0, scale: 1.0 },
  [VisualState.HUNT]: { saturation: 1.3, brightness: 1.1, scale: 1.05 },
  [VisualState.FLEE]: { saturation: 0.7, brightness: 0.85, scale: 0.88 },
  [VisualState.MATE]: { saturation: 1.4, brightness: 1.2, scale: 1.1 },
  [VisualState.HURT]: { saturation: 0.5, brightness: 0.7, scale: 0.82 },
};

/** Biome tint values applied via sprite.tint (GPU-side). 0xffffff = no tint. */
export const BIOME_TINTS: Record<string, number> = {
  tundra: 0xbbccee,
  alpine: 0xaabbdd,
  desert: 0xeedd99,
  volcanic_vent: 0xffbb77,
  swamp: 0xaaddaa,
  cave: 0xccaaee,
  deep_ocean: 0x8899cc,
  // All other biomes: no tint
};

export function getBiomeTint(biomeName: string): number {
  return BIOME_TINTS[biomeName] ?? 0xffffff;
}
