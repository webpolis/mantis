/**
 * Shared types, enums, and constants for the creature sprite system.
 */

export const BODY_PLANS = [
  "predator",
  "grazer",
  "omnivore",
  "scavenger",
  "decomposer",
  "sessile_autotroph",
  "mobile_autotroph",
  "filter_feeder",
  "parasite",
] as const;

export type BodyPlan = (typeof BODY_PLANS)[number];

export enum Direction {
  DOWN = 0,
  UP = 1,
  RIGHT = 2,
  LEFT = 3,
}

/** Drawn directions (LEFT is a horizontal flip of RIGHT). */
export const DRAWN_DIRECTIONS = [Direction.DOWN, Direction.UP, Direction.RIGHT] as const;
export type DrawnDirection = (typeof DRAWN_DIRECTIONS)[number];

export enum VisualState {
  IDLE = 0,
  HUNT = 1,
  FLEE = 2,
  MATE = 3,
  HURT = 4,
}

/** Maps behavioral state strings to VisualState. */
export function behaviorToVisualState(state: string, energy: number): VisualState {
  if (energy < 20) return VisualState.HURT;
  switch (state) {
    case "hunt":
      return VisualState.HUNT;
    case "flee":
      return VisualState.FLEE;
    case "mate":
      return VisualState.MATE;
    default:
      return VisualState.IDLE;
  }
}

export const SPRITE_SIZE = 96;
export const ICON_SIZE = 32;
export const ANIM_FRAMES = 6;
export const ANIM_PERIOD = 1200; // ms per cycle

export interface DrawParams {
  ctx: CanvasRenderingContext2D;
  size: number;
  frame: number;          // 0 .. ANIM_FRAMES-1
  direction: DrawnDirection;
  state: VisualState;
}
