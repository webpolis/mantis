/**
 * Barrel re-exports for backward compatibility.
 * Consumers (PixiApp, SimulationCanvas, SpeciesPanel) import from here.
 */
export { BODY_PLAN_COLORS } from "./palette";
export { CreatureRenderer } from "./CreatureRenderer";
export { getSpriteSheet, destroySpriteSheet } from "./spriteSheet";

// Re-export types used externally
export { Direction, VisualState, SPRITE_SIZE, ICON_SIZE, ANIM_FRAMES } from "./types";
export type { BodyPlan, DrawParams } from "./types";

import { getSpriteSheet } from "./spriteSheet";

/** Get a data URL for a creature icon (for use in React components). */
export function getCreatureIconDataURL(bodyPlan: string): string {
  return getSpriteSheet().getIconDataURL(bodyPlan);
}
