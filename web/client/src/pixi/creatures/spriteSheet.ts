/**
 * SpriteSheetGenerator: pre-generates all creature textures at startup.
 * 9 plans × 5 states × 3 drawn directions × 6 frames = 810 sprite textures + 9 icons.
 * All rendered via Canvas 2D → PIXI.Texture, cached for the app lifetime.
 */
import { Texture } from "pixi.js";
import {
  BODY_PLANS, DRAWN_DIRECTIONS, SPRITE_SIZE, ICON_SIZE, ANIM_FRAMES,
  VisualState, Direction,
} from "./types";
import type { BodyPlan, DrawnDirection } from "./types";
import { DRAW_FUNCTIONS } from "./drawBodies";
import { STATE_MODIFIERS } from "./palette";

/** Cache key: "predator|HUNT|RIGHT|3" */
function cacheKey(plan: BodyPlan, state: VisualState, dir: DrawnDirection, frame: number): string {
  return `${plan}|${state}|${dir}|${frame}`;
}

const ALL_VISUAL_STATES = [
  VisualState.IDLE,
  VisualState.HUNT,
  VisualState.FLEE,
  VisualState.MATE,
  VisualState.HURT,
];

/**
 * Apply saturation + brightness adjustment to ImageData pixels.
 * Works on all browsers (no ctx.filter dependency).
 */
function applyColorModifiers(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  saturation: number, brightness: number,
) {
  if (saturation === 1 && brightness === 1) return;

  const imgData = ctx.getImageData(0, 0, w, h);
  const d = imgData.data;

  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] === 0) continue; // skip fully transparent

    let r = d[i], g = d[i + 1], b = d[i + 2];

    // Saturation via luminance interpolation
    if (saturation !== 1) {
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      r = lum + (r - lum) * saturation;
      g = lum + (g - lum) * saturation;
      b = lum + (b - lum) * saturation;
    }

    // Brightness
    if (brightness !== 1) {
      r *= brightness;
      g *= brightness;
      b *= brightness;
    }

    d[i]     = Math.max(0, Math.min(255, r + 0.5)) | 0;
    d[i + 1] = Math.max(0, Math.min(255, g + 0.5)) | 0;
    d[i + 2] = Math.max(0, Math.min(255, b + 0.5)) | 0;
  }

  ctx.putImageData(imgData, 0, 0);
}

export class SpriteSheetGenerator {
  private textures = new Map<string, Texture>();
  private icons = new Map<string, Texture>();
  private iconDataURLs = new Map<string, string>();

  /** Generate all textures. Call once during app init. */
  generate() {
    for (const plan of BODY_PLANS) {
      const drawFn = DRAW_FUNCTIONS[plan];

      for (const state of ALL_VISUAL_STATES) {
        const mod = STATE_MODIFIERS[state];

        for (const dir of DRAWN_DIRECTIONS) {
          for (let frame = 0; frame < ANIM_FRAMES; frame++) {
            const canvas = document.createElement("canvas");
            canvas.width = SPRITE_SIZE;
            canvas.height = SPRITE_SIZE;
            const ctx = canvas.getContext("2d")!;
            ctx.clearRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
            drawFn({ ctx, size: SPRITE_SIZE, frame, direction: dir, state });
            applyColorModifiers(ctx, SPRITE_SIZE, SPRITE_SIZE, mod.saturation, mod.brightness);
            this.textures.set(cacheKey(plan, state, dir, frame), Texture.from(canvas));
          }
        }
      }

      // Icon: IDLE, DOWN, frame 0 at 32x32
      const iconCanvas = document.createElement("canvas");
      iconCanvas.width = ICON_SIZE;
      iconCanvas.height = ICON_SIZE;
      const iconCtx = iconCanvas.getContext("2d")!;
      iconCtx.clearRect(0, 0, ICON_SIZE, ICON_SIZE);
      drawFn({ ctx: iconCtx, size: ICON_SIZE, frame: 0, direction: Direction.DOWN, state: VisualState.IDLE });
      this.icons.set(plan, Texture.from(iconCanvas));
      this.iconDataURLs.set(plan, iconCanvas.toDataURL());
    }
  }

  /** Get a sprite texture for the given combination. */
  getTexture(plan: BodyPlan, state: VisualState, dir: DrawnDirection, frame: number): Texture {
    return this.textures.get(cacheKey(plan, state, dir, frame)) ?? this.textures.get(cacheKey("grazer", VisualState.IDLE, Direction.DOWN, 0))!;
  }

  /** Get the 32x32 icon texture for a body plan. */
  getIcon(plan: string): Texture {
    return this.icons.get(plan as BodyPlan) ?? this.icons.get("grazer")!;
  }

  /** Get a data URL for a creature icon (for React components). */
  getIconDataURL(plan: string): string {
    return this.iconDataURLs.get(plan as BodyPlan) ?? this.iconDataURLs.get("grazer")!;
  }

  destroy() {
    for (const tex of this.textures.values()) tex.destroy(true);
    for (const tex of this.icons.values()) tex.destroy(true);
    this.textures.clear();
    this.icons.clear();
    this.iconDataURLs.clear();
  }
}

/** Singleton — created lazily, shared across the app. */
let _instance: SpriteSheetGenerator | null = null;

export function getSpriteSheet(): SpriteSheetGenerator {
  if (!_instance) {
    _instance = new SpriteSheetGenerator();
    _instance.generate();
  }
  return _instance;
}

export function destroySpriteSheet() {
  _instance?.destroy();
  _instance = null;
}
