/**
 * Creature renderer with direction tracking, visual-state texture selection,
 * biome tinting, and sprite pooling. Replaces the old monolithic renderer.
 *
 * Key changes from the old version:
 * - 4-directional facing derived from position deltas
 * - Visual states shown as different sprite textures (no stateRing Graphics)
 * - Biome tinting via sprite.tint (GPU-side)
 * - 6-frame animation, 96px sprites
 */
import { Container, Sprite, Graphics, Text, TextStyle } from "pixi.js";
import type { AgentSnapshot, SpeciesInfo, BiomeData } from "../../types/simulation";
import {
  Direction, VisualState, SPRITE_SIZE, ANIM_FRAMES, ANIM_PERIOD,
  behaviorToVisualState,
} from "./types";
import type { DrawnDirection } from "./types";
import { getBiomeTint } from "./palette";
import { getSpriteSheet } from "./spriteSheet";
import type { SpriteSheetGenerator } from "./spriteSheet";

interface PooledAgent {
  sprite: Sprite;
  highlight: Graphics;
  energyBar: Graphics;
  countLabel: Text;
  inUse: boolean;
}

interface TrackedAgent {
  prevX: number;
  prevY: number;
  direction: Direction;
}

const INITIAL_POOL_SIZE = 500;
const DIRECTION_DEAD_ZONE = 0.5;

const COUNT_LABEL_STYLE = new TextStyle({
  fontFamily: "Rajdhani, monospace",
  fontSize: 13,
  fontWeight: "700",
  fill: "#ffffff",
  stroke: { color: "#000000", width: 3 },
});

export class CreatureRenderer {
  private container: Container;
  private pool: PooledAgent[] = [];
  private activeCount = 0;
  private worldSize: number;
  private speciesMap = new Map<number, SpeciesInfo>();
  private tracked = new Map<string, TrackedAgent>();
  private spriteSheet: SpriteSheetGenerator;

  // Biome data for tinting — set externally
  private biomeMap = new Map<number, BiomeData>();
  private biomeCenters: Map<number, [number, number]> = new Map();

  constructor(container: Container, worldSize: number) {
    this.container = container;
    this.worldSize = worldSize;
    this.spriteSheet = getSpriteSheet();
    this.growPool(INITIAL_POOL_SIZE);
  }

  private growPool(count: number) {
    for (let i = 0; i < count; i++) {
      const sprite = new Sprite();
      sprite.anchor.set(0.5);
      sprite.visible = false;

      const highlight = new Graphics();
      highlight.visible = false;

      const energyBar = new Graphics();
      energyBar.visible = false;

      const countLabel = new Text({ text: "", style: COUNT_LABEL_STYLE });
      countLabel.anchor.set(0.5);
      countLabel.visible = false;

      this.container.addChild(sprite);
      this.container.addChild(highlight);
      this.container.addChild(energyBar);
      this.container.addChild(countLabel);

      this.pool.push({ sprite, highlight, energyBar, countLabel, inUse: false });
    }
  }

  private acquire(): PooledAgent {
    for (let i = this.activeCount; i < this.pool.length; i++) {
      if (!this.pool[i].inUse) {
        this.pool[i].inUse = true;
        return this.pool[i];
      }
    }
    this.growPool(100);
    const entry = this.pool[this.pool.length - 1];
    entry.inUse = true;
    return entry;
  }

  /** Update biome data for tinting. Called from PixiApp when biomes change. */
  updateBiomes(biomes: BiomeData[], centers: Map<number, [number, number]>) {
    this.biomeMap = new Map(biomes.map(b => [b.lid, b]));
    this.biomeCenters = centers;
  }

  /** Derive facing direction from position delta. */
  private deriveDirection(uid: string, x: number, y: number): Direction {
    const isCluster = uid.startsWith("cluster_");
    if (isCluster) return Direction.DOWN;

    const prev = this.tracked.get(uid);
    if (!prev) {
      this.tracked.set(uid, { prevX: x, prevY: y, direction: Direction.DOWN });
      return Direction.DOWN;
    }

    const dx = x - prev.prevX;
    const dy = y - prev.prevY;
    const dist = Math.sqrt(dx * dx + dy * dy);

    prev.prevX = x;
    prev.prevY = y;

    if (dist < DIRECTION_DEAD_ZONE) return prev.direction;

    let newDir: Direction;
    if (Math.abs(dx) > Math.abs(dy)) {
      newDir = dx > 0 ? Direction.RIGHT : Direction.LEFT;
    } else {
      newDir = dy > 0 ? Direction.DOWN : Direction.UP;
    }
    prev.direction = newDir;
    return newDir;
  }

  /** Map Direction → DrawnDirection (LEFT flips RIGHT). */
  private toDrawnDirection(dir: Direction): { drawn: DrawnDirection; flip: boolean } {
    if (dir === Direction.LEFT) {
      return { drawn: Direction.RIGHT as DrawnDirection, flip: true };
    }
    return { drawn: dir as DrawnDirection, flip: false };
  }

  /** Find the nearest biome to a world position and return its tint. */
  private getBiomeTintAt(x: number, y: number): number {
    if (this.biomeMap.size === 0) return 0xffffff;

    let bestLid = -1;
    let bestDist = Infinity;
    for (const [lid, [cx, cy]] of this.biomeCenters) {
      const dx = x - cx;
      const dy = y - cy;
      const d = dx * dx + dy * dy;
      if (d < bestDist) {
        bestDist = d;
        bestLid = lid;
      }
    }

    if (bestLid < 0) return 0xffffff;
    const biome = this.biomeMap.get(bestLid);
    return biome ? getBiomeTint(biome.name) : 0xffffff;
  }

  update(agents: AgentSnapshot[], species: SpeciesInfo[], hoveredUid?: string | null) {
    this.speciesMap = new Map(species.map((s) => [s.sid, s]));

    // Release all
    for (let i = 0; i < this.pool.length; i++) {
      const p = this.pool[i];
      p.inUse = false;
      p.sprite.visible = false;
      p.highlight.visible = false;
      p.energyBar.visible = false;
      p.countLabel.visible = false;
    }
    this.activeCount = 0;

    // Prune tracked agents not in current set
    const currentUids = new Set(agents.filter(a => !a.dead).map(a => a.uid));
    for (const uid of this.tracked.keys()) {
      if (!currentUids.has(uid)) this.tracked.delete(uid);
    }

    for (const agent of agents) {
      if (agent.dead) continue;

      const entry = this.acquire();
      this.activeCount++;

      const sp = this.speciesMap.get(agent.species_sid);
      const bodyPlan = (sp?.plan || "grazer") as import("./types").BodyPlan;

      // Direction
      const direction = this.deriveDirection(agent.uid, agent.x, agent.y);
      const { drawn, flip } = this.toDrawnDirection(direction);

      // Visual state
      const visualState = behaviorToVisualState(agent.state, agent.energy);

      // Store metadata for animation updates
      const spriteAny = entry.sprite as any;
      spriteAny.__bodyPlan = bodyPlan;
      spriteAny.__drawnDir = drawn;
      spriteAny.__visualState = visualState;
      spriteAny.__aid = agent.uid;

      // Get texture for current frame (frame 0, animation will update)
      const frameIndex = spriteAny.__animFrame ?? 0;
      entry.sprite.texture = this.spriteSheet.getTexture(bodyPlan, visualState, drawn, frameIndex % ANIM_FRAMES);

      // Scale: base + count (state scale is baked into textures via drawBodies canvas transform)
      const baseScale = 0.4 + Math.sqrt(agent.count || 1) * 0.15;
      entry.sprite.scale.set(flip ? -baseScale : baseScale, baseScale);
      entry.sprite.x = agent.x;
      entry.sprite.y = agent.y;
      entry.sprite.visible = true;

      // Biome tinting
      entry.sprite.tint = this.getBiomeTintAt(agent.x, agent.y);

      // Hover highlight
      if (agent.uid === hoveredUid) {
        entry.highlight.visible = true;
        entry.highlight.clear();
        const hlR = SPRITE_SIZE * Math.abs(baseScale) * 0.55;
        entry.highlight.circle(agent.x, agent.y, hlR);
        entry.highlight.stroke({ color: 0xffffff, alpha: 0.7, width: 2 });
      }

      // Energy bar (low energy)
      if (agent.energy < 20) {
        entry.energyBar.visible = true;
        entry.energyBar.clear();
        const barW = 10;
        const barH = 2;
        const barX = agent.x - barW / 2;
        const barY = agent.y - SPRITE_SIZE * Math.abs(baseScale) * 0.4 - 4;

        entry.energyBar.rect(barX, barY, barW, barH);
        entry.energyBar.fill({ color: 0x333333, alpha: 0.5 });
        entry.energyBar.rect(barX, barY, (agent.energy / 100) * barW, barH);
        entry.energyBar.fill({ color: 0xffff00, alpha: 0.8 });
      }

      // Count badge for clusters
      if (agent.count > 1) {
        entry.countLabel.visible = true;
        entry.countLabel.text = String(agent.count);
        entry.countLabel.x = agent.x;
        entry.countLabel.y = agent.y - SPRITE_SIZE * Math.abs(baseScale) * 0.5 - 6;
      }
    }
  }

  updateAnimations(now: number) {
    const phase = (now % ANIM_PERIOD) / ANIM_PERIOD;
    const frameIndex = Math.floor(phase * ANIM_FRAMES);

    for (let i = 0; i < this.pool.length; i++) {
      const p = this.pool[i];
      if (!p.inUse) continue;

      const spriteAny = p.sprite as any;
      const prev = spriteAny.__animFrame;
      if (prev !== frameIndex) {
        spriteAny.__animFrame = frameIndex;
        const bodyPlan = spriteAny.__bodyPlan || "grazer";
        const drawnDir = spriteAny.__drawnDir ?? Direction.DOWN;
        const visualState = spriteAny.__visualState ?? VisualState.IDLE;
        p.sprite.texture = this.spriteSheet.getTexture(bodyPlan, visualState, drawnDir, frameIndex);
      }
    }
  }

  destroy() {
    // Sprite sheet is a singleton, destroyed separately
    this.tracked.clear();
  }
}
