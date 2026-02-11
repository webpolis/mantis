/**
 * Procedural biome texture generator with noise-based Voronoi regions.
 * 1024px noise textures per biome type + vegetation sprites + water animation.
 */
import { Container, Sprite, Texture, TilingSprite, Graphics } from "pixi.js";
import { SimplexNoise } from "./noise";
import type { BiomeData, VegetationPatchData } from "../types/simulation";

const TEX_SIZE = 1024;
const BIOME_SEED = 42;

interface BiomeColorProfile {
  base: [number, number, number];
  accent: [number, number, number];
  noiseScale: number;
  octaves: number;
}

const BIOME_PROFILES: Record<string, BiomeColorProfile> = {
  shallows:      { base: [40, 100, 160], accent: [80, 180, 220], noiseScale: 0.008, octaves: 3 },
  reef:          { base: [30, 130, 140], accent: [60, 200, 180], noiseScale: 0.012, octaves: 3 },
  deep_ocean:    { base: [10, 20, 60],   accent: [30, 60, 120],  noiseScale: 0.005, octaves: 3 },
  tidal_pools:   { base: [40, 90, 130],  accent: [90, 180, 210], noiseScale: 0.015, octaves: 3 },
  mangrove:      { base: [30, 60, 30],   accent: [60, 100, 50],  noiseScale: 0.01,  octaves: 3 },
  savanna:       { base: [140, 120, 50], accent: [100, 140, 60], noiseScale: 0.008, octaves: 3 },
  forest:        { base: [15, 50, 20],   accent: [40, 90, 35],   noiseScale: 0.01,  octaves: 3 },
  rainforest:    { base: [10, 55, 25],   accent: [30, 100, 50],  noiseScale: 0.012, octaves: 3 },
  desert:        { base: [170, 145, 80], accent: [200, 175, 110],noiseScale: 0.006, octaves: 3 },
  tundra:        { base: [140, 160, 180],accent: [190, 210, 230],noiseScale: 0.007, octaves: 3 },
  cave:          { base: [15, 12, 20],   accent: [30, 25, 40],   noiseScale: 0.015, octaves: 2 },
  volcanic_vent: { base: [25, 10, 5],    accent: [180, 60, 20],  noiseScale: 0.02,  octaves: 3 },
  meadow:        { base: [60, 120, 50],  accent: [100, 180, 80], noiseScale: 0.01,  octaves: 3 },
  swamp:         { base: [35, 50, 25],   accent: [60, 75, 35],   noiseScale: 0.012, octaves: 3 },
  alpine:        { base: [120, 140, 170],accent: [180, 195, 220],noiseScale: 0.008, octaves: 3 },
};

const WATER_BIOMES = new Set(["shallows", "reef", "deep_ocean", "tidal_pools"]);

// --- Seeded PRNG (same as useWebSocket) ---
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function biomeCenters(biomes: BiomeData[], size: number): Map<number, [number, number]> {
  const centers = new Map<number, [number, number]>();
  for (const b of biomes) {
    const rng = mulberry32(b.lid * 7919 + 31);
    const margin = size * 0.15;
    const x = margin + rng() * (size - 2 * margin);
    const y = margin + rng() * (size - 2 * margin);
    centers.set(b.lid, [x, y]);
  }
  return centers;
}

// --- Biome texture generation ---

function generateBiomeTexture(biomes: BiomeData[], worldSize: number): Texture | null {
  if (biomes.length === 0) return null;

  const canvas = document.createElement("canvas");
  canvas.width = TEX_SIZE;
  canvas.height = TEX_SIZE;
  const ctx = canvas.getContext("2d")!;

  const noise = new SimplexNoise(BIOME_SEED);
  const noise2 = new SimplexNoise(BIOME_SEED + 1000);
  const noise3 = new SimplexNoise(BIOME_SEED + 2000);

  const centers = biomeCenters(biomes, TEX_SIZE);
  const biomeMap = new Map(biomes.map((b) => [b.lid, b]));
  const centerList = Array.from(centers.entries());

  const imgData = ctx.createImageData(TEX_SIZE, TEX_SIZE);
  const data = imgData.data;

  for (let py = 0; py < TEX_SIZE; py++) {
    for (let px = 0; px < TEX_SIZE; px++) {
      // Find nearest Voronoi center
      let bestLid = centerList[0][0];
      let bestDist = Infinity;
      let secondDist = Infinity;
      for (const [lid, [cx, cy]] of centerList) {
        const dx = px - cx;
        const dy = py - cy;
        const d = dx * dx + dy * dy;
        if (d < bestDist) {
          secondDist = bestDist;
          bestDist = d;
          bestLid = lid;
        } else if (d < secondDist) {
          secondDist = d;
        }
      }

      const biome = biomeMap.get(bestLid);
      const biomeName = biome?.name || "meadow";
      const profile = BIOME_PROFILES[biomeName] || BIOME_PROFILES.meadow;

      // Multi-octave noise for texture variety
      const n1 = noise.fbm(px * profile.noiseScale, py * profile.noiseScale, profile.octaves);
      const n2 = noise2.fbm(px * profile.noiseScale * 2, py * profile.noiseScale * 2, 2);
      const n3 = noise3.fbm(px * profile.noiseScale * 4, py * profile.noiseScale * 4, 2);

      // Edge darkening near Voronoi borders
      const edgeFactor = Math.min(1, Math.sqrt(secondDist - bestDist) / 40);

      // Mix base and accent colors using noise
      const mix = n1 * 0.6 + n2 * 0.3 + n3 * 0.1;
      let r = profile.base[0] + (profile.accent[0] - profile.base[0]) * mix;
      let g = profile.base[1] + (profile.accent[1] - profile.base[1]) * mix;
      let b = profile.base[2] + (profile.accent[2] - profile.base[2]) * mix;

      // Special biome effects
      if (biomeName === "volcanic_vent") {
        // Glowing orange fissure cracks
        const crackNoise = noise3.noise2D(px * 0.03, py * 0.03);
        if (crackNoise > 0.55) {
          const glow = (crackNoise - 0.55) / 0.45;
          r = r + (220 - r) * glow;
          g = g + (80 - g) * glow * 0.5;
          b = b * (1 - glow);
        }
      } else if (biomeName === "cave") {
        // Rare luminescent cyan dots
        const dotNoise = noise3.noise2D(px * 0.05, py * 0.05);
        if (dotNoise > 0.7) {
          const glow = (dotNoise - 0.7) / 0.3;
          r = r + (40 - r) * glow;
          g = g + (200 - g) * glow;
          b = b + (180 - b) * glow;
        }
      } else if (biomeName === "meadow") {
        // Scattered flower dots
        const flowerNoise = noise3.noise2D(px * 0.06, py * 0.06);
        if (flowerNoise > 0.65) {
          const flowerType = noise2.noise2D(px * 0.1, py * 0.1);
          if (flowerType > 0.5) {
            r = Math.min(255, r + 80); g = Math.min(255, g + 40);
          } else {
            r = Math.min(255, r + 40); b = Math.min(255, b + 60);
          }
        }
      } else if (biomeName === "desert") {
        // Wind-line directional pattern
        const windNoise = noise.noise2D(px * 0.003, py * 0.02);
        const windFactor = (windNoise + 1) * 0.5;
        r = r + (profile.accent[0] - r) * windFactor * 0.3;
        g = g + (profile.accent[1] - g) * windFactor * 0.3;
      } else if (WATER_BIOMES.has(biomeName)) {
        // Caustic bright spots
        const caustic = noise2.noise2D(px * 0.02, py * 0.02);
        if (caustic > 0.5) {
          const intensity = (caustic - 0.5) * 2;
          r = Math.min(255, r + 30 * intensity);
          g = Math.min(255, g + 50 * intensity);
          b = Math.min(255, b + 40 * intensity);
        }
      } else if (biomeName === "tundra" || biomeName === "alpine") {
        // Crystalline sparkle pattern
        const sparkle = noise3.noise2D(px * 0.04, py * 0.04);
        if (sparkle > 0.6) {
          const intensity = (sparkle - 0.6) / 0.4;
          r = Math.min(255, r + 40 * intensity);
          g = Math.min(255, g + 40 * intensity);
          b = Math.min(255, b + 50 * intensity);
        }
      }

      // Apply edge darkening
      r *= 0.7 + 0.3 * edgeFactor;
      g *= 0.7 + 0.3 * edgeFactor;
      b *= 0.7 + 0.3 * edgeFactor;

      // Vegetation brightness modulation
      const vegFactor = biome ? 0.6 + 0.4 * Math.min(1, biome.vegetation) : 0.6;
      r *= vegFactor;
      g *= vegFactor;
      b *= vegFactor;

      const idx = (py * TEX_SIZE + px) * 4;
      data[idx] = Math.min(255, Math.max(0, r));
      data[idx + 1] = Math.min(255, Math.max(0, g));
      data[idx + 2] = Math.min(255, Math.max(0, b));
      data[idx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return Texture.from(canvas);
}

// --- Vegetation sprite types ---

interface VegSprite {
  sprite: Sprite;
  baseY: number;
  swayPhase: number;
}

const VEG_COLORS: Record<string, [number, number, number]> = {
  forest: [30, 80, 25],
  rainforest: [20, 90, 35],
  meadow: [60, 140, 50],
  savanna: [130, 120, 40],
  swamp: [40, 60, 30],
  mangrove: [35, 70, 30],
  desert: [120, 110, 50],
  tundra: [100, 120, 110],
  alpine: [80, 100, 90],
  cave: [30, 100, 80],
  volcanic_vent: [80, 40, 15],
};

function generateVegTexture(biomeName: string): Texture {
  const size = 24;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const c = VEG_COLORS[biomeName] || [60, 120, 50];

  if (biomeName === "desert") {
    // Cactus
    ctx.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    ctx.fillRect(size * 0.4, size * 0.2, size * 0.2, size * 0.7);
    ctx.fillRect(size * 0.2, size * 0.35, size * 0.2, size * 0.15);
    ctx.fillRect(size * 0.6, size * 0.25, size * 0.2, size * 0.15);
  } else if (biomeName === "cave") {
    // Mushroom
    ctx.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    ctx.beginPath();
    ctx.ellipse(size / 2, size * 0.35, size * 0.35, size * 0.25, 0, Math.PI, 0);
    ctx.fill();
    ctx.fillRect(size * 0.42, size * 0.35, size * 0.16, size * 0.45);
    // Bioluminescent spots
    ctx.fillStyle = `rgba(80, 220, 180, 0.6)`;
    ctx.beginPath();
    ctx.arc(size * 0.4, size * 0.3, size * 0.04, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(size * 0.6, size * 0.25, size * 0.03, 0, Math.PI * 2);
    ctx.fill();
  } else if (WATER_BIOMES.has(biomeName)) {
    // Coral/seaweed
    ctx.strokeStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    for (let i = 0; i < 3; i++) {
      const x = size * 0.3 + i * size * 0.2;
      ctx.beginPath();
      ctx.moveTo(x, size * 0.9);
      ctx.quadraticCurveTo(x + (i - 1) * 4, size * 0.5, x, size * 0.2);
      ctx.stroke();
    }
  } else if (biomeName === "forest" || biomeName === "rainforest") {
    // Tree
    ctx.fillStyle = `rgb(80, 60, 30)`;
    ctx.fillRect(size * 0.42, size * 0.5, size * 0.16, size * 0.4);
    ctx.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    ctx.beginPath();
    ctx.arc(size / 2, size * 0.35, size * 0.35, 0, Math.PI * 2);
    ctx.fill();
  } else {
    // Grass tufts
    ctx.strokeStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    ctx.lineWidth = 1.5;
    ctx.lineCap = "round";
    for (let i = 0; i < 5; i++) {
      const x = size * 0.2 + i * size * 0.15;
      ctx.beginPath();
      ctx.moveTo(x, size * 0.9);
      ctx.quadraticCurveTo(x + (i - 2) * 2, size * 0.5, x + (i - 2), size * 0.2 + i * size * 0.05);
      ctx.stroke();
    }
  }

  return Texture.from(canvas);
}

// --- Vegetation cache ---
const vegTextureCache = new Map<string, Texture>();

function getVegTexture(biomeName: string): Texture {
  let tex = vegTextureCache.get(biomeName);
  if (!tex) {
    tex = generateVegTexture(biomeName);
    vegTextureCache.set(biomeName, tex);
  }
  return tex;
}

// --- Biome Renderer ---

export class BiomeRenderer {
  private biomeContainer: Container;
  private vegContainer: Container;
  private worldSize: number;
  private biomeSprite: Sprite | null = null;
  private vegSprites: VegSprite[] = [];
  private biomeKey = "";
  private waterOverlay: Graphics | null = null;
  private waterPhase = 0;
  private waterBiomeCenters: Array<{ x: number; y: number; radius: number }> = [];

  constructor(biomeContainer: Container, vegContainer: Container, worldSize: number) {
    this.biomeContainer = biomeContainer;
    this.vegContainer = vegContainer;
    this.worldSize = worldSize;
  }

  update(biomes: BiomeData[]) {
    const key = biomes.map((b) => `${b.lid}:${b.name}:${b.vegetation.toFixed(2)}`).join(",");
    if (key === this.biomeKey) return;
    this.biomeKey = key;

    // Regenerate biome texture (destroy old texture to free GPU memory)
    if (this.biomeSprite) {
      const oldTex = this.biomeSprite.texture;
      this.biomeContainer.removeChild(this.biomeSprite);
      this.biomeSprite.destroy();
      oldTex?.destroy(true);
    }

    const tex = generateBiomeTexture(biomes, this.worldSize);
    if (tex) {
      this.biomeSprite = new Sprite(tex);
      this.biomeSprite.width = this.worldSize;
      this.biomeSprite.height = this.worldSize;
      this.biomeContainer.addChild(this.biomeSprite);
    }

    // Regenerate vegetation sprites
    for (const vs of this.vegSprites) {
      this.vegContainer.removeChild(vs.sprite);
      vs.sprite.destroy();
    }
    this.vegSprites = [];

    const biomeMap = new Map(biomes.map((b) => [b.lid, b]));

    // Identify water biome regions for overlay
    this.waterBiomeCenters = [];
    const centers = biomeCenters(biomes, this.worldSize);
    for (const b of biomes) {
      if (WATER_BIOMES.has(b.name)) {
        const c = centers.get(b.lid);
        if (c) this.waterBiomeCenters.push({ x: c[0], y: c[1], radius: this.worldSize * 0.2 });
      }
    }

    for (const biome of biomes) {
      const vegTex = getVegTexture(biome.name);
      for (const patch of biome.patches) {
        if (patch.density < 0.05) continue;
        // Place several vegetation sprites per patch
        const count = Math.ceil(patch.density * 5);
        const rng = mulberry32(patch.x * 1000 + patch.y);
        for (let i = 0; i < count; i++) {
          const angle = rng() * Math.PI * 2;
          const dist = rng() * patch.radius * 0.7;
          const sx = patch.x + Math.cos(angle) * dist;
          const sy = patch.y + Math.sin(angle) * dist;

          if (sx < 0 || sx > this.worldSize || sy < 0 || sy > this.worldSize) continue;

          const sprite = new Sprite(vegTex);
          sprite.anchor.set(0.5, 1);
          sprite.x = sx;
          sprite.y = sy;
          const scale = 0.8 + rng() * 0.6;
          sprite.scale.set(scale);
          sprite.alpha = 0.5 + patch.density * 0.5;
          this.vegContainer.addChild(sprite);

          this.vegSprites.push({
            sprite,
            baseY: sy,
            swayPhase: rng() * Math.PI * 2,
          });
        }
      }
    }

    // Water shimmer overlay
    if (this.waterOverlay) {
      this.biomeContainer.removeChild(this.waterOverlay);
      this.waterOverlay = null;
    }
    if (this.waterBiomeCenters.length > 0) {
      this.waterOverlay = new Graphics();
      this.biomeContainer.addChild(this.waterOverlay);
      this.updateWaterOverlay();
    }
  }

  updateWaterOverlay() {
    if (!this.waterOverlay || this.waterBiomeCenters.length === 0) return;
    this.waterPhase += 0.002;

    this.waterOverlay.clear();
    for (const wc of this.waterBiomeCenters) {
      const shimmerAlpha = 0.03 + Math.sin(this.waterPhase * 2 + wc.x * 0.01) * 0.02;
      this.waterOverlay.circle(wc.x, wc.y, wc.radius);
      this.waterOverlay.fill({ color: 0x88ccff, alpha: Math.max(0, shimmerAlpha) });
    }
  }

  /** Call in ticker for vegetation sway. */
  updateAnimations(now: number) {
    const time = now * 0.001;
    for (const vs of this.vegSprites) {
      vs.sprite.x += Math.sin(time * 1.5 + vs.swayPhase) * 0.03;
    }
    this.updateWaterOverlay();
  }

  destroy() {
    for (const vs of this.vegSprites) {
      vs.sprite.destroy();
    }
    this.vegSprites = [];
    for (const tex of vegTextureCache.values()) {
      tex.destroy(true);
    }
    vegTextureCache.clear();
    if (this.biomeSprite) {
      const tex = this.biomeSprite.texture;
      this.biomeSprite.destroy();
      tex?.destroy(true);
    }
    if (this.waterOverlay) this.waterOverlay.destroy();
  }
}
