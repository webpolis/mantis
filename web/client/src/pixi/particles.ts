/**
 * Lightweight custom particle system with sprite pooling.
 * Presets for birth, death, hunt, catastrophes, epoch transitions.
 */
import { Container, Graphics, Sprite, Texture } from "pixi.js";

interface Particle {
  sprite: Sprite;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  fadeStart: number; // fraction of life when fade begins
  scaleStart: number;
  scaleEnd: number;
  inUse: boolean;
}

const MAX_PARTICLES = 1500;

// Generate small circle textures for particles
function makeCircleTexture(color: number, size: number): Texture {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const r = (color >> 16) & 0xff;
  const g = (color >> 8) & 0xff;
  const b = color & 0xff;
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 1)`);
  grad.addColorStop(0.6, `rgba(${r}, ${g}, ${b}, 0.6)`);
  grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
  ctx.fill();
  return Texture.from(canvas);
}

// Pre-made particle textures
const PARTICLE_TEXTURES: Record<string, Texture> = {};

function getParticleTexture(name: string, color: number, size: number = 8): Texture {
  const key = `${name}_${color}_${size}`;
  if (!PARTICLE_TEXTURES[key]) {
    PARTICLE_TEXTURES[key] = makeCircleTexture(color, size);
  }
  return PARTICLE_TEXTURES[key];
}

export class ParticleSystem {
  private container: Container;
  private pool: Particle[] = [];
  private worldSize: number;

  constructor(container: Container, worldSize: number) {
    this.container = container;
    this.worldSize = worldSize;
    this.initPool();
  }

  private initPool() {
    const defaultTex = getParticleTexture("default", 0xffffff, 8);
    for (let i = 0; i < MAX_PARTICLES; i++) {
      const sprite = new Sprite(defaultTex);
      sprite.anchor.set(0.5);
      sprite.visible = false;
      this.container.addChild(sprite);
      this.pool.push({
        sprite,
        vx: 0, vy: 0,
        life: 0, maxLife: 1,
        fadeStart: 0.5,
        scaleStart: 1, scaleEnd: 0,
        inUse: false,
      });
    }
  }

  private emit(
    x: number, y: number,
    count: number,
    texture: Texture,
    config: {
      vxRange: [number, number];
      vyRange: [number, number];
      lifeRange: [number, number];
      scaleStart?: number;
      scaleEnd?: number;
      fadeStart?: number;
      alpha?: number;
    }
  ) {
    let spawned = 0;
    for (const p of this.pool) {
      if (spawned >= count) break;
      if (p.inUse) continue;

      p.inUse = true;
      p.sprite.texture = texture;
      p.sprite.visible = true;
      p.sprite.x = x;
      p.sprite.y = y;
      p.sprite.alpha = config.alpha ?? 1;
      p.vx = config.vxRange[0] + Math.random() * (config.vxRange[1] - config.vxRange[0]);
      p.vy = config.vyRange[0] + Math.random() * (config.vyRange[1] - config.vyRange[0]);
      p.maxLife = config.lifeRange[0] + Math.random() * (config.lifeRange[1] - config.lifeRange[0]);
      p.life = 0;
      p.scaleStart = config.scaleStart ?? 1;
      p.scaleEnd = config.scaleEnd ?? 0;
      p.fadeStart = config.fadeStart ?? 0.5;
      p.sprite.scale.set(p.scaleStart);

      spawned++;
    }
  }

  update(dt: number) {
    const dtSec = dt / 1000;
    for (const p of this.pool) {
      if (!p.inUse) continue;
      p.life += dtSec;
      if (p.life >= p.maxLife || p.maxLife <= 0) {
        p.inUse = false;
        p.sprite.visible = false;
        continue;
      }

      const t = p.life / p.maxLife;
      p.sprite.x += p.vx * dtSec;
      p.sprite.y += p.vy * dtSec;

      // Scale interpolation
      const scale = p.scaleStart + (p.scaleEnd - p.scaleStart) * t;
      p.sprite.scale.set(Math.max(0.01, scale));

      // Fade
      if (t > p.fadeStart) {
        const fadeDenom = 1 - p.fadeStart;
        const fadeT = fadeDenom > 0 ? (t - p.fadeStart) / fadeDenom : 1;
        p.sprite.alpha = Math.max(0, 1 - fadeT);
      }
    }
  }

  // --- Presets ---

  emitBirth(x: number, y: number) {
    const tex = getParticleTexture("birth", 0x66ff88, 8);
    this.emit(x, y, 15, tex, {
      vxRange: [-30, 30],
      vyRange: [-60, -20],
      lifeRange: [0.6, 1.2],
      scaleStart: 0.8,
      scaleEnd: 0.1,
      fadeStart: 0.4,
    });
    // Gold sparkles
    const goldTex = getParticleTexture("birthGold", 0xffcc44, 6);
    this.emit(x, y, 8, goldTex, {
      vxRange: [-40, 40],
      vyRange: [-50, -15],
      lifeRange: [0.5, 1.0],
      scaleStart: 0.6,
      scaleEnd: 0.05,
      fadeStart: 0.3,
    });
  }

  emitDeath(x: number, y: number) {
    const tex = getParticleTexture("death", 0x888888, 6);
    this.emit(x, y, 10, tex, {
      vxRange: [-15, 15],
      vyRange: [10, 40],
      lifeRange: [0.8, 1.5],
      scaleStart: 0.6,
      scaleEnd: 0.1,
      fadeStart: 0.3,
      alpha: 0.6,
    });
  }

  emitHuntStreak(fromX: number, fromY: number, toX: number, toY: number) {
    const tex = getParticleTexture("hunt", 0xff3333, 6);
    const dx = toX - fromX;
    const dy = toY - fromY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 1) return;
    const count = Math.min(8, Math.ceil(dist / 15));
    for (let i = 0; i < count; i++) {
      const t = i / count;
      const px = fromX + dx * t;
      const py = fromY + dy * t;
      this.emit(px, py, 1, tex, {
        vxRange: [dx * 0.5, dx * 0.8],
        vyRange: [dy * 0.5, dy * 0.8],
        lifeRange: [0.2, 0.5],
        scaleStart: 0.5,
        scaleEnd: 0.1,
        fadeStart: 0.2,
      });
    }
  }

  emitSpeciation(x: number, y: number) {
    const goldTex = getParticleTexture("speciation", 0xffcc33, 10);
    this.emit(x, y, 20, goldTex, {
      vxRange: [-50, 50],
      vyRange: [-80, -20],
      lifeRange: [0.8, 1.6],
      scaleStart: 1.0,
      scaleEnd: 0.1,
      fadeStart: 0.4,
    });
  }

  emitCatastrophe(kind: string, screenW: number, screenH: number) {
    // Catastrophe particles are in world-space, spread across the visible area
    const cx = this.worldSize / 2;
    const cy = this.worldSize / 2;
    const spread = this.worldSize * 0.4;

    switch (kind) {
      case "meteor_impact": {
        const tex = getParticleTexture("meteor", 0xff4400, 12);
        const whiteTex = getParticleTexture("meteorWhite", 0xffeecc, 10);
        // Massive radial burst
        for (let i = 0; i < 80; i++) {
          const angle = Math.random() * Math.PI * 2;
          const speed = 40 + Math.random() * 120;
          const x = cx + (Math.random() - 0.5) * 100;
          const y = cy + (Math.random() - 0.5) * 100;
          this.emit(x, y, 1, i < 50 ? tex : whiteTex, {
            vxRange: [Math.cos(angle) * speed, Math.cos(angle) * speed],
            vyRange: [Math.sin(angle) * speed, Math.sin(angle) * speed],
            lifeRange: [1.0, 2.5],
            scaleStart: 1.2,
            scaleEnd: 0.1,
            fadeStart: 0.3,
          });
        }
        break;
      }
      case "ice_age": {
        const tex = getParticleTexture("frost", 0xaaddff, 8);
        // Frost particles drifting inward from edges
        for (let i = 0; i < 60; i++) {
          const side = Math.floor(Math.random() * 4);
          let x: number, y: number, vx: number, vy: number;
          if (side === 0) { x = 0; y = Math.random() * this.worldSize; vx = 20 + Math.random() * 30; vy = (Math.random() - 0.5) * 20; }
          else if (side === 1) { x = this.worldSize; y = Math.random() * this.worldSize; vx = -(20 + Math.random() * 30); vy = (Math.random() - 0.5) * 20; }
          else if (side === 2) { x = Math.random() * this.worldSize; y = 0; vx = (Math.random() - 0.5) * 20; vy = 20 + Math.random() * 30; }
          else { x = Math.random() * this.worldSize; y = this.worldSize; vx = (Math.random() - 0.5) * 20; vy = -(20 + Math.random() * 30); }
          this.emit(x, y, 1, tex, {
            vxRange: [vx, vx],
            vyRange: [vy, vy],
            lifeRange: [2.0, 4.0],
            scaleStart: 0.8,
            scaleEnd: 0.2,
            fadeStart: 0.5,
            alpha: 0.7,
          });
        }
        break;
      }
      case "volcanic_winter": {
        const ashTex = getParticleTexture("ash", 0x555555, 6);
        const glowTex = getParticleTexture("lavaGlow", 0xff6600, 10);
        // Falling ash
        for (let i = 0; i < 50; i++) {
          const x = Math.random() * this.worldSize;
          this.emit(x, 0, 1, ashTex, {
            vxRange: [-10, 10],
            vyRange: [15, 40],
            lifeRange: [2.0, 4.0],
            scaleStart: 0.5,
            scaleEnd: 0.2,
            fadeStart: 0.6,
            alpha: 0.5,
          });
        }
        // Ground glow spots
        for (let i = 0; i < 15; i++) {
          const x = Math.random() * this.worldSize;
          const y = Math.random() * this.worldSize;
          this.emit(x, y, 1, glowTex, {
            vxRange: [0, 0],
            vyRange: [-5, 5],
            lifeRange: [1.5, 3.0],
            scaleStart: 1.5,
            scaleEnd: 0.3,
            fadeStart: 0.3,
            alpha: 0.4,
          });
        }
        break;
      }
      case "tsunami": {
        const tex = getParticleTexture("wave", 0x88ccff, 10);
        // Horizontal sweep
        for (let i = 0; i < 40; i++) {
          const y = Math.random() * this.worldSize;
          this.emit(0, y, 1, tex, {
            vxRange: [60, 120],
            vyRange: [-10, 10],
            lifeRange: [1.5, 3.0],
            scaleStart: 1.0,
            scaleEnd: 0.3,
            fadeStart: 0.4,
            alpha: 0.6,
          });
        }
        break;
      }
      case "drought": {
        const tex = getParticleTexture("heat", 0xffaa44, 6);
        // Rising shimmer
        for (let i = 0; i < 30; i++) {
          const x = Math.random() * this.worldSize;
          const y = this.worldSize * 0.6 + Math.random() * this.worldSize * 0.4;
          this.emit(x, y, 1, tex, {
            vxRange: [-5, 5],
            vyRange: [-20, -10],
            lifeRange: [1.5, 3.0],
            scaleStart: 0.4,
            scaleEnd: 0.1,
            fadeStart: 0.4,
            alpha: 0.3,
          });
        }
        break;
      }
      case "plague": {
        const tex = getParticleTexture("plague", 0xaacc33, 8);
        // Pulsing random spots
        for (let i = 0; i < 25; i++) {
          const x = Math.random() * this.worldSize;
          const y = Math.random() * this.worldSize;
          this.emit(x, y, 1, tex, {
            vxRange: [-5, 5],
            vyRange: [-5, 5],
            lifeRange: [1.0, 2.5],
            scaleStart: 0.8,
            scaleEnd: 1.2,
            fadeStart: 0.5,
            alpha: 0.4,
          });
        }
        break;
      }
      default: {
        // Generic catastrophe
        const tex = getParticleTexture("generic", 0xff6644, 8);
        for (let i = 0; i < 30; i++) {
          const x = cx + (Math.random() - 0.5) * spread;
          const y = cy + (Math.random() - 0.5) * spread;
          const angle = Math.random() * Math.PI * 2;
          const speed = 20 + Math.random() * 60;
          this.emit(x, y, 1, tex, {
            vxRange: [Math.cos(angle) * speed, Math.cos(angle) * speed],
            vyRange: [Math.sin(angle) * speed, Math.sin(angle) * speed],
            lifeRange: [1.0, 2.0],
            scaleStart: 0.8,
            scaleEnd: 0.1,
            fadeStart: 0.4,
          });
        }
      }
    }
  }

  emitEpochTransition(worldCenterX: number, worldCenterY: number) {
    const tex = getParticleTexture("epoch", 0xffcc33, 10);
    for (let i = 0; i < 100; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 30 + Math.random() * 80;
      const x = worldCenterX + (Math.random() - 0.5) * 200;
      const y = worldCenterY + (Math.random() - 0.5) * 200;
      this.emit(x, y, 1, tex, {
        vxRange: [Math.cos(angle) * speed, Math.cos(angle) * speed],
        vyRange: [Math.sin(angle) * speed, Math.sin(angle) * speed],
        lifeRange: [1.0, 2.5],
        scaleStart: 1.0,
        scaleEnd: 0.1,
        fadeStart: 0.3,
      });
    }
  }

  destroy() {
    for (const p of this.pool) {
      p.sprite.destroy();
    }
    this.pool = [];
    for (const key in PARTICLE_TEXTURES) {
      PARTICLE_TEXTURES[key].destroy(true);
      delete PARTICLE_TEXTURES[key];
    }
  }
}
