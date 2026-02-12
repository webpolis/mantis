/**
 * Core PixiJS orchestration: layers, lifecycle, update methods.
 * Manual integration with React via ref — no @pixi/react.
 */
import { Application, Container, Graphics, Sprite, Texture, RenderTexture } from "pixi.js";
import { Camera } from "./camera";
import { BiomeRenderer, biomeCenters } from "./biomes";
import { CreatureRenderer, destroySpriteSheet } from "./creatures";
import { ParticleSystem } from "./particles";
import type { AgentSnapshot, SpeciesInfo, BiomeData, SimulationEvent } from "../types/simulation";

const EPOCH_NAMES: Record<number, string> = {
  1: "Primordial",
  2: "Cambrian",
  3: "Ecosystem",
  4: "Intelligence",
};

export interface HUDData {
  tick: number;
  epoch: number;
  agentCount: number;
  speciesCount: number;
}

export class PixiApp {
  app!: Application;
  camera!: Camera;

  // Layers (z-order bottom to top)
  private worldContainer!: Container;
  private biomeLayer!: Container;
  private vegetationLayer!: Container;
  private gridLayer!: Container;
  private agentLayer!: Container;
  private particleLayer!: Container;

  private biomeRenderer!: BiomeRenderer;
  private creatureRenderer!: CreatureRenderer;
  particleSystem!: ParticleSystem;

  private worldSize = 1000;
  private lastTime = 0;
  private gridGraphics!: Graphics;
  private minimapTexture!: RenderTexture;
  private minimapSprite!: Sprite;
  private minimapSize = 150;
  private minimapBorder!: Graphics;
  private minimapViewport!: Graphics;
  private minimapDragging = false;
  private resizeObserver?: ResizeObserver;

  // Track previous agent set for birth/death detection
  private prevAgentIds = new Set<string>();

  private initialized = false;
  private destroyed = false;
  private initPromise: Promise<void> | null = null;

  async init(container: HTMLElement, worldSize: number = 1000) {
    this.initPromise = this._init(container, worldSize);
    return this.initPromise;
  }

  private async _init(container: HTMLElement, worldSize: number) {
    this.worldSize = worldSize;
    this.app = new Application();
    await this.app.init({
      resizeTo: container,
      backgroundColor: 0x0a0a14,
      antialias: true,
      resolution: Math.min(window.devicePixelRatio, 2),
      autoDensity: true,
    });

    // If destroy() was called while awaiting init, bail out
    if (this.destroyed) return;

    container.appendChild(this.app.canvas);

    const w = this.app.screen.width;
    const h = this.app.screen.height;

    this.camera = new Camera(worldSize, w, h);

    // Layers
    this.worldContainer = new Container();
    this.app.stage.addChild(this.worldContainer);

    this.biomeLayer = new Container();
    this.vegetationLayer = new Container();
    this.gridLayer = new Container();
    this.agentLayer = new Container();
    this.particleLayer = new Container();

    this.worldContainer.addChild(this.biomeLayer);
    this.worldContainer.addChild(this.vegetationLayer);
    this.worldContainer.addChild(this.gridLayer);
    this.worldContainer.addChild(this.agentLayer);
    this.worldContainer.addChild(this.particleLayer);

    // Grid
    this.gridGraphics = new Graphics();
    this.gridLayer.addChild(this.gridGraphics);
    this.drawGrid();

    // Renderers
    this.biomeRenderer = new BiomeRenderer(this.biomeLayer, this.vegetationLayer, worldSize);
    this.creatureRenderer = new CreatureRenderer(this.agentLayer, worldSize);
    this.particleSystem = new ParticleSystem(this.particleLayer, worldSize);

    // Minimap
    this.minimapTexture = RenderTexture.create({ width: this.minimapSize, height: this.minimapSize });
    this.minimapSprite = new Sprite(this.minimapTexture);
    this.minimapSprite.x = 10;
    this.minimapSprite.y = h - this.minimapSize - 10;
    this.app.stage.addChild(this.minimapSprite);

    this.minimapBorder = new Graphics();
    this.minimapBorder.rect(
      this.minimapSprite.x - 1,
      this.minimapSprite.y - 1,
      this.minimapSize + 2,
      this.minimapSize + 2,
    );
    this.minimapBorder.stroke({ color: 0xffffff, alpha: 0.15, width: 1 });
    this.app.stage.addChild(this.minimapBorder);

    this.minimapViewport = new Graphics();
    this.app.stage.addChild(this.minimapViewport);

    // Tick loop
    this.app.ticker.add(() => {
      if (this.destroyed) return;
      const now = performance.now();
      const dt = now - this.lastTime;
      this.lastTime = now;

      this.camera.update(dt);
      const t = this.camera.getTransform();
      this.worldContainer.x = t.x;
      this.worldContainer.y = t.y;
      this.worldContainer.scale.set(t.scale);

      this.creatureRenderer.updateAnimations(now);
      this.biomeRenderer.updateAnimations(now);
      this.particleSystem.update(dt);
    });

    // Handle resize
    const observer = new ResizeObserver(() => {
      if (this.destroyed) return;
      const w = this.app.screen.width;
      const h = this.app.screen.height;
      this.camera.resize(w, h);
      this.minimapSprite.y = h - this.minimapSize - 10;
      this.minimapBorder.clear();
      this.minimapBorder.rect(
        this.minimapSprite.x - 1,
        this.minimapSprite.y - 1,
        this.minimapSize + 2,
        this.minimapSize + 2,
      );
      this.minimapBorder.stroke({ color: 0xffffff, alpha: 0.15, width: 1 });
    });
    observer.observe(container);
    this.resizeObserver = observer;

    this.lastTime = performance.now();
    this.initialized = true;
  }

  updateBiomes(biomes: BiomeData[]) {
    if (!this.initialized || this.destroyed) return;
    this.biomeRenderer.update(biomes);
    // Pass biome data to creature renderer for per-agent tinting
    const centers = biomeCenters(biomes, this.worldSize);
    this.creatureRenderer.updateBiomes(biomes, centers);
  }

  updateAgents(agents: AgentSnapshot[], species: SpeciesInfo[], hoveredUid?: string | null, rawAgents?: AgentSnapshot[]) {
    if (!this.initialized || this.destroyed) return;

    // Detect births/deaths using raw (unclustered) agents to avoid
    // false positives from synthetic cluster UIDs changing every frame
    const trackingAgents = rawAgents ?? agents;
    const currentIds = new Set<string>();
    for (const a of trackingAgents) {
      if (!a.dead) currentIds.add(a.uid);
    }

    for (const uid of currentIds) {
      if (!this.prevAgentIds.has(uid)) {
        const a = trackingAgents.find((ag) => ag.uid === uid);
        if (a) this.particleSystem.emitBirth(a.x, a.y);
      }
    }
    for (const uid of this.prevAgentIds) {
      if (!currentIds.has(uid)) {
        // Agent died — find last known position from previous frame
        // We don't have previous position here, so skip death particles for missing agents
      }
    }
    this.prevAgentIds = currentIds;

    this.creatureRenderer.update(agents, species, hoveredUid);
    this.updateMinimap();
  }

  updateEvents(events: SimulationEvent[]) {
    if (!this.initialized || this.destroyed) return;
    for (const evt of events) {
      if (evt.event_type === "catastrophe") {
        const kind = evt.detail.split("|")[0];
        this.particleSystem.emitCatastrophe(kind, this.app.screen.width, this.app.screen.height);
        if (kind === "meteor_impact") {
          this.camera.shake(15, 1500);
        } else if (kind === "tsunami") {
          this.camera.shake(8, 1000);
        } else if (kind === "volcanic_winter") {
          this.camera.shake(6, 800);
        } else if (kind === "ice_age") {
          this.camera.shake(4, 600);
        }
      } else if (evt.event_type === "speciation") {
        // Emit gold sparkle at a random location
        const x = Math.random() * this.worldSize;
        const y = Math.random() * this.worldSize;
        this.particleSystem.emitSpeciation(x, y);
      }
    }
  }

  updateHUD(_data: HUDData) {
    // HUD is rendered as React overlays, not in PixiJS
  }

  private drawGrid() {
    const g = this.gridGraphics;
    g.clear();
    const step = 100;
    for (let x = 0; x <= this.worldSize; x += step) {
      g.moveTo(x, 0);
      g.lineTo(x, this.worldSize);
    }
    for (let y = 0; y <= this.worldSize; y += step) {
      g.moveTo(0, y);
      g.lineTo(this.worldSize, y);
    }
    g.stroke({ color: 0xffffff, alpha: 0.04, width: 0.5 });
  }

  private updateMinimap() {
    if (this.destroyed || !this.initialized || !this.app.renderer) return;

    // Render world container to minimap texture at reduced scale
    const prevX = this.worldContainer.x;
    const prevY = this.worldContainer.y;
    const prevScaleX = this.worldContainer.scale.x;
    const prevScaleY = this.worldContainer.scale.y;

    const minimapScale = this.minimapSize / this.worldSize;
    this.worldContainer.x = 0;
    this.worldContainer.y = 0;
    this.worldContainer.scale.set(minimapScale);

    this.app.renderer.render({
      container: this.worldContainer,
      target: this.minimapTexture,
      clear: true,
    });

    // Restore
    this.worldContainer.x = prevX;
    this.worldContainer.y = prevY;
    this.worldContainer.scale.set(prevScaleX, prevScaleY);

    // Draw viewport rectangle
    this.minimapViewport.clear();
    const cam = this.camera.state;
    const viewW = this.app.screen.width / (Math.min(this.app.screen.width, this.app.screen.height) / this.worldSize * cam.zoom);
    const viewH = this.app.screen.height / (Math.min(this.app.screen.width, this.app.screen.height) / this.worldSize * cam.zoom);
    const vx = (cam.x - viewW / 2) * minimapScale + this.minimapSprite.x;
    const vy = (cam.y - viewH / 2) * minimapScale + this.minimapSprite.y;
    const vw = viewW * minimapScale;
    const vh = viewH * minimapScale;
    this.minimapViewport.rect(vx, vy, vw, vh);
    this.minimapViewport.stroke({ color: 0xffffff, alpha: 0.5, width: 1 });
  }

  /** Check if screen coordinates fall within the minimap bounds. */
  hitTestMinimap(screenX: number, screenY: number): boolean {
    if (!this.initialized || this.destroyed) return false;
    const mx = this.minimapSprite.x;
    const my = this.minimapSprite.y;
    return screenX >= mx && screenX <= mx + this.minimapSize
        && screenY >= my && screenY <= my + this.minimapSize;
  }

  /** Convert screen coords on the minimap to world coords and pan camera there. */
  minimapPanTo(screenX: number, screenY: number) {
    if (!this.initialized || this.destroyed) return;
    const minimapScale = this.minimapSize / this.worldSize;
    const worldX = (screenX - this.minimapSprite.x) / minimapScale;
    const worldY = (screenY - this.minimapSprite.y) / minimapScale;
    this.camera.panTo(worldX, worldY);
  }

  get isMinimapDragging(): boolean {
    return this.minimapDragging;
  }

  set isMinimapDragging(v: boolean) {
    this.minimapDragging = v;
  }

  destroy() {
    this.destroyed = true;
    this.initialized = false;
    this.resizeObserver?.disconnect();

    // Wait for init to finish before tearing down PixiJS internals
    const teardown = () => {
      this.creatureRenderer?.destroy();
      this.biomeRenderer?.destroy();
      this.particleSystem?.destroy();
      destroySpriteSheet();
      this.minimapTexture?.destroy(true);
      this.app?.destroy(true, { children: true });
    };

    if (this.initPromise) {
      this.initPromise.then(teardown, teardown);
    } else {
      teardown();
    }
  }
}

export { EPOCH_NAMES };
