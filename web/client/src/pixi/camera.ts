/**
 * Smooth lerp-based camera with zoom, pan, screen shake, and minimap support.
 */

export interface CameraState {
  x: number;       // World-space center X
  y: number;       // World-space center Y
  zoom: number;    // Current zoom level
  targetX: number;
  targetY: number;
  targetZoom: number;
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 20;
const LERP_FACTOR = 0.12;
const ZOOM_SPEED = 0.001;

export class Camera {
  state: CameraState;
  private shakeIntensity = 0;
  private shakeDuration = 0;
  private shakeElapsed = 0;
  private shakeOffsetX = 0;
  private shakeOffsetY = 0;
  private worldSize: number;
  private screenWidth: number;
  private screenHeight: number;

  constructor(worldSize: number, screenWidth: number, screenHeight: number) {
    this.worldSize = worldSize;
    this.screenWidth = screenWidth;
    this.screenHeight = screenHeight;
    this.state = {
      x: worldSize / 2,
      y: worldSize / 2,
      zoom: 1,
      targetX: worldSize / 2,
      targetY: worldSize / 2,
      targetZoom: 1,
    };
  }

  resize(screenWidth: number, screenHeight: number) {
    this.screenWidth = screenWidth;
    this.screenHeight = screenHeight;
  }

  zoomAt(screenX: number, screenY: number, delta: number) {
    const worldBefore = this.screenToWorld(screenX, screenY);
    const oldZoom = this.state.targetZoom;
    const zoomDelta = -delta * ZOOM_SPEED * oldZoom;
    this.state.targetZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, oldZoom + zoomDelta));

    // Keep the world point under the cursor fixed
    const worldAfter = this.screenToWorldAtZoom(screenX, screenY, this.state.targetZoom, this.state.targetX, this.state.targetY);
    this.state.targetX += worldBefore.x - worldAfter.x;
    this.state.targetY += worldBefore.y - worldAfter.y;
    this.clampTarget();
  }

  panBy(dx: number, dy: number) {
    if (this.state.targetZoom <= 1) return;
    this.state.targetX -= dx / this.state.zoom;
    this.state.targetY -= dy / this.state.zoom;
    this.clampTarget();
  }

  reset() {
    this.state.targetX = this.worldSize / 2;
    this.state.targetY = this.worldSize / 2;
    this.state.targetZoom = 1;
  }

  shake(intensity: number, duration: number) {
    this.shakeIntensity = intensity;
    this.shakeDuration = duration;
    this.shakeElapsed = 0;
  }

  /** Call every frame with dt in ms. */
  update(dt: number) {
    // Lerp toward targets
    this.state.x += (this.state.targetX - this.state.x) * LERP_FACTOR;
    this.state.y += (this.state.targetY - this.state.y) * LERP_FACTOR;
    this.state.zoom += (this.state.targetZoom - this.state.zoom) * LERP_FACTOR;

    // Snap when close
    if (Math.abs(this.state.zoom - this.state.targetZoom) < 0.001) {
      this.state.zoom = this.state.targetZoom;
    }
    if (Math.abs(this.state.x - this.state.targetX) < 0.01) this.state.x = this.state.targetX;
    if (Math.abs(this.state.y - this.state.targetY) < 0.01) this.state.y = this.state.targetY;

    // Screen shake
    if (this.shakeDuration > 0 && this.shakeElapsed < this.shakeDuration) {
      this.shakeElapsed += dt;
      const decay = Math.max(0, 1 - this.shakeElapsed / this.shakeDuration);
      const intensity = this.shakeIntensity * decay;
      this.shakeOffsetX = (Math.random() - 0.5) * 2 * intensity;
      this.shakeOffsetY = (Math.random() - 0.5) * 2 * intensity;
    } else {
      this.shakeOffsetX = 0;
      this.shakeOffsetY = 0;
    }
  }

  /** Get the transform to apply to the world container. */
  getTransform(): { x: number; y: number; scale: number } {
    const scale = Math.min(this.screenWidth, this.screenHeight) / this.worldSize * this.state.zoom;
    const x = this.screenWidth / 2 - this.state.x * scale + this.shakeOffsetX;
    const y = this.screenHeight / 2 - this.state.y * scale + this.shakeOffsetY;
    return { x, y, scale };
  }

  screenToWorld(screenX: number, screenY: number): { x: number; y: number } {
    return this.screenToWorldAtZoom(screenX, screenY, this.state.zoom, this.state.x, this.state.y);
  }

  worldToScreen(worldX: number, worldY: number): { x: number; y: number } {
    const t = this.getTransform();
    return {
      x: worldX * t.scale + t.x,
      y: worldY * t.scale + t.y,
    };
  }

  private screenToWorldAtZoom(screenX: number, screenY: number, zoom: number, cx: number, cy: number): { x: number; y: number } {
    const scale = Math.min(this.screenWidth, this.screenHeight) / this.worldSize * zoom;
    const ox = this.screenWidth / 2 - cx * scale;
    const oy = this.screenHeight / 2 - cy * scale;
    return {
      x: (screenX - ox) / scale,
      y: (screenY - oy) / scale,
    };
  }

  private clampTarget() {
    const halfView = this.worldSize / (2 * this.state.targetZoom);
    this.state.targetX = Math.max(halfView, Math.min(this.worldSize - halfView, this.state.targetX));
    this.state.targetY = Math.max(halfView, Math.min(this.worldSize - halfView, this.state.targetY));

    if (this.state.targetZoom <= 1) {
      this.state.targetX = this.worldSize / 2;
      this.state.targetY = this.worldSize / 2;
    }
  }

  get isZoomed(): boolean {
    return this.state.zoom > 1.05;
  }
}
