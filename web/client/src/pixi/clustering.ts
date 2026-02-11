/**
 * Grid-based spatial clustering per species — O(n), runs every frame.
 * Zoomed out: aggregated cluster dots with counts.
 * Zoomed in: cells shrink until agents are isolated → individual sprites.
 */
import type { AgentSnapshot } from "../types/simulation";

const BASE_CELL_SIZE = 120;
const MIN_CELL_SIZE = 5;

export function clusterAgents(agents: AgentSnapshot[], zoom: number): AgentSnapshot[] {
  const cellSize = Math.max(MIN_CELL_SIZE, BASE_CELL_SIZE / zoom);

  // Group by (species_sid, cellX, cellY)
  const buckets = new Map<string, AgentSnapshot[]>();
  const dead: AgentSnapshot[] = [];

  for (const a of agents) {
    if (a.dead) { dead.push(a); continue; }
    const cx = Math.floor(a.x / cellSize);
    const cy = Math.floor(a.y / cellSize);
    const key = `${a.species_sid}|${cx}|${cy}`;
    let bucket = buckets.get(key);
    if (!bucket) {
      bucket = [];
      buckets.set(key, bucket);
    }
    bucket.push(a);
  }

  const result: AgentSnapshot[] = [];

  for (const [key, bucket] of buckets) {
    if (bucket.length === 1) {
      result.push(bucket[0]);
      continue;
    }

    // Merge into cluster
    let sumX = 0, sumY = 0, sumEnergy = 0, sumAge = 0, totalCount = 0;
    const stateCounts = new Map<string, number>();

    for (const a of bucket) {
      const c = a.count || 1;
      sumX += a.x * c;
      sumY += a.y * c;
      sumEnergy += a.energy * c;
      sumAge += a.age * c;
      totalCount += c;
      stateCounts.set(a.state, (stateCounts.get(a.state) || 0) + c);
    }

    // Dominant state
    let dominantState = bucket[0].state;
    let maxStateCount = 0;
    for (const [state, count] of stateCounts) {
      if (count > maxStateCount) {
        maxStateCount = count;
        dominantState = state;
      }
    }

    const first = bucket[0];
    result.push({
      uid: `cluster_${key}`,
      aid: first.aid,
      species_sid: first.species_sid,
      x: sumX / totalCount,
      y: sumY / totalCount,
      energy: sumEnergy / totalCount,
      age: Math.round(sumAge / totalCount),
      state: dominantState,
      target_aid: null,
      dead: false,
      count: totalCount,
    });
  }

  // Append dead agents collected in first pass
  for (const a of dead) result.push(a);

  return result;
}
