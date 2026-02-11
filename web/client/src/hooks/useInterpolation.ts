import { useRef, useCallback } from "react";
import type { AgentSnapshot } from "../types/simulation";

export class AgentInterpolator {
  private prevSnapshot: Map<string, AgentSnapshot> = new Map();
  private nextSnapshot: Map<string, AgentSnapshot> = new Map();
  private interpolateDuration = 66.7;
  private snapshotTime = 0;

  updateSnapshot(agents: AgentSnapshot[], timestamp: number, duration: number) {
    this.prevSnapshot = this.nextSnapshot;
    this.nextSnapshot = new Map(agents.map((a) => [a.uid, a]));
    this.snapshotTime = timestamp;
    this.interpolateDuration = duration;
  }

  getInterpolatedPositions(currentTime: number): AgentSnapshot[] {
    const elapsed = currentTime - this.snapshotTime;
    const t = Math.min(1.0, elapsed / this.interpolateDuration);

    const result: AgentSnapshot[] = [];

    for (const [uid, next] of this.nextSnapshot) {
      const prev = this.prevSnapshot.get(uid);

      if (!prev) {
        result.push(next);
        continue;
      }

      result.push({
        uid,
        aid: next.aid,
        species_sid: next.species_sid,
        x: prev.x + (next.x - prev.x) * t,
        y: prev.y + (next.y - prev.y) * t,
        energy: prev.energy + (next.energy - prev.energy) * t,
        age: next.age,
        state: next.state,
        target_aid: next.target_aid,
        dead: next.dead,
        count: next.count,
      });
    }

    return result;
  }
}

export function useInterpolation() {
  const interpolatorRef = useRef(new AgentInterpolator());

  const updateSnapshot = useCallback(
    (agents: AgentSnapshot[], duration: number) => {
      interpolatorRef.current.updateSnapshot(agents, performance.now(), duration);
    },
    []
  );

  const getInterpolated = useCallback((): AgentSnapshot[] => {
    return interpolatorRef.current.getInterpolatedPositions(performance.now());
  }, []);

  return { updateSnapshot, getInterpolated };
}
