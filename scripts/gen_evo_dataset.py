#!/usr/bin/env python3
"""
Generate evolutionary simulation traces for MANTIS training.

Thin CLI wrapper around the simulation engine + serializer.
Supports parallel generation via ProcessPoolExecutor.

Usage:
    python scripts/gen_evo_dataset.py --worlds 100 --max-generations 50 --verbose
    python scripts/gen_evo_dataset.py --worlds 10000 --max-generations 200 --workers 8 --seed 42
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


_sim_cache = None


def _import_simulation():
    """Import simulation modules without triggering mantis/__init__.py (CUDA deps)."""
    global _sim_cache
    if _sim_cache is not None:
        return _sim_cache

    import importlib.util
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_path = os.path.join(base, "mantis", "simulation")

    # Ensure mantis and mantis.simulation are registered as namespace packages
    # so relative imports within simulation work.
    import types
    if "mantis" not in sys.modules:
        pkg = types.ModuleType("mantis")
        pkg.__path__ = [os.path.join(base, "mantis")]
        pkg.__package__ = "mantis"
        sys.modules["mantis"] = pkg

    # Load mantis.simulation subpackage init
    spec = importlib.util.spec_from_file_location(
        "mantis.simulation", os.path.join(sim_path, "__init__.py"),
        submodule_search_locations=[sim_path],
    )
    sim_mod = importlib.util.module_from_spec(spec)
    sys.modules["mantis.simulation"] = sim_mod
    spec.loader.exec_module(sim_mod)
    _sim_cache = sim_mod
    return sim_mod


def simulate_world(wid: int, seed: int, max_gens: int, keyframe_interval: int,
                   enable_agents: bool = False, agent_epoch: str = "INTELLIGENCE",
                   agent_threshold: float = 15.0, compact: bool = False):
    """Run one world simulation. Standalone function for process pool."""
    sim = _import_simulation()
    World = sim.World
    Serializer = sim.Serializer

    world = World(wid, seed, enable_agents=enable_agents,
                  agent_epoch=agent_epoch, agent_threshold=agent_threshold)
    serializer = Serializer(keyframe_interval=keyframe_interval, compact=compact)

    blocks = []
    for _ in range(max_gens):
        world.step()
        block = serializer.serialize_tick(world)
        blocks.append(block)

        # Stop early if all species are dead
        if not any(sp.alive for sp in world.species):
            break

    text = "\n".join(blocks) + "\n"

    stats = {
        "wid": wid,
        "ticks": world.tick,
        "epoch": world.epoch.value,
        "alive": sum(1 for sp in world.species if sp.alive),
        "total_species": len(world.species),
        "max_tier": max((sp.max_tier() for sp in world.species if sp.alive), default=0),
        "spotlights": sum(1 for sp in world.species if sp.alive and sp.spotlight_score() > 15),
    }

    return text, stats


def generate_dataset(args):
    """Generate the full dataset, optionally in parallel."""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic per-world seeds from master seed
    master_rng = np.random.default_rng(args.seed)
    world_seeds = master_rng.integers(0, 2**31, size=args.worlds).tolist()

    t0 = time.time()
    epoch_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    max_tier_global = 0
    total_species = 0
    total_spotlights = 0

    agent_kwargs = dict(
        enable_agents=args.enable_agents,
        agent_epoch=args.agent_epoch,
        agent_threshold=args.agent_threshold,
        compact=args.compact,
    )

    def accumulate_stats(stats):
        nonlocal max_tier_global, total_species, total_spotlights
        epoch_counts[stats["epoch"]] = epoch_counts.get(stats["epoch"], 0) + 1
        max_tier_global = max(max_tier_global, stats["max_tier"])
        total_species += stats["total_species"]
        total_spotlights += stats["spotlights"]

    with open(output, "w") as f:
        if args.workers <= 1:
            # Sequential — flush after each world
            for i, (wid, seed) in enumerate(zip(range(args.worlds), world_seeds)):
                text, stats = simulate_world(wid, seed, args.max_generations, args.keyframe_interval, **agent_kwargs)
                f.write(text + "\n")  # blank line = EOS boundary between worlds
                f.flush()

                accumulate_stats(stats)

                if args.verbose and (i + 1) % max(1, args.worlds // 20) == 0:
                    pct = (i + 1) / args.worlds * 100
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    print(
                        f"  [{pct:5.1f}%] W{wid}: epoch={stats['epoch']}, "
                        f"{stats['alive']}/{stats['total_species']} alive, "
                        f"tier={stats['max_tier']} ({rate:.1f} worlds/s)"
                    )
        else:
            # Parallel — stream to disk in world-order as results arrive
            pending = {}  # wid -> text for out-of-order completions
            next_wid = 0

            def flush_pending():
                nonlocal next_wid
                while next_wid in pending:
                    f.write(pending.pop(next_wid))
                    next_wid += 1
                f.flush()

            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(simulate_world, wid, seed, args.max_generations,
                                args.keyframe_interval, **agent_kwargs): wid
                    for wid, seed in zip(range(args.worlds), world_seeds)
                }

                done_count = 0
                for future in as_completed(futures):
                    wid = futures[future]
                    text, stats = future.result()
                    pending[wid] = text + "\n"
                    flush_pending()

                    accumulate_stats(stats)

                    done_count += 1
                    if args.verbose and done_count % max(1, args.worlds // 20) == 0:
                        pct = done_count / args.worlds * 100
                        elapsed = time.time() - t0
                        rate = done_count / elapsed
                        print(
                            f"  [{pct:5.1f}%] {done_count}/{args.worlds} done, "
                            f"{next_wid} written ({rate:.1f} worlds/s)"
                        )

    elapsed = time.time() - t0
    tier_names = ["Physical", "Behavioral", "Cognitive", "Cultural", "Abstract"]
    epoch_names = {1: "Primordial", 2: "Cambrian", 3: "Ecosystem", 4: "Intelligence"}

    print(f"\nDataset generated: {output}")
    print(f"  Worlds: {args.worlds}")
    print(f"  Max generations: {args.max_generations}")
    print(f"  Keyframe interval: {args.keyframe_interval}")
    print(f"  Total species created: {total_species}")
    print(f"  Worlds with spotlights: {total_spotlights}")
    print(f"  Max trait tier reached: T{max_tier_global} ({tier_names[max_tier_global]})")
    print(f"  Epoch distribution: {', '.join(f'{epoch_names[k]}={v}' for k, v in sorted(epoch_counts.items()) if v > 0)}")
    print(f"  Time: {elapsed:.1f}s ({args.worlds / elapsed:.1f} worlds/s)")
    print(f"  File size: {output.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evolutionary simulation traces for MANTIS training",
    )
    parser.add_argument("--worlds", type=int, default=10000,
                        help="Number of independent worlds (default: 10000)")
    parser.add_argument("--max-generations", type=int, default=100,
                        help="Max generations per world (default: 100)")
    parser.add_argument("--output", type=str, default="data/evo_train.txt",
                        help="Output file path (default: data/evo_train.txt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed (default: 42)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1, sequential)")
    parser.add_argument("--keyframe-interval", type=int, default=20,
                        help="Ticks between full keyframes (default: 20)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress during generation")
    parser.add_argument("--enable-agents", action="store_true",
                        help="Enable agent-based simulation (spatial individual agents)")
    parser.add_argument("--agent-epoch", type=str, default="ECOSYSTEM",
                        choices=["ECOSYSTEM", "INTELLIGENCE"],
                        help="Epoch at which agents activate (default: ECOSYSTEM)")
    parser.add_argument("--agent-threshold", type=float, default=15.0,
                        help="Spotlight score threshold for INTELLIGENCE agent activation (default: 15.0)")
    parser.add_argument("--compact", action="store_true",
                        help="Use compact v2 format (int-scaled, space-separated, ~50%% fewer tokens)")
    args = parser.parse_args()

    generate_dataset(args)


if __name__ == "__main__":
    main()
