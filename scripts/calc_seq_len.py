#!/usr/bin/env python3
"""
Calculate optimal sequence length for evolution model training.

Runs simulations matching the 3 training partitions (bio, eco, intel),
serializes with the real tokenizer, and reports:
  1. Per-tick token distributions (keyframe vs delta, per epoch)
  2. Per-component breakdown (SP, INT, EVT, AGENT, SPOT tokens)
  3. World-level chunking simulation (how EvoWorldDataset would slice)
  4. Optimal seq_len recommendations per partition

Usage:
    python scripts/calc_seq_len.py
    python scripts/calc_seq_len.py --worlds 500 --max-generations 300
"""

import sys
import os
import importlib.util
import argparse
import statistics
import types
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Import simulation + tokenizer without triggering mantis/__init__.py (CUDA)
# ---------------------------------------------------------------------------

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sim_path = os.path.join(_base, "mantis", "simulation")

if "mantis" not in sys.modules:
    pkg = types.ModuleType("mantis")
    pkg.__path__ = [os.path.join(_base, "mantis")]
    pkg.__package__ = "mantis"
    sys.modules["mantis"] = pkg

spec = importlib.util.spec_from_file_location(
    "mantis.simulation", os.path.join(_sim_path, "__init__.py"),
    submodule_search_locations=[_sim_path],
)
sim_mod = importlib.util.module_from_spec(spec)
sys.modules["mantis.simulation"] = sim_mod
spec.loader.exec_module(sim_mod)

World = sim_mod.World
Serializer = sim_mod.Serializer
Epoch = sim_mod.Epoch

_tok_spec = importlib.util.spec_from_file_location(
    "mantis_tokenizer",
    os.path.join(_base, "mantis", "tokenizer.py"),
)
_tok_mod = importlib.util.module_from_spec(_tok_spec)
sys.modules["mantis_tokenizer"] = _tok_mod
_tok_spec.loader.exec_module(_tok_mod)
MANTISTokenizer = _tok_mod.MANTISTokenizer

W = 90  # output width


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def percentile(data_sorted, p):
    """Get percentile from sorted list."""
    if not data_sorted:
        return 0
    idx = min(int(len(data_sorted) * p), len(data_sorted) - 1)
    return data_sorted[idx]


def dist_str(data):
    """One-line distribution summary."""
    s = sorted(data)
    n = len(s)
    if n == 0:
        return "(no data)"
    return (f"n={n:,}  mean={statistics.mean(data):.0f}  "
            f"med={statistics.median(data):.0f}  "
            f"P90={percentile(s, 0.90)}  P95={percentile(s, 0.95)}  "
            f"P99={percentile(s, 0.99)}  max={s[-1]}")


def component_tokens(text: str, tokenizer) -> dict[str, int]:
    """Break a tick's text into component token counts."""
    counts = defaultdict(int)
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("=EPOCH"):
            key = "EPOCH"
        elif stripped.startswith("@BIO"):
            key = "BIO"
        elif stripped.startswith("@SP") or stripped.startswith("T ") or stripped.startswith("T:") or stripped.startswith("E ") or stripped.startswith("E:"):
            key = "SP"
        elif stripped.startswith("@INT"):
            key = "INT"
        elif stripped.startswith("@EVT") or (stripped.startswith("M") and any(stripped.startswith(p) for p in ["M+", "M-", "Mpoint", "Mdrift", "Mleap", "Mfuse"])):
            key = "EVT"
        elif stripped.startswith("@SPOT") or stripped.startswith("CTX") or stripped.startswith("ACTORS") or stripped.startswith("INTENT") or stripped.startswith("REACT") or stripped.startswith("RESOLVE") or stripped.startswith("EFFECT"):
            key = "SPOT"
        elif stripped.startswith("@AGENT") or stripped.startswith("N A") or stripped.startswith("N:A") or stripped.startswith("G ") or stripped.startswith("G(") or (stripped.startswith("A") and stripped.endswith("\u2020")):
            key = "AGENT"
        elif stripped == "---":
            key = "SEP"
        else:
            key = "OTHER"
        counts[key] += count_tokens(line, tokenizer)
    # newlines between lines
    n_lines = len([l for l in text.split("\n") if l.strip()])
    if n_lines > 1:
        counts["NL"] = n_lines - 1
    return dict(counts)


def simulate_partition(name, n_worlds, max_gen, seed, kf_interval, compact,
                       enable_agents, agent_epoch, max_epoch, tokenizer):
    """Simulate worlds for one partition. Returns tick and world data."""
    tick_tokens = {"keyframe": [], "delta": []}
    tick_components = {"keyframe": [], "delta": []}
    world_token_lists = []  # list of per-world token sequences
    species_counts = []
    agent_token_totals = []

    for w in range(n_worlds):
        if (w + 1) % max(1, n_worlds // 5) == 0:
            print(f"    {name}: World {w+1}/{n_worlds}...")

        world = World(
            wid=w + seed, seed=seed + w,
            enable_agents=enable_agents,
            agent_epoch=agent_epoch,
        )
        serializer = Serializer(keyframe_interval=kf_interval, compact=compact)
        world_tick_tokens = []

        if max_epoch is not None:
            epoch_cap = Epoch[max_epoch].value

        for _ in range(max_gen):
            world.step()
            if max_epoch is not None and world.epoch.value > epoch_cap:
                break

            text = serializer.serialize_tick(world)
            toks = count_tokens(text, tokenizer)

            is_kf = (world.tick % kf_interval == 1) or world.tick == 1 or world.epoch_just_changed
            tt = "keyframe" if is_kf else "delta"

            tick_tokens[tt].append(toks)
            world_tick_tokens.append(toks)

            comp = component_tokens(text, tokenizer)
            tick_components[tt].append(comp)
            agent_token_totals.append(comp.get("AGENT", 0))

            alive = [sp for sp in world.species if sp.alive]
            species_counts.append(len(alive))

            if not alive:
                break

        world_token_lists.append(world_tick_tokens)

    return {
        "tick_tokens": tick_tokens,
        "tick_components": tick_components,
        "world_token_lists": world_token_lists,
        "species_counts": species_counts,
        "agent_token_totals": agent_token_totals,
    }


def chunking_analysis(world_token_lists, seq_len):
    """Simulate EvoWorldDataset chunking: concatenate world tokens, chunk with stride=seq_len.
    Returns (n_sequences, avg_utilization, n_pad_heavy)."""
    n_seqs = 0
    total_tokens = 0
    total_capacity = 0
    pad_heavy = 0  # sequences with >50% padding

    for world_toks in world_token_lists:
        world_total = sum(world_toks) + 1  # +1 for EOS
        if world_total < 2:
            continue
        # Chunk into seq_len+1 windows (input+label)
        for start in range(0, world_total, seq_len):
            chunk_len = min(seq_len + 1, world_total - start)
            if chunk_len < 2:
                continue
            real_input_tokens = chunk_len - 1  # exclude label position
            n_seqs += 1
            total_tokens += real_input_tokens
            total_capacity += seq_len
            if real_input_tokens < seq_len * 0.5:
                pad_heavy += 1

    utilization = total_tokens / total_capacity * 100 if total_capacity > 0 else 0
    return n_seqs, utilization, pad_heavy


def print_partition_report(name, data, tokenizer):
    """Print analysis for one partition."""
    print(f"\n{'  ' + name + '  ':=^{W}}")

    tt = data["tick_tokens"]
    tc = data["tick_components"]

    for tick_type in ["keyframe", "delta"]:
        toks = tt[tick_type]
        if not toks:
            continue
        print(f"\n  {tick_type.upper()} ticks: {dist_str(toks)}")

        # Average component breakdown
        comps = tc[tick_type]
        if comps:
            avg = defaultdict(float)
            mx = defaultdict(int)
            for c in comps:
                for k, v in c.items():
                    avg[k] += v
                    mx[k] = max(mx[k], v)
            for k in avg:
                avg[k] /= len(comps)

            parts = []
            for comp in ["EPOCH", "BIO", "SP", "INT", "EVT", "SPOT", "AGENT", "SEP", "NL"]:
                if avg.get(comp, 0) > 0.5:
                    parts.append(f"{comp}={avg[comp]:.0f}/{mx[comp]}")
            print(f"    Components (avg/max): {', '.join(parts)}")

    # Species stats
    sc = data["species_counts"]
    if sc:
        print(f"\n  Species: mean={statistics.mean(sc):.1f}, max={max(sc)}")

    # Agent dominance check
    at = data["agent_token_totals"]
    agent_ticks = [t for t in at if t > 0]
    if agent_ticks:
        all_toks = tt["keyframe"] + tt["delta"]
        total_tok = sum(all_toks)
        total_agent = sum(at)
        print(f"  Agent tokens: {total_agent:,} / {total_tok:,} total ({total_agent/total_tok*100:.1f}%)")
        print(f"    Agent ticks: {len(agent_ticks)}, avg agent tokens/tick: {statistics.mean(agent_ticks):.0f}")


def recommend(name, data, seq_lens_to_test):
    """Print seq_len recommendations for a partition."""
    all_ticks = data["tick_tokens"]["keyframe"] + data["tick_tokens"]["delta"]
    kf = data["tick_tokens"]["keyframe"]
    worlds = data["world_token_lists"]

    if not all_ticks:
        return

    kf_sorted = sorted(kf)
    all_sorted = sorted(all_ticks)
    n = len(all_sorted)
    n_kf = len(kf_sorted)

    print(f"\n  --- {name} Recommendations ---")
    print(f"  Keyframes: P50={percentile(kf_sorted, 0.50):>6d}  P90={percentile(kf_sorted, 0.90):>6d}  "
          f"P95={percentile(kf_sorted, 0.95):>6d}  P99={percentile(kf_sorted, 0.99):>6d}  max={kf_sorted[-1]:>6d}")
    print()
    print(f"  {'seq_len':>8s}  {'Util%':>6s}  {'KF trunc%':>10s}  {'Seqs':>8s}  {'Pad>50%':>8s}  Notes")
    print(f"  {'--------':>8s}  {'------':>6s}  {'----------':>10s}  {'--------':>8s}  {'--------':>8s}  -----")

    for sl in seq_lens_to_test:
        n_seqs, util, pad_heavy = chunking_analysis(worlds, sl)
        kf_trunc = sum(1 for t in kf if t > sl) / n_kf * 100 if n_kf > 0 else 0
        note = ""
        if kf_trunc == 0:
            note = "all keyframes fit"
        elif kf_trunc < 2:
            note = "nearly all keyframes fit"
        elif kf_trunc < 10:
            note = "some keyframes truncated"
        else:
            note = "many keyframes truncated"
        print(f"  {sl:>8d}  {util:>5.1f}%  {kf_trunc:>9.1f}%  {n_seqs:>8,}  {pad_heavy:>8,}  {note}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate optimal sequence length for evolution training",
    )
    parser.add_argument("--worlds", type=int, default=200,
                        help="Worlds per partition (default: 200)")
    parser.add_argument("--max-generations", type=int, default=200,
                        help="Max generations per world (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--keyframe-interval", type=int, default=20,
                        help="Keyframe interval (default: 20)")
    parser.add_argument("--compact", action="store_true", default=True,
                        help="Use compact v2 format (default: True)")
    parser.add_argument("--no-compact", action="store_false", dest="compact")
    args = parser.parse_args()

    tokenizer = MANTISTokenizer()
    print(f"Tokenizer: {len(tokenizer)} tokens\n")

    # =======================================================================
    # Simulate the 3 training partitions matching gen_evo_dataset.py usage
    # =======================================================================
    # Bio:   --max-epoch CAMBRIAN (no agents)
    # Eco:   --max-epoch ECOSYSTEM (agents at ECOSYSTEM)
    # Intel: all epochs (agents at INTELLIGENCE)

    partitions = {
        "BIO (PRIMORDIAL+CAMBRIAN, no agents)": {
            "enable_agents": False, "agent_epoch": "INTELLIGENCE",
            "max_epoch": "CAMBRIAN",
        },
        "ECO (->ECOSYSTEM, agents@ECOSYSTEM)": {
            "enable_agents": True, "agent_epoch": "ECOSYSTEM",
            "max_epoch": "ECOSYSTEM",
        },
        "INTEL (all epochs, agents@INTELLIGENCE)": {
            "enable_agents": True, "agent_epoch": "INTELLIGENCE",
            "max_epoch": None,
        },
    }

    # Eco/Intel partitions with agents are very slow, use fewer worlds
    agent_worlds = max(10, args.worlds // 10)

    results = {}
    for pname, pcfg in partitions.items():
        n_w = agent_worlds if pcfg["enable_agents"] else args.worlds
        print(f"Simulating {pname} ({n_w} worlds)...")
        results[pname] = simulate_partition(
            name=pname,
            n_worlds=n_w,
            max_gen=args.max_generations,
            seed=args.seed + hash(pname) % 10000,
            kf_interval=args.keyframe_interval,
            compact=args.compact,
            enable_agents=pcfg["enable_agents"],
            agent_epoch=pcfg["agent_epoch"],
            max_epoch=pcfg["max_epoch"],
            tokenizer=tokenizer,
        )

    # =======================================================================
    # Report
    # =======================================================================
    print(f"\n{'=' * W}")
    print(f"{'SEQUENCE LENGTH ANALYSIS':^{W}}")
    print(f"{'=' * W}")

    for pname, data in results.items():
        print_partition_report(pname, data, tokenizer)

    # =======================================================================
    # Recommendations per partition
    # =======================================================================
    print(f"\n{'=' * W}")
    print(f"{'RECOMMENDATIONS':^{W}}")
    print(f"{'=' * W}")
    print()
    print("  Util% = token utilization (higher = less padding waste)")
    print("  KF trunc% = keyframe ticks exceeding seq_len (lower = better)")
    print("  Pad>50% = sequences that are >50% padding (waste)")

    seq_lens = [256, 384, 512, 768, 1024, 1280, 1536, 2048, 4096, 8192, 16384, 32768]

    for pname, data in results.items():
        recommend(pname, data, seq_lens)

    # =======================================================================
    # Final summary
    # =======================================================================
    print(f"\n{'=' * W}")
    print(f"{'SUMMARY':^{W}}")
    print(f"{'=' * W}")

    for pname, data in results.items():
        kf = sorted(data["tick_tokens"]["keyframe"])
        dl = sorted(data["tick_tokens"]["delta"])
        if not kf:
            continue

        # Find optimal seq_len: smallest that keeps KF truncation < 2%
        # and utilization > 40%
        best = None
        for sl in range(128, 65536 + 1, 128):
            kf_trunc = sum(1 for t in kf if t > sl) / len(kf) * 100
            _, util, _ = chunking_analysis(data["world_token_lists"], sl)
            if kf_trunc < 2.0 and util > 30.0:
                best = sl
                break

        # Also find nearest power of 2
        best_p2 = None
        for p in range(7, 17):
            sl = 1 << p
            kf_trunc = sum(1 for t in kf if t > sl) / len(kf) * 100
            if kf_trunc < 2.0:
                best_p2 = sl
                break

        print(f"\n  {pname}:")
        print(f"    Keyframe P95={percentile(kf, 0.95):>6d}  P99={percentile(kf, 0.99):>6d}  max={kf[-1]:>6d}")
        print(f"    Delta    P95={percentile(dl, 0.95):>6d}  P99={percentile(dl, 0.99):>6d}  max={dl[-1] if dl else 0:>6d}")
        if best:
            _, util, _ = chunking_analysis(data["world_token_lists"], best)
            print(f"    -> Recommended seq_len (128-aligned): {best:>6d}  (util={util:.1f}%)")
        if best_p2:
            _, util, _ = chunking_analysis(data["world_token_lists"], best_p2)
            print(f"    -> Recommended seq_len (power-of-2):  {best_p2:>6d}  (util={util:.1f}%)")

        # Agent warning
        at = data["agent_token_totals"]
        agent_ticks = [t for t in at if t > 0]
        if agent_ticks and statistics.mean(agent_ticks) > 1000:
            print(f"    !! Agent blocks dominate: avg {statistics.mean(agent_ticks):.0f} tokens/tick")
            print(f"    !! Consider reducing agent count or using coarser agent serialization")

    print(f"\n{'=' * W}")


if __name__ == "__main__":
    main()
