"""
Behavior system for agent-based simulation.

Utility-based AI with softmax selection, hysteresis commitment periods,
emergency overrides, and steering behaviors for movement.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .agent import Agent, AgentManager
    from .spatial import SpatialHash, VegetationPatch


# ---------------------------------------------------------------------------
# Commitment periods (ticks an agent stays in a state before reconsidering)
# ---------------------------------------------------------------------------

COMMITMENT = {
    "flee": 10,
    "hunt": 8,
    "mate": 5,
    "forage": 3,
    "rest": 1,
    "flock": 3,
}

EMERGENCY_ENERGY = 10.0
SOFTMAX_TEMP = 0.5


# ---------------------------------------------------------------------------
# Utility computation
# ---------------------------------------------------------------------------

def compute_utilities(
    agent: Agent,
    neighbors: list[Agent],
    vegetation_patches: list[VegetationPatch],
    prey_sids: set[int],
    predator_sids: set[int],
    simple_mode: bool = False,
) -> dict[str, float]:
    """Return {action: utility_score} for each possible action."""

    hunger = max(0.0, 100.0 - agent.energy) / 100.0
    utilities: dict[str, float] = {"rest": 0.1}

    # Vegetation density at agent position
    veg_density = 0.0
    for p in vegetation_patches:
        d = p.get_density_at(agent.x, agent.y)
        if d > veg_density:
            veg_density = d

    # Forage: scales with hunger x vegetation x metabolism
    metab = agent.traits.get("metab", 1.0)
    utilities["forage"] = hunger * max(veg_density, 0.1) * metab * 0.5

    if simple_mode:
        # ECOSYSTEM epoch: basic flocking + grazing only
        conspecifics = [n for n in neighbors if n.species_sid == agent.species_sid and n.aid != agent.aid]
        if conspecifics:
            social = agent.traits.get("social", 0.0)
            utilities["flock"] = social * 0.3
        return utilities

    # Hunt: scales with hunger x prey availability x predator advantage
    prey_neighbors = [n for n in neighbors if n.species_sid in prey_sids and n.alive]
    if prey_neighbors:
        closest_prey = min(prey_neighbors, key=lambda p: _dist(agent, p))
        pred_advantage = (
            agent.traits.get("speed", 0.0) +
            agent.traits.get("sense", 0.0) -
            closest_prey.traits.get("camo", 0.0) * 0.7
        )
        d = _dist(agent, closest_prey)
        hunt_score = pred_advantage * hunger * (1.0 / (1.0 + d * 0.01))
        utilities["hunt"] = max(0.0, hunt_score * 0.3)

    # Flee: scales with predator proximity x threat level
    pred_neighbors = [n for n in neighbors if n.species_sid in predator_sids and n.alive]
    if pred_neighbors:
        closest_pred = min(pred_neighbors, key=lambda p: _dist(agent, p))
        threat = (
            closest_pred.traits.get("speed", 0.0) +
            closest_pred.traits.get("sense", 0.0) -
            agent.traits.get("camo", 0.0) * 0.7
        )
        d = _dist(agent, closest_pred)
        flee_score = threat * (1.0 / (1.0 + d * 0.01))
        utilities["flee"] = max(0.0, flee_score * 2.0)  # survival priority

    # Mate: only when energy surplus
    if agent.energy > 70:
        conspecifics = [n for n in neighbors if n.species_sid == agent.species_sid and n.aid != agent.aid]
        if conspecifics:
            social = agent.traits.get("social", 0.0)
            utilities["mate"] = social * 0.3

    return utilities


# ---------------------------------------------------------------------------
# Action selection with hysteresis + emergency overrides
# ---------------------------------------------------------------------------

def select_action(
    agent: Agent,
    utilities: dict[str, float],
    rng: np.random.Generator,
) -> str:
    """Softmax sampling with hysteresis and emergency overrides."""

    # Emergency override: critical energy
    if agent.energy < EMERGENCY_ENERGY:
        agent.state_commitment = 0
        if utilities.get("forage", 0) > utilities.get("hunt", 0):
            return "forage"
        elif utilities.get("hunt", 0) > 0:
            return "hunt"
        else:
            return "rest"

    # Hysteresis: continue current state if committed
    if agent.state_commitment > 0:
        agent.state_commitment -= 1
        return agent.state

    # Softmax selection
    if not utilities:
        return "rest"

    actions = list(utilities.keys())
    scores = np.array([utilities[a] for a in actions], dtype=np.float64)

    # Numerical stability
    scores = scores - scores.max()
    exp_scores = np.exp(scores / SOFTMAX_TEMP)
    total = exp_scores.sum()
    if total == 0:
        return "rest"
    probs = exp_scores / total

    new_action = str(rng.choice(actions, p=probs))
    agent.state_commitment = COMMITMENT.get(new_action, 1)
    return new_action


# ---------------------------------------------------------------------------
# Steering behaviors
# ---------------------------------------------------------------------------

def compute_velocity(
    agent: Agent,
    neighbors: list[Agent],
    vegetation_patches: list[VegetationPatch],
    prey_sids: set[int],
    predator_sids: set[int],
    all_managers: dict[int, AgentManager],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute desired velocity from current state."""

    if agent.state == "hunt" and agent.target_aid is not None and agent.target_sid is not None:
        # Find target in specific prey species
        mgr = all_managers.get(agent.target_sid)
        if mgr is not None:
            for prey_a in mgr.agents:
                if prey_a.aid == agent.target_aid and prey_a.alive:
                    return _steer_towards(agent, prey_a.x, prey_a.y)
        # Target not found, clear it
        agent.target_aid = None
        agent.target_sid = None

    if agent.state == "flee":
        pred_neighbors = [n for n in neighbors if n.species_sid in predator_sids and n.alive]
        if pred_neighbors:
            closest = min(pred_neighbors, key=lambda p: _dist(agent, p))
            return _steer_away(agent, closest.x, closest.y)

    if agent.state == "forage":
        best_patch = _nearest_patch(agent, vegetation_patches)
        if best_patch is not None:
            return _steer_towards(agent, best_patch.x, best_patch.y)

    if agent.state == "flock":
        conspecifics = [n for n in neighbors if n.species_sid == agent.species_sid and n.aid != agent.aid]
        if conspecifics:
            cx = sum(n.x for n in conspecifics) / len(conspecifics)
            cy = sum(n.y for n in conspecifics) / len(conspecifics)
            return _steer_towards(agent, cx, cy)

    if agent.state == "mate":
        conspecifics = [n for n in neighbors if n.species_sid == agent.species_sid and n.aid != agent.aid and n.energy > 60]
        if conspecifics:
            closest = min(conspecifics, key=lambda n: _dist(agent, n))
            return _steer_towards(agent, closest.x, closest.y)

    # Default: random walk
    return (float(rng.normal(0, 0.5)), float(rng.normal(0, 0.5)))


def _steer_towards(agent: Agent, tx: float, ty: float) -> tuple[float, float]:
    dx, dy = tx - agent.x, ty - agent.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1.0:
        return (0.0, 0.0)
    speed = agent.traits.get("speed", 1.0) * 2.0
    return (dx / dist * speed, dy / dist * speed)


def _steer_away(agent: Agent, tx: float, ty: float) -> tuple[float, float]:
    vx, vy = _steer_towards(agent, tx, ty)
    return (-vx, -vy)


def _nearest_patch(agent: Agent, patches: list[VegetationPatch]) -> VegetationPatch | None:
    best = None
    best_density = 0.0
    for p in patches:
        d = p.get_density_at(agent.x, agent.y)
        if d > best_density:
            best_density = d
            best = p
    # If no patch at current position, move toward nearest with capacity
    if best is None and patches:
        best_dist = float("inf")
        for p in patches:
            if p.capacity > 1.0:
                dx = p.x - agent.x
                dy = p.y - agent.y
                d2 = dx * dx + dy * dy
                if d2 < best_dist:
                    best_dist = d2
                    best = p
    return best


def _dist(a: Agent, b: Agent) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Main update function (called by AgentManager.step)
# ---------------------------------------------------------------------------

def update_behaviors(
    agents: list[Agent],
    spatial_hash: SpatialHash,
    all_managers: dict[int, AgentManager],
    vegetation_patches: list[VegetationPatch],
    prey_sids: set[int],
    predator_sids: set[int],
    rng: np.random.Generator,
    simple_mode: bool = False,
) -> None:
    """Update behavior and velocity for all agents in a species."""
    for a in agents:
        if not a.alive:
            continue

        # Query neighbors (all species within sense range)
        sense_range = max(50.0, a.traits.get("sense", 3.0) * 15.0)
        sense_range = min(sense_range, 100.0)  # cap at cell size

        # Get neighbors from own spatial hash
        neighbors = spatial_hash.query_radius(a.x, a.y, sense_range)
        # Also get neighbors from other species
        for sid, mgr in all_managers.items():
            if sid == a.species_sid:
                continue
            cross_neighbors = mgr.spatial_hash.query_radius(a.x, a.y, sense_range)
            neighbors.extend(cross_neighbors)

        # Remove self from neighbors
        neighbors = [n for n in neighbors if n.aid != a.aid or n.species_sid != a.species_sid]

        # Compute utilities
        utilities = compute_utilities(
            a, neighbors, vegetation_patches,
            prey_sids, predator_sids,
            simple_mode=simple_mode,
        )

        # Select action
        new_state = select_action(a, utilities, rng)

        # Set hunt target if switching to hunt
        if new_state == "hunt" and a.state != "hunt":
            prey_neighbors = [n for n in neighbors if n.species_sid in prey_sids and n.alive]
            if prey_neighbors:
                closest = min(prey_neighbors, key=lambda p: _dist(a, p))
                a.target_aid = closest.aid
                a.target_sid = closest.species_sid
            else:
                new_state = "forage"  # No prey, forage instead

        a.state = new_state

        # Compute velocity
        vx, vy = compute_velocity(
            a, neighbors, vegetation_patches,
            prey_sids, predator_sids,
            all_managers, rng,
        )
        a.vx = vx
        a.vy = vy
