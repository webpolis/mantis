"""
Protocol format serializer for the ecological evolution simulator.

Outputs structured text matching the tokens defined in MANTISTokenizer:
    =EPOCH, @BIO, @SP, @INT, @EVT, @SPOT, ---, Δ, M+, M-, etc.

Supports keyframe (full state dump) and delta (changes only) encoding
to reduce output size while preserving all causal information.
"""

from __future__ import annotations

import math
from typing import Optional

from .agent_serializer import AgentSerializerState
from .constants import EPOCH_CONFIG, TRAIT_TO_TIER, ATTRIBUTE_TRIGGERS, Epoch
from .engine import World, Interaction, SpotlightEvent


class Serializer:
    """Serialize World state into the MANTIS protocol text format."""

    def __init__(self, keyframe_interval: int = 20, compact: bool = False):
        self.keyframe_interval = keyframe_interval
        self.compact = compact
        # Track last keyframe state for delta encoding
        self._last_keyframe: dict[int, dict] = {}  # sid -> snapshot
        self._species_since_keyframe: set[int] = set()  # new species since last keyframe
        self._agent_serializer = AgentSerializerState()

    def serialize_tick(self, world: World) -> str:
        """Serialize one tick of the world. Returns full protocol text block."""
        is_keyframe = (world.tick % self.keyframe_interval == 1) or world.tick == 1 or world.epoch_just_changed
        lines: list[str] = []
        c = self.compact

        # Epoch header (always on keyframe or epoch change)
        if is_keyframe or world.epoch_just_changed:
            tick_scale = EPOCH_CONFIG[world.epoch]["tick_scale"]
            if c:
                lines.append(f"=EPOCH {world.epoch.value} {tick_scale} W{world.wid}")
            else:
                lines.append(f"=EPOCH:{world.epoch.value}|TICK_SCALE:{tick_scale}gen|W{world.wid}")

        # World-level events (catastrophes — before species blocks)
        for evt in world.world_events:
            if evt.startswith("catastrophe:"):
                payload = evt.split(":", 1)[1]
                if c:
                    # "volcanic_winter|dur=8" -> "volcanic_winter dur 8"
                    parts = payload.replace("|", " ").replace("=", " ")
                    lines.append(f"@EVT WORLD catastrophe {parts}")
                else:
                    lines.append(f"@EVT|WORLD|catastrophe|{payload}")
            elif evt.startswith("catastrophe_end:"):
                ctype = evt.split(":", 1)[1]
                if c:
                    lines.append(f"@EVT WORLD catastrophe_end {ctype}")
                else:
                    lines.append(f"@EVT|WORLD|catastrophe_end|type={ctype}")

        # Biome state
        if is_keyframe:
            if c:
                bio_parts = [b.serialize_header_compact() for b in world.biomes]
                lines.append(f"@BIO {' '.join(bio_parts)}")
            else:
                bio_parts = [b.serialize_header() for b in world.biomes]
                lines.append(f"@BIO|{'|'.join(bio_parts)}")

        # Species blocks
        alive = [sp for sp in world.species if sp.alive]
        dead_this_tick = [
            sp for sp in world.species
            if not sp.alive and sp.sid in world.events
        ]

        for sp in alive:
            if is_keyframe or sp.sid in self._species_since_keyframe:
                lines.extend(self._serialize_species_keyframe(sp, world))
                self._snapshot_species(sp, world)
                self._species_since_keyframe.discard(sp.sid)
            else:
                delta_lines = self._serialize_species_delta(sp, world)
                if delta_lines:
                    lines.extend(delta_lines)

        # Dead species this tick
        for sp in dead_this_tick:
            events = world.events.get(sp.sid, [])
            for evt in events:
                if evt.startswith("extinction:"):
                    cause = evt.split(":", 1)[1]
                    if c:
                        lines.append(f"@EVT S{sp.sid} extinction {cause}")
                    else:
                        lines.append(f"@EVT|S{sp.sid}|extinction|cause={cause}")

        # Interactions
        for interaction in world.interactions:
            lines.append(self._serialize_interaction(interaction))

        # Mutation events (trait emergence, body plan changes)
        for sp in alive:
            muts = world.mutations.get(sp.sid, [])
            evts = world.events.get(sp.sid, [])
            for mut in muts:
                if mut.startswith("M+"):
                    parts = mut.split("|", 1)
                    if c:
                        lines.append(self._compact_evt(sp.sid, parts))
                    else:
                        lines.append(f"@EVT|S{sp.sid}|{parts[0]}|{parts[1]}")
                elif mut.startswith("Mfuse"):
                    parts = mut.split("|", 1)
                    if c:
                        lines.append(self._compact_evt(sp.sid, parts))
                    else:
                        lines.append(f"@EVT|S{sp.sid}|{parts[0]}|{parts[1]}")
            for evt in evts:
                if evt.startswith("body_plan:"):
                    transition = evt.split(":", 1)[1]
                    if c:
                        lines.append(f"@EVT S{sp.sid} body_plan {transition}")
                    else:
                        lines.append(f"@EVT|S{sp.sid}|body_plan|{transition}")
                elif evt.startswith("speciation:"):
                    child = evt.split(":", 1)[1]
                    if c:
                        lines.append(f"@EVT S{sp.sid} speciation {child}")
                    else:
                        lines.append(f"@EVT|S{sp.sid}|speciation|child={child}")
                    try:
                        child_sid = int(child[1:])
                        self._species_since_keyframe.add(child_sid)
                    except ValueError:
                        pass
                elif evt.startswith("disease:"):
                    payload = evt.split(":", 1)[1]  # "plague|pop-=2000"
                    if c:
                        # "plague|pop-=2000" -> "plague p-2000"
                        parts = payload.split("|")
                        dtype = parts[0]
                        pop_part = parts[1].replace("pop-=", "p-") if len(parts) > 1 else ""
                        lines.append(f"@EVT S{sp.sid} disease {dtype} {pop_part}")
                    else:
                        lines.append(f"@EVT|S{sp.sid}|disease|{payload}")
                elif evt.startswith("symbiogenesis:"):
                    payload = evt.split(":", 1)[1]  # "S3+S5->S8|gained=photosynth,toxin_resist"
                    if c:
                        # Parse "S3+S5->S8|gained=photosynth,toxin_resist"
                        main, gained_part = payload.split("|", 1) if "|" in payload else (payload, "")
                        parents, child_id = main.split("->") if "->" in main else (main, "")
                        gained = gained_part.replace("gained=", "").replace(",", " ") if gained_part else ""
                        lines.append(f"@EVT S{sp.sid} symbiogenesis {parents} {child_id} {gained}".rstrip())
                    else:
                        lines.append(f"@EVT|S{sp.sid}|symbiogenesis|{payload}")
                    # Track child for keyframe
                    if "->" in payload:
                        try:
                            child_id_str = payload.split("->")[1].split("|")[0]
                            child_sid = int(child_id_str[1:])
                            self._species_since_keyframe.add(child_sid)
                        except (ValueError, IndexError):
                            pass
                elif evt.startswith("evo_trap:"):
                    if c:
                        lines.append(f"@EVT S{sp.sid} evo_trap low_var")
                    else:
                        lines.append(f"@EVT|S{sp.sid}|evo_trap|low_variance")

        # Spotlight blocks
        for spot in world.spotlights:
            lines.extend(self._serialize_spotlight(spot, world))

        # Separator
        lines.append("---")

        # Update keyframe tracking
        if is_keyframe:
            self._species_since_keyframe.clear()
            for sp in alive:
                self._snapshot_species(sp, world)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Species keyframe (full state)
    # ------------------------------------------------------------------

    def _serialize_species_keyframe(self, sp, world: World) -> list[str]:
        """Full species state dump."""
        lines = []
        c = self.compact

        # Location(s)
        locs = ",".join(sorted(sp.locations)) if not c else " ".join(sorted(sp.locations))

        # Diet
        diet_str = self._format_diet(sp)

        # Population with variance proxy
        pop_var = max(1, int(sp.population * 0.1))

        # Repro strategy
        repro = sp.reproduction_strategy()
        repro_rate = sp.traits["repro"].mean if "repro" in sp.traits else 1.0

        if c:
            # v2: @SP S0 L0 scavenger 6565 D det 51 plt 35 sol 14
            header = f"@SP S{sp.sid} {locs} {sp.body_plan.name} {sp.population} D {diet_str}"
            lines.append(header)

            # Trait line — space-separated, ×10 int-scaled
            trait_parts = []
            for trait_name, td in sorted(sp.traits.items(), key=lambda x: (TRAIT_TO_TIER.get(x[0], 0), x[0])):
                trait_parts.append(f"{trait_name} {_int10(td.mean)}\u00b1{_int10(math.sqrt(td.variance))}")
            for trait_name, td in sorted(sp.fused_traits.items()):
                trait_parts.append(f"*{trait_name} {_int10(td.mean)}\u00b1{_int10(math.sqrt(td.variance))}")
            if trait_parts:
                lines.append(f"  T {' '.join(trait_parts)}")

            # Energy line — positional: income cost store strategy rate×10
            e_income, e_cost = world.energy_log.get(sp.sid, (0.0, 0.0))
            lines.append(f"  E {int(round(e_income))} {int(round(e_cost))} {int(round(sp.energy_store))} {repro} {_int10(repro_rate)}")
        else:
            # Include agent count in header if agents active
            agent_info = ""
            if sp.agent_manager is not None:
                agent_info = f"({len(sp.agent_manager.agents)} agents)"

            header = (
                f"@SP|S{sp.sid}|{locs}|plan={sp.body_plan.name}"
                f"|pop={sp.population}\u00b1{pop_var}{agent_info}|diet={{{diet_str}}}"
            )
            lines.append(header)

            # Trait line — group by tier
            trait_parts = []
            for trait_name, td in sorted(sp.traits.items(), key=lambda x: (TRAIT_TO_TIER.get(x[0], 0), x[0])):
                trait_parts.append(f"{trait_name}={td.mean:.1f}\u00b1{math.sqrt(td.variance):.1f}")
            for trait_name, td in sorted(sp.fused_traits.items()):
                trait_parts.append(f"*{trait_name}={td.mean:.1f}\u00b1{math.sqrt(td.variance):.1f}")
            if trait_parts:
                lines.append(f"  T:{','.join(trait_parts)}")

            # Energy line
            e_income, e_cost = world.energy_log.get(sp.sid, (0.0, 0.0))
            lines.append(
                f"  E:in={e_income:.0f},out={e_cost:.0f},store={sp.energy_store:.0f}"
                f"|repro={repro}(rate={repro_rate:.1f})"
            )

        # Agent block (if active)
        if sp.agent_manager is not None:
            agent_lines = self._agent_serializer.serialize(
                sp.sid, sp.agent_manager, is_keyframe=True,
                compact=c,
            )
            lines.extend(agent_lines)

        return lines

    # ------------------------------------------------------------------
    # Species delta (changes since last keyframe)
    # ------------------------------------------------------------------

    def _serialize_species_delta(self, sp, world: World) -> list[str]:
        """Delta encoding: only changed fields."""
        prev = self._last_keyframe.get(sp.sid)
        if prev is None:
            return self._serialize_species_keyframe(sp, world)

        lines = []
        changes = []
        c = self.compact

        # Population change
        if sp.population != prev.get("pop"):
            changes.append(f"pop {sp.population}" if c else f"pop={sp.population}")

        # Location change (snapshot always stores comma-separated)
        locs_key = ",".join(sorted(sp.locations))
        if locs_key != prev.get("locs"):
            if c:
                changes.append(f"locs {' '.join(sorted(sp.locations))}")
            else:
                changes.append(f"locs={locs_key}")

        if changes:
            if c:
                lines.append(f"@SP S{sp.sid} {' '.join(changes)}")
            else:
                lines.append(f"@SP|S{sp.sid}|{'|'.join(changes)}")

        # Trait deltas
        trait_deltas = []
        for trait_name, td in sp.traits.items():
            prev_mean = prev.get("traits", {}).get(trait_name)
            if prev_mean is not None:
                delta = td.mean - prev_mean
                if abs(delta) >= 0.05:
                    if c:
                        trait_deltas.append(f"\u0394{trait_name} {_int10(delta):+d}")
                    else:
                        sign = "+" if delta > 0 else ""
                        trait_deltas.append(f"\u0394{trait_name}={sign}{delta:.1f}")
            else:
                # New trait since keyframe
                if c:
                    trait_deltas.append(f"{trait_name} {_int10(td.mean)}\u00b1{_int10(math.sqrt(td.variance))}")
                else:
                    trait_deltas.append(f"{trait_name}={td.mean:.1f}\u00b1{math.sqrt(td.variance):.1f}")

        for trait_name, td in sp.fused_traits.items():
            prev_mean = prev.get("fused", {}).get(trait_name)
            if prev_mean is not None:
                delta = td.mean - prev_mean
                if abs(delta) >= 0.05:
                    if c:
                        trait_deltas.append(f"\u0394*{trait_name} {_int10(delta):+d}")
                    else:
                        sign = "+" if delta > 0 else ""
                        trait_deltas.append(f"\u0394*{trait_name}={sign}{delta:.1f}")
            else:
                if c:
                    trait_deltas.append(f"*{trait_name} {_int10(td.mean)}\u00b1{_int10(math.sqrt(td.variance))}")
                else:
                    trait_deltas.append(f"*{trait_name}={td.mean:.1f}\u00b1{math.sqrt(td.variance):.1f}")

        if trait_deltas and not lines:
            lines.append(f"@SP S{sp.sid}" if c else f"@SP|S{sp.sid}")

        if trait_deltas:
            if c:
                lines.append(f"  T {' '.join(trait_deltas)}")
            else:
                lines.append(f"  T:{','.join(trait_deltas)}")

        # Mutations on this tick
        muts = world.mutations.get(sp.sid, [])
        for mut in muts:
            if not mut.startswith("M+") and not mut.startswith("Mfuse"):
                parts = mut.split("|", 1)
                if len(parts) == 2:
                    if c:
                        lines.append(f"  {parts[0]} {self._compact_mutation_payload(parts[1])}")
                    else:
                        lines.append(f"  {parts[0]}:{parts[1]}")

        # Agent block delta (if active)
        if sp.agent_manager is not None:
            agent_lines = self._agent_serializer.serialize(
                sp.sid, sp.agent_manager, is_keyframe=False,
                compact=c,
            )
            if agent_lines:
                if not lines:
                    lines.append(f"@SP S{sp.sid}" if c else f"@SP|S{sp.sid}")
                lines.extend(agent_lines)

        # Update snapshot
        self._snapshot_species(sp, world)

        return lines

    # ------------------------------------------------------------------
    # Snapshot for delta tracking
    # ------------------------------------------------------------------

    def _snapshot_species(self, sp, world: World) -> None:
        """Take a snapshot of species state for delta computation."""
        self._last_keyframe[sp.sid] = {
            "pop": sp.population,
            "locs": ",".join(sorted(sp.locations)),
            "traits": {t: td.mean for t, td in sp.traits.items()},
            "fused": {t: td.mean for t, td in sp.fused_traits.items()},
            "energy_store": sp.energy_store,
        }

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

    def _serialize_interaction(self, interaction: Interaction) -> str:
        """Serialize an interaction event."""
        if self.compact:
            return self._compact_interaction(interaction)

        parts = [f"@INT|S{interaction.actor_sid} {interaction.verb} S{interaction.target_sid}"]
        parts.append(f"success={interaction.success:.2f}")

        for sid, effects in interaction.effects.items():
            effect_strs = []
            for key, val in effects.items():
                if key == "pop":
                    sign = "+" if val > 0 else ""
                    effect_strs.append(f"pop{sign}={val}")
                elif key == "E":
                    sign = "+" if val > 0 else ""
                    effect_strs.append(f"E{sign}={val:.0f}")
            if effect_strs:
                parts.append(f"S{sid}:{','.join(effect_strs)}")

        return "|".join(parts)

    def _compact_interaction(self, interaction: Interaction) -> str:
        """Compact v2 interaction: @INT S3 compete S1 79 S1 p-242."""
        parts = [f"@INT S{interaction.actor_sid} {interaction.verb} S{interaction.target_sid}"]
        parts.append(str(_int100(interaction.success)))

        for sid, effects in interaction.effects.items():
            for key, val in effects.items():
                if key == "pop":
                    parts.append(f"S{sid} p{val:+d}" if val >= 0 else f"S{sid} p{val}")
                elif key == "E":
                    parts.append(f"S{sid} e{int(round(val)):+d}" if val >= 0 else f"S{sid} e{int(round(val))}")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Spotlights
    # ------------------------------------------------------------------

    def _serialize_spotlight(self, spot: SpotlightEvent, world: World) -> list[str]:
        """Serialize a spotlight event block."""
        lines = []

        if self.compact:
            return self._compact_spotlight(spot, world)

        # Header
        lines.append(f"@SPOT|W{spot.world_id}G{spot.gen}|L{spot.biome_lid}|S{spot.species_sid}")

        # CTX line
        sp = next((s for s in world.species if s.sid == spot.species_sid), None)
        ctx_parts = []
        if sp:
            intel = sp.traits["intel"].mean if "intel" in sp.traits else 0.0
            social = sp.traits["social"].mean if "social" in sp.traits else 0.0
            ctx_parts.append(f"S{sp.sid}:intel={intel:.1f},social={social:.1f}")
        # Cultural memories
        for mem in spot.cultural_memories:
            ctx_parts.append(f"Cmem:{mem.name}={mem.mtype}({mem.strength:.1f})")
        if ctx_parts:
            lines.append(f"  CTX|{'|'.join(ctx_parts)}")

        # ACTORS
        actor_parts = [f"H{h.hid}:{h.role}(inf={h.influence:.1f})" for h in spot.heroes]
        lines.append(f"  ACTORS|{'|'.join(actor_parts)}")

        # INTENT
        lines.append(f"  INTENT|{spot.intent}")

        # REACT
        lines.append(f"  REACT|{spot.react}")

        # RESOLVE
        lines.append(f"  RESOLVE|{spot.resolve}")

        # EFFECT
        lines.append(f"  EFFECT|{spot.effect}")

        return lines

    def _compact_spotlight(self, spot: SpotlightEvent, world: World) -> list[str]:
        """Compact v2 spotlight block."""
        lines = []

        # Header — space-separated
        lines.append(f"@SPOT W{spot.world_id}G{spot.gen} L{spot.biome_lid} S{spot.species_sid}")

        # CTX — int-scaled traits
        sp = next((s for s in world.species if s.sid == spot.species_sid), None)
        ctx_parts = []
        if sp:
            intel = sp.traits["intel"].mean if "intel" in sp.traits else 0.0
            social = sp.traits["social"].mean if "social" in sp.traits else 0.0
            ctx_parts.append(f"S{sp.sid} intel {_int10(intel)} social {_int10(social)}")
        for mem in spot.cultural_memories:
            ctx_parts.append(f"Cmem {mem.name} {mem.mtype} {_int10(mem.strength)}")
        if ctx_parts:
            lines.append(f"  CTX {' '.join(ctx_parts)}")

        # ACTORS — compact: H1 Elder 72 H3 Scout 31
        actor_parts = [f"H{h.hid} {h.role} {_int10(h.influence)}" for h in spot.heroes]
        lines.append(f"  ACTORS {' '.join(actor_parts)}")

        # Narrative lines unchanged — highest loss weight, structural tokens already efficient
        lines.append(f"  INTENT|{spot.intent}")
        lines.append(f"  REACT|{spot.react}")
        lines.append(f"  RESOLVE|{spot.resolve}")
        lines.append(f"  EFFECT|{spot.effect}")

        return lines

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_diet(self, sp) -> str:
        """Format diet vector as compact string."""
        sources = sp.diet.sources()
        parts = []
        if self.compact:
            for source, proportion in sorted(sources.items(), key=lambda x: -x[1]):
                if proportion >= 0.01:
                    abbrev = _DIET_ABBREV.get(source, source)
                    parts.append(f"{abbrev} {_int100(proportion)}")
            return " ".join(parts) if parts else "none 100"
        for source, proportion in sorted(sources.items(), key=lambda x: -x[1]):
            if proportion >= 0.01:
                parts.append(f"{source}:{proportion:.2f}")
        return ",".join(parts) if parts else "none:1.0"

    def _compact_evt(self, sid: int, parts: list[str]) -> str:
        """Compact v2 @EVT line for M+/Mfuse mutations."""
        # parts[0] = "M+" or "Mfuse:speed+armor", parts[1] = payload like "repro=1.5±0.7"
        tag = parts[0]
        payload = self._compact_mutation_payload(parts[1]) if len(parts) > 1 else ""
        return f"@EVT S{sid} {tag} {payload}".rstrip()

    @staticmethod
    def _compact_mutation_payload(payload: str) -> str:
        """Convert mutation payload values to ×10 int-scaled form."""
        # e.g. "repro=1.5±0.7" -> "repro 15±7", "speed=-0.5" -> "speed -5"
        result = []
        for item in payload.split(","):
            item = item.strip()
            if "=" in item:
                key, val = item.split("=", 1)
                if "\u00b1" in val:
                    mean_s, var_s = val.split("\u00b1")
                    try:
                        result.append(f"{key} {_int10(float(mean_s))}\u00b1{_int10(float(var_s))}")
                    except ValueError:
                        result.append(f"{key} {val}")
                else:
                    try:
                        result.append(f"{key} {_int10(float(val))}")
                    except ValueError:
                        result.append(f"{key} {val}")
            else:
                result.append(item)
        return " ".join(result)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

_DIET_ABBREV = {
    "detritus": "det",
    "plant": "plt",
    "solar": "sol",
}


def _int10(f: float) -> int:
    """Scale ×10 and round to int."""
    return int(round(f * 10))


def _int100(f: float) -> int:
    """Scale ×100 and round to int."""
    return int(round(f * 100))
