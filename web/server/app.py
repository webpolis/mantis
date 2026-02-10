"""
Flask + Socket.IO backend for the MANTIS agent simulation playground.

Streams simulation tick data to the React frontend at 10-20 TPS.
The client interpolates to 60fps for smooth rendering.
"""

from __future__ import annotations

import sys
import os
import time

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit

from stream_simulation import (
    parse_protocol_to_ticks,
    serialize_agent_for_frontend,
    serialize_species_for_frontend,
    serialize_biome_for_frontend,
    serialize_event_for_frontend,
    split_worlds,
)

app = Flask(__name__, static_folder=None)
socketio = SocketIO(app, cors_allowed_origins="*")

is_playing = False
current_speed = 1.0
TICK_RATE = 15  # Server updates at 15 ticks/sec
_sim_gen = 0  # Incremented on each new simulation; lets old loops exit cleanly


def _wait_or_abort(gen: int) -> bool:
    """Spin-wait while paused. Returns False if a new simulation superseded us."""
    while not is_playing:
        if _sim_gen != gen:
            return False
        socketio.sleep(0.1)
    return _sim_gen == gen

# Cached model inference engine (lazily loaded)
_engine_cache: dict = {"path": None, "engine": None}


@app.route("/")
def index():
    """Serve the frontend (if built)."""
    client_dist = os.path.join(os.path.dirname(__file__), "..", "client", "dist")
    if os.path.exists(os.path.join(client_dist, "index.html")):
        return send_from_directory(client_dist, "index.html")
    return "<h1>MANTIS Playground</h1><p>Build the client: cd web/client && npm run build</p>"


@app.route("/assets/<path:filename>")
def assets(filename):
    client_dist = os.path.join(os.path.dirname(__file__), "..", "client", "dist", "assets")
    return send_from_directory(client_dist, filename)


@socketio.on("start_simulation")
def handle_start(data=None):
    """Stream simulation ticks to client at reduced rate."""
    global is_playing, _sim_gen
    _sim_gen += 1
    my_gen = _sim_gen
    is_playing = True

    data = data or {}
    data_file = data.get("file", None)
    data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

    # Find a protocol file to stream
    filepath = None
    if data_file:
        # Restrict to filenames within the data directory
        candidate = os.path.realpath(os.path.join(data_dir, os.path.basename(data_file)))
        if candidate.startswith(data_dir) and os.path.exists(candidate):
            filepath = candidate

    if filepath is None:
        # Look for any .txt file in data/
        if os.path.isdir(data_dir):
            for f in sorted(os.listdir(data_dir)):
                if f.endswith(".txt"):
                    filepath = os.path.join(data_dir, f)
                    break

    if filepath is None or not os.path.exists(filepath):
        emit("error", {"message": "No simulation data file found. Generate one first."})
        return

    with open(filepath) as f:
        text = f.read()

    worlds = split_worlds(text)
    world_index = max(0, min(data.get("world_index", 0), len(worlds) - 1)) if worlds else 0
    world_text = worlds[world_index] if worlds else text

    ticks = parse_protocol_to_ticks(world_text)
    if not ticks:
        emit("error", {"message": "No ticks parsed from file."})
        return

    emit("simulation_info", {
        "total_ticks": len(ticks),
        "file": os.path.basename(filepath),
        "world_index": world_index,
        "world_count": len(worlds),
    })

    # Emit biome data from first tick that has it (patches empty — frontend generates)
    for t in ticks:
        if t.biomes:
            emit("environment_init", {
                "biomes": [serialize_biome_for_frontend(b) for b in t.biomes],
            })
            break

    for tick in ticks:
        if not _wait_or_abort(my_gen):
            return

        species_data = [serialize_species_for_frontend(sp) for sp in tick.species]
        agent_data = [serialize_agent_for_frontend(agent) for agent in tick.agents]
        event_data = [serialize_event_for_frontend(evt) for evt in tick.events]

        tick_payload = {
            "tick": tick.number,
            "epoch": tick.epoch,
            "species": species_data,
            "agents": agent_data,
            "interpolate_duration": 1000 / TICK_RATE,
        }
        if event_data:
            tick_payload["events"] = event_data
        emit("tick_update", tick_payload)

        socketio.sleep(1.0 / (TICK_RATE * current_speed))

    if _sim_gen == my_gen:
        emit("simulation_complete", {"total_ticks": len(ticks)})


def _classify_live_event(target: str, raw: str) -> dict | None:
    """Convert a raw event string from the simulation engine into a frontend event dict.

    Event strings look like: "catastrophe:meteor_impact|dur=5", "disease:plague|pop-=150",
    "symbiogenesis:S0+S1->S2|gained=speed,armor", "evo_trap:low_variance",
    "catastrophe_end:meteor_impact", "extinction:starvation", etc.
    """
    if ":" not in raw:
        return None
    event_type, _, detail = raw.partition(":")
    return {"target": target, "event_type": event_type, "detail": detail}


@socketio.on("start_live")
def handle_start_live(data=None):
    """Run a live simulation and stream ticks in real-time."""
    global is_playing, _sim_gen
    _sim_gen += 1
    my_gen = _sim_gen
    is_playing = True

    data = data or {}
    max_gens = data.get("max_generations", 100)
    seed = data.get("seed", 42)
    enable_agents = data.get("enable_agents", True)
    agent_epoch = data.get("agent_epoch", "ECOSYSTEM")

    # Import simulation
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if base not in sys.path:
        sys.path.insert(0, base)

    import importlib
    import types
    sim_path = os.path.join(base, "mantis", "simulation")
    if "mantis" not in sys.modules:
        pkg = types.ModuleType("mantis")
        pkg.__path__ = [os.path.join(base, "mantis")]
        pkg.__package__ = "mantis"
        sys.modules["mantis"] = pkg

    spec = importlib.util.spec_from_file_location(
        "mantis.simulation", os.path.join(sim_path, "__init__.py"),
        submodule_search_locations=[sim_path],
    )
    sim_mod = importlib.util.module_from_spec(spec)
    sys.modules["mantis.simulation"] = sim_mod
    spec.loader.exec_module(sim_mod)

    World = sim_mod.World
    Serializer = sim_mod.Serializer

    world = World(0, seed, enable_agents=enable_agents, agent_epoch=agent_epoch)
    serializer = Serializer(keyframe_interval=20)

    emit("simulation_info", {
        "total_ticks": max_gens,
        "mode": "live",
    })

    # Initialize vegetation patches and emit biome data with real patch positions
    biome_data = []
    for biome in world.biomes:
        if not biome.vegetation_patches:
            biome.init_vegetation_patches(world.rng, world_size=1000)
        patches = [
            {"x": p.x, "y": p.y, "radius": p.radius, "density": p.density}
            for p in biome.vegetation_patches
        ]
        biome_data.append({
            "lid": biome.lid,
            "name": biome.name,
            "vegetation": biome.vegetation,
            "detritus": biome.detritus,
            "nitrogen": getattr(biome, "nitrogen", 0.5),
            "phosphorus": getattr(biome, "phosphorus", 0.3),
            "patches": patches,
        })
    emit("environment_init", {"biomes": biome_data})

    veg_update_counter = 0
    for gen in range(max_gens):
        if not _wait_or_abort(my_gen):
            return

        world.step()

        # Collect species data
        species_data = []
        agent_data = []
        for sp in world.species:
            if not sp.alive:
                continue
            species_data.append({
                "sid": sp.sid,
                "plan": sp.body_plan.name,
                "population": sp.population,
                "locations": list(sp.locations),
            })
            if sp.agent_manager is not None:
                for a in sp.agent_manager.agents:
                    agent_data.append({
                        "aid": a.aid,
                        "species_sid": sp.sid,
                        "x": a.x,
                        "y": a.y,
                        "energy": a.energy,
                        "age": a.age,
                        "state": a.state,
                        "target_aid": a.target_aid,
                        "dead": not a.alive,
                    })

        # Collect events from world
        event_data = []
        for raw in getattr(world, "world_events", []):
            evt = _classify_live_event("WORLD", raw)
            if evt:
                event_data.append(evt)
        for sid, evts in getattr(world, "events", {}).items():
            for raw in evts:
                evt = _classify_live_event(f"S{sid}", raw)
                if evt:
                    event_data.append(evt)

        tick_payload = {
            "tick": world.tick,
            "epoch": world.epoch.value,
            "species": species_data,
            "agents": agent_data,
            "interpolate_duration": 1000 / TICK_RATE,
        }
        if event_data:
            tick_payload["events"] = event_data
        emit("tick_update", tick_payload)

        # Periodic vegetation density update
        veg_update_counter += 1
        if veg_update_counter >= 10:
            veg_update_counter = 0
            patch_updates = []
            for biome in world.biomes:
                for p in biome.vegetation_patches:
                    patch_updates.append({
                        "lid": biome.lid,
                        "x": p.x, "y": p.y,
                        "density": p.density,
                    })
            emit("vegetation_update", {"patches": patch_updates})

        socketio.sleep(1.0 / (TICK_RATE * current_speed))

    if _sim_gen == my_gen:
        emit("simulation_complete", {"total_ticks": world.tick})


@socketio.on("list_datasets")
def handle_list_datasets():
    """Scan data/ directory and emit available dataset files."""
    data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    files = []
    if os.path.isdir(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(".txt"):
                fpath = os.path.join(data_dir, f)
                files.append({"name": f, "size": os.path.getsize(fpath)})
    emit("dataset_list", files)


@socketio.on("list_worlds")
def handle_list_worlds(data):
    """Read a dataset file and report how many worlds it contains."""
    data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    filename = os.path.basename(data.get("file", ""))
    filepath = os.path.realpath(os.path.join(data_dir, filename))
    if not filepath.startswith(data_dir) or not os.path.exists(filepath):
        emit("error", {"message": f"File not found: {filename}"})
        return
    with open(filepath) as f:
        text = f.read()
    worlds = split_worlds(text)
    worlds_with_agents = [i for i, w in enumerate(worlds) if "@AGENT" in w]
    emit("world_list", {
        "file": filename,
        "world_count": len(worlds),
        "worlds_with_agents": worlds_with_agents,
    })


@socketio.on("list_models")
def handle_list_models():
    """Scan checkpoints/ directory for .pt files."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ckpt_dir = os.path.join(base, "checkpoints")
    files = []
    if os.path.isdir(ckpt_dir):
        for root, _dirs, fnames in os.walk(ckpt_dir):
            for fname in sorted(fnames):
                if fname.endswith(".pt"):
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, ckpt_dir)
                    files.append({"name": rel, "size": os.path.getsize(fpath)})
    emit("model_list", files)


def _build_seed_prompt() -> str:
    """Return a minimal v1 protocol seed to prime model generation."""
    return (
        "=EPOCH:1|TICK_SCALE:1000gen|W0\n"
        "@SP|S0|plan=herbivore|pop=120|L0,L1\n"
    )


def _get_engine(checkpoint_rel: str):
    """Lazily load InferenceEngine, caching by checkpoint path."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ckpt_dir = os.path.join(base, "checkpoints")
    full_path = os.path.realpath(os.path.join(ckpt_dir, checkpoint_rel))
    if not full_path.startswith(os.path.realpath(ckpt_dir)):
        raise ValueError("Invalid checkpoint path")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_rel}")

    if _engine_cache["path"] == full_path and _engine_cache["engine"] is not None:
        return _engine_cache["engine"]

    # Lazy import to avoid torch/nvidia-smi overhead at server startup
    sys.path.insert(0, base) if base not in sys.path else None
    from inference import InferenceEngine
    engine = InferenceEngine(full_path)
    _engine_cache["path"] = full_path
    _engine_cache["engine"] = engine
    return engine


@socketio.on("start_model")
def handle_start_model(data=None):
    """Run model inference and stream parsed ticks to client."""
    global is_playing, _sim_gen
    _sim_gen += 1
    my_gen = _sim_gen
    is_playing = True

    data = data or {}
    model_name = data.get("model", "")
    temperature = float(data.get("temperature", 0.8))
    max_tokens = int(data.get("max_tokens", 4096))

    try:
        engine = _get_engine(model_name)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        emit("error", {"message": str(e)})
        return

    seed_prompt = _build_seed_prompt()
    text_buffer = seed_prompt

    emit("simulation_info", {"total_ticks": 0, "mode": "model"})

    ticks_sent = 0
    biomes_sent = False
    for token_text in engine.generate_streaming(
        prompt=seed_prompt,
        max_length=max_tokens,
        temperature=temperature,
    ):
        if not _wait_or_abort(my_gen):
            return

        text_buffer += token_text

        # Buffer overflow guard
        if len(text_buffer) > 10000:
            break

        # Try to parse completed ticks (delimited by ---)
        if "---" not in text_buffer:
            continue

        # Split on --- and keep the incomplete tail
        parts = text_buffer.split("---")
        incomplete = parts[-1]
        completed_chunks = parts[:-1]

        for chunk in completed_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            ticks = parse_protocol_to_ticks(chunk + "\n---")
            for tick_data in ticks:
                # Emit biome data from first tick that has it
                if tick_data.biomes and not biomes_sent:
                    emit("environment_init", {
                        "biomes": [serialize_biome_for_frontend(b) for b in tick_data.biomes],
                    })
                    biomes_sent = True
                species_data = [serialize_species_for_frontend(sp) for sp in tick_data.species]
                agent_data = [serialize_agent_for_frontend(agent) for agent in tick_data.agents]
                event_data = [serialize_event_for_frontend(evt) for evt in tick_data.events]
                model_tick = {
                    "tick": ticks_sent,
                    "epoch": tick_data.epoch,
                    "species": species_data,
                    "agents": agent_data,
                    "interpolate_duration": 1000 / TICK_RATE,
                }
                if event_data:
                    model_tick["events"] = event_data
                emit("tick_update", model_tick)
                ticks_sent += 1
                socketio.sleep(1.0 / (TICK_RATE * current_speed))

                if not _wait_or_abort(my_gen):
                    return

        text_buffer = incomplete

    if _sim_gen == my_gen:
        emit("simulation_complete", {"total_ticks": ticks_sent})


@socketio.on("pause")
def handle_pause():
    global is_playing
    is_playing = False


@socketio.on("resume")
def handle_resume():
    global is_playing
    is_playing = True


@socketio.on("set_speed")
def handle_speed(data):
    global current_speed
    current_speed = max(0.1, min(20.0, data.get("speed", 1.0)))


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
