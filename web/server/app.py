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
)

app = Flask(__name__, static_folder=None)
socketio = SocketIO(app, cors_allowed_origins="*")

is_playing = False
current_speed = 1.0
TICK_RATE = 15  # Server updates at 15 ticks/sec


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
    global is_playing
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

    ticks = parse_protocol_to_ticks(text)
    if not ticks:
        emit("error", {"message": "No ticks parsed from file."})
        return

    emit("simulation_info", {
        "total_ticks": len(ticks),
        "file": os.path.basename(filepath),
    })

    for tick in ticks:
        if not is_playing:
            break

        species_data = [serialize_species_for_frontend(sp) for sp in tick.species]

        # Agents: group by the @SP context they appeared under
        agent_data = []
        for sp in tick.species:
            for agent in tick.agents:
                agent_data.append(serialize_agent_for_frontend(agent, sp.sid))

        emit("tick_update", {
            "tick": tick.number,
            "epoch": tick.epoch,
            "species": species_data,
            "agents": agent_data,
            "interpolate_duration": 1000 / TICK_RATE,
        })

        socketio.sleep(1.0 / (TICK_RATE * current_speed))

    emit("simulation_complete", {"total_ticks": len(ticks)})


@socketio.on("start_live")
def handle_start_live(data=None):
    """Run a live simulation and stream ticks in real-time."""
    global is_playing
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

    for gen in range(max_gens):
        if not is_playing:
            break

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

        emit("tick_update", {
            "tick": world.tick,
            "epoch": world.epoch.value,
            "species": species_data,
            "agents": agent_data,
            "interpolate_duration": 1000 / TICK_RATE,
        })

        socketio.sleep(1.0 / (TICK_RATE * current_speed))

    emit("simulation_complete", {"total_ticks": world.tick})


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
