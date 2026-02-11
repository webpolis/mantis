import { useWebSocket } from "./hooks/useWebSocket";
import { SimulationCanvas } from "./components/SimulationCanvas";
import { Controls } from "./components/Controls";
import { SpeciesPanel } from "./components/SpeciesPanel";
import { Timeline } from "./components/Timeline";
import { EventLog } from "./components/EventLog";
import { EpochIndicator } from "./components/EpochIndicator";

const glassPanel: React.CSSProperties = {
  background: "rgba(10, 10, 20, 0.7)",
  backdropFilter: "blur(12px)",
  WebkitBackdropFilter: "blur(12px)",
  border: "1px solid rgba(255, 255, 255, 0.08)",
  borderRadius: "8px",
};

function App() {
  const {
    tick,
    epoch,
    species,
    agents,
    isPlaying,
    interpolateDuration,
    play,
    pause,
    resume,
    setSpeed,
    datasets,
    selectedFile,
    worldCount,
    worldsWithAgents,
    worldsWithSpotlights,
    selectedWorld,
    selectFile,
    selectWorld,
    models,
    selectedModel,
    selectModel,
    biomes,
    events,
    eventLog,
    historyLength,
    viewIndex,
    isFollowing,
    seekTo,
    followLatest,
    epochs,
    populationHistory,
  } = useWebSocket();

  return (
    <div style={{ width: "100vw", height: "100vh", position: "relative", overflow: "hidden" }}>
      {/* Full-screen PixiJS canvas */}
      <SimulationCanvas
        agents={agents}
        species={species}
        worldSize={1000}
        tick={tick}
        epoch={epoch}
        interpolateDuration={interpolateDuration}
        isPlaying={isPlaying}
        biomes={biomes}
        events={events}
      />

      {/* HUD: top-left — epoch indicator */}
      <div style={{ position: "absolute", top: 16, left: 16, zIndex: 10 }}>
        <EpochIndicator epoch={epoch} tick={tick} agentCount={agents.filter(a => !a.dead).length} speciesCount={species.length} />
      </div>

      {/* HUD: top-center — controls */}
      <div style={{ position: "absolute", top: 16, left: "50%", transform: "translateX(-50%)", zIndex: 10, ...glassPanel, padding: "8px 16px" }}>
        <Controls
          onPlay={play}
          onPause={pause}
          onResume={resume}
          onSpeed={setSpeed}
          isPlaying={isPlaying}
          datasets={datasets}
          selectedFile={selectedFile}
          worldCount={worldCount}
          worldsWithAgents={worldsWithAgents}
          worldsWithSpotlights={worldsWithSpotlights}
          selectedWorld={selectedWorld}
          onSelectFile={selectFile}
          onSelectWorld={selectWorld}
          models={models}
          selectedModel={selectedModel}
          onSelectModel={selectModel}
        />
      </div>

      {/* Right side — species panel + event log */}
      <div style={{ position: "absolute", top: 16, right: 16, zIndex: 10, display: "flex", flexDirection: "column", gap: 8, maxHeight: "calc(100vh - 100px)" }}>
        <div style={{ ...glassPanel, padding: "10px 12px", maxHeight: "55vh", overflowY: "auto" }}>
          <SpeciesPanel species={species} populationHistory={populationHistory} />
        </div>
        <div style={{ ...glassPanel, padding: "10px 12px", maxHeight: "35vh", overflowY: "auto" }}>
          <EventLog eventLog={eventLog} />
        </div>
      </div>

      {/* Bottom — timeline */}
      <div style={{ position: "absolute", bottom: 16, left: 180, right: 16, zIndex: 10, ...glassPanel, padding: "8px 16px" }}>
        <Timeline
          historyLength={historyLength}
          viewIndex={viewIndex}
          currentEpoch={epoch}
          isFollowing={isFollowing}
          epochs={epochs}
          onSeek={seekTo}
          onFollowLatest={followLatest}
        />
      </div>
    </div>
  );
}

export default App;
