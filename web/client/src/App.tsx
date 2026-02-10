import { useWebSocket } from "./hooks/useWebSocket";
import { SimulationCanvas } from "./components/SimulationCanvas";
import { Controls } from "./components/Controls";
import { SpeciesPanel } from "./components/SpeciesPanel";

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
    selectedWorld,
    selectFile,
    selectWorld,
    models,
    selectedModel,
    selectModel,
  } = useWebSocket();

  return (
    <div style={{ padding: "16px", maxWidth: "1120px", margin: "0 auto" }}>
      <h1
        style={{
          fontSize: "20px",
          marginBottom: "12px",
          color: "#e94560",
          fontFamily: "monospace",
        }}
      >
        MANTIS Playground
      </h1>
      <div style={{ display: "flex", gap: "12px" }}>
        <div style={{ flex: 1 }}>
          <Controls
            onPlay={play}
            onPause={pause}
            onResume={resume}
            onSpeed={setSpeed}
            isPlaying={isPlaying}
            datasets={datasets}
            selectedFile={selectedFile}
            worldCount={worldCount}
            selectedWorld={selectedWorld}
            onSelectFile={selectFile}
            onSelectWorld={selectWorld}
            models={models}
            selectedModel={selectedModel}
            onSelectModel={selectModel}
          />
          <SimulationCanvas
            agents={agents}
            species={species}
            worldSize={1000}
            tick={tick}
            epoch={epoch}
            interpolateDuration={interpolateDuration}
            isPlaying={isPlaying}
          />
        </div>
        <SpeciesPanel species={species} />
      </div>
    </div>
  );
}

export default App;
