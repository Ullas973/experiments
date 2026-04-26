import { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [experiments, setExperiments] = useState([]);
  const [selectedExpId, setSelectedExpId] = useState("");
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState("");

  // Load experiments list on component mount
  useEffect(() => {
    fetch("http://127.0.0.1:8000/experiments")
      .then((res) => res.json())
      .then((data) => {
        setExperiments(data.experiments || []);
      })
      .catch((err) => {
        console.error("Failed to load experiments list:", err);
      });
  }, []);

  const handleGenerate = async () => {
  if (!selectedExpId) return alert("Please select an experiment");

  const experimentIdInt = parseInt(selectedExpId, 10);
  if (isNaN(experimentIdInt)) {
    alert("Invalid experiment selected");
    return;
  }

  setLoading(true);
  setVideoUrl("");

  try {
    console.log("Sending experiment_id:", experimentIdInt);

    const res = await fetch("http://127.0.0.1:8000/generate_video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ experiment_id: experimentIdInt }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP error ${res.status}`);
    }

    const data = await res.json();
    setVideoUrl(data.video_path ? `http://127.0.0.1:8000${data.video_path}` : "");
  } catch (err) {
    console.error(err);
    alert("Error generating video: " + err.message);
  }

  setLoading(false);
};


  return (
    <div className="app-container">
      <h1>🔬 AI Science Lab</h1>
      <p>Select a science experiment to generate a video.</p>

      <select
        value={selectedExpId}
        onChange={(e) => setSelectedExpId(e.target.value)}
      >
        <option key="select-an-experiment" value="">-- Select an experiment --</option>
        {experiments.map((exp, index) => (
          <option key={`${exp.id}-${index}`} value={exp.id}>
            {exp.title}
          </option>
        ))}
      </select>

      <button onClick={handleGenerate} disabled={loading}>
        {loading ? "Generating..." : "Generate Experiment Video"}
      </button>

      {videoUrl && (
        <div className="video-container" style={{ marginTop: 20 }}>
          <h3>Experiment Video:</h3>
          <video controls src={videoUrl} width="640" height="360" />
        </div>
      )}
    </div>
  );
}

export default App;
