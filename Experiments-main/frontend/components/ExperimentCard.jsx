import React from "react";

export default function ExperimentCard({ exp, onGenerate, generating, status }) {
  // normalize title/description/video fields from various CSV shapes
  const title = exp["Experiment Title"] || exp["ExperimentName"] || exp["Experiment"] || exp["title"] || exp["ExperimentName "] || `Experiment ${exp.ID ?? exp.id ?? exp.ExpNo}`;
  const desc = exp["Outcome / Observation"] || exp["Description"] || exp["Procedure / Steps"] || exp["Procedure"] || exp["description"] || "";

  return (
    <div className="card">
      <div className="card-header">
        <h3>{title}</h3>
        <div className="id-badge">#{exp.ID ?? exp.id ?? exp.ExpNo ?? ""}</div>
      </div>

      <p className="desc">{desc.length > 180 ? desc.slice(0, 180) + "…" : desc}</p>

      <div className="card-footer">
        <button className="generate-btn" onClick={onGenerate} disabled={generating}>
          {generating ? "Generating…" : "Generate Video"}
        </button>

        <div className="status">
          {status && <small>{status}</small>}
        </div>
      </div>
    </div>
  );
}
