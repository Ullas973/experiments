import React from "react";

export default function VideoModal({ videoUrl, onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <button className="close" onClick={onClose}>✖</button>
        <video controls autoPlay style={{ width: "100%", borderRadius: 10 }}>
          <source src={videoUrl} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div style={{ marginTop: 8, color: "#cfefff", fontSize: 13 }}>
          Tip: Use headphones for best narration clarity.
        </div>
      </div>
    </div>
  );
}
