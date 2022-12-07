import Recorder from "voice-recorder-react";

import RecorderUI from "./RecorderUI";

export default function App() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh"
      }}
    >
      <h3>Accent Classifier</h3>
      <Recorder Render={RecorderUI} />
    </div>
  );
}
