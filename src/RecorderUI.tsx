import { RecorderProps } from "voice-recorder-react";
import { useEffect, useRef, useState } from "react";

// Recorder UI component
export default function RecorderUI({
  time,
  stop,
  data,
  start,
  pause,
  resume,
  paused,
  recording
}: RecorderProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [hasRecording, setHasRecording] = useState(false);

  const togglePlay = () => {
    if (audioRef.current?.paused) {
      audioRef.current?.play();
    } else {
      audioRef.current?.pause();
    }
  };
  console.log(data.url);
  useEffect(() => {
    if (data.url && audioRef.current) {
      audioRef.current.src = data.url;
    }
  }, [data.url]);

  return (
    <div style={{ display: "flex" }}>
      <button
        type="button"
        onClick={() => {
          if (recording) {
            stop();
            setHasRecording(true);
          } else {
            start();
            setHasRecording(false);
          }
        }}
        style={{
          margin: "3px",
          backgroundColor: "#BC09C7",
          border: "2px solid #BC09C7",
          color: "white",
          padding: "10px 20px",
          borderRadius: "50px"
        }}
      >
        {recording ? "Stop" : "Start Speaking"}
      </button>

      {recording && (
        <>
          <button
            type="button"
            onClick={() => {
              if (recording) {
                if (paused) resume();
                else pause();
              }
            }}
            style={{
              margin: "3px",
              backgroundColor: "white",
              border: "2px solid #BC09C7",
              color: "#BC09C7",
              padding: "10px 20px",
              borderRadius: "50px"
            }}
          >
            {paused ? "Resume" : "Pause"}
          </button>
          <p>
            {time.h}:{time.m}:{time.s}
          </p>
        </>
      )}

      {!recording && hasRecording && (
        <>
          <br />
          <br />
          <button
            type="button"
            onClick={togglePlay}
            style={{
              margin: "3px",
              backgroundColor: "white",
              border: "2px solid #BC09C7",
              color: "#BC09C7",
              padding: "10px 20px",
              borderRadius: "50px"
            }}
          >
            Play/Pause
          </button>
        </>
      )}

      <audio ref={audioRef} hidden />
    </div>
  );
}
