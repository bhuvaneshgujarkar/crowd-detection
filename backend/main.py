from fastapi import FastAPI
import cv2
import base64
import time
import winsound
from shared.detector import process_frame

app = FastAPI()

cap = None
last_alert_time = 0


@app.post("/start")
def start(video_path: str):
    global cap
    cap = cv2.VideoCapture(video_path)
    return {"status": "started"}


@app.get("/frame")
def get_frame():
    global cap, last_alert_time

    if cap is None:
        return {"status": "not_started"}

    ret, frame = cap.read()
    if not ret:
        return {"status": "ended"}

    result = process_frame(frame)

    label = result.get("label", "WARMING UP")
    confidence = result.get("confidence", 0)

    # Draw label
    color = (0, 255, 0) if label == "NORMAL" else (0, 0, 255)
    cv2.putText(frame, f"{label} {confidence:.2f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Alert system (limit beeps to once every 3 seconds)
    if label == "VIOLENCE":
        now = time.time()
        if now - last_alert_time > 3:
            winsound.Beep(1200, 600)
            last_alert_time = now
            with open("alerts_log.txt", "a") as f:
                f.write(f"VIOLENCE detected at {time.ctime()}\n")

    # Encode frame
    _, buffer = cv2.imencode(".jpg", frame)
    frame_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": "ok",
        "label": label,
        "confidence": confidence,
        "frame": frame_base64
    }
