from fastapi import FastAPI
import cv2
import base64
import time
from shared.yolo_detector import detect_and_classify
from shared.yolo_detector import last_label, last_conf

app = FastAPI()

cap = None


# -------- START VIDEO --------
@app.post("/start")
def start(video_path: str):
    global cap

    if video_path == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    # 🔥 IMPORTANT: reduce buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Camera opened:", cap.isOpened())

    return {"status": "started"}


# -------- STREAM FRAME --------
@app.get("/frame")
def get_frame():
    global cap

    if cap is None or not cap.isOpened():
        return {"status": "no_camera"}

    # 🔥 DROP OLD FRAMES (VERY IMPORTANT)
    for _ in range(2):
        cap.grab()

    ret, frame = cap.read()

    if not ret:
        return {"status": "no_frame"}

    # 🔥 Resize (balanced)
    frame = cv2.resize(frame, (640, 360))

    # -------- YOLO + VIOLENCE --------
    frame, person_count = detect_and_classify(frame)

    # -------- ENCODE --------
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    frame_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": "ok",
        "frame": frame_base64,
        "person_count": person_count,
        "label": last_label,
        "confidence": last_conf
    }