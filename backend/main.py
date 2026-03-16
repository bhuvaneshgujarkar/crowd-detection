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
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(2)
    else:
        cap = cv2.VideoCapture(video_path)

    print("Camera opened:", cap.isOpened())

    return {"status": "started"}


# -------- STREAM FRAME --------
@app.get("/frame")
def get_frame():
    global cap

    if cap is None or not cap.isOpened():
        return {"status": "no_camera"}

    ret, frame = cap.read()

    print("Frame read:", ret)

    if not ret:
        return {"status": "no_frame"}

    # -------- YOLO + VIOLENCE DETECTION --------
    frame, person_count = detect_and_classify(frame)

    # -------- ENCODE FRAME --------
    _, buffer = cv2.imencode(".jpg", frame)
    frame_base64 = base64.b64encode(buffer).decode("utf-8")

    # -------- API RESPONSE --------
    return {
        "status": "ok",
        "frame": frame_base64,
        "person_count": person_count,
        "label": last_label,
        "confidence": last_conf
    }