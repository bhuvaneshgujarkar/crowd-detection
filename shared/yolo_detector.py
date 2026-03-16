from shared.alert_system import trigger_violence_alert
from shared.crowd_detector import detect_crowd
from ultralytics import YOLO
import cv2
from shared.detector import process_frame

yolo_model = YOLO("yolov8n.pt")

frame_count = 0
last_label = "NORMAL"
last_conf = 0

def detect_and_classify(frame):
    global frame_count, last_label, last_conf

    frame_count += 1
    person_count = 0   # NEW: count persons

    results = yolo_model(frame, imgsz=320, conf=0.4, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])

        if cls == 0:  # person
            person_count += 1  # NEW: increment counter

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]

            # Run CNN+LSTM only every 5 frames
            if frame_count % 5 == 0:
                result = process_frame(person_crop)
                last_label = result.get("label", "NORMAL")
                last_conf = result.get("confidence", 0)

            color = (0,0,255) if last_label=="VIOLENCE" else (0,255,0)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{last_label} {last_conf:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------------------------
    # CROWD DETECTION
    # -------------------------

    crowd_risk, crowd_status = detect_crowd(person_count)

    cv2.putText(frame,
                f"Persons: {person_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                2)

    cv2.putText(frame,
                f"Status: {crowd_status}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255) if crowd_risk else (0,255,0),
                2)

    return frame,person_count 