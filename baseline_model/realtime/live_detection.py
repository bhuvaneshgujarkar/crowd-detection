import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

from models.lstm_model import LSTMModel

# -----------------------------
# CONFIG
# -----------------------------
SEQ_LEN = 16
CONFIDENCE_THRESHOLD = 0.6
PREDICTION_SMOOTHING = 10  # number of recent predictions to vote

# -----------------------------
# DEVICE (CPU)
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# LOAD RESNET18 (FEATURE EXTRACTOR)
# -----------------------------
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()          # 🔴 VERY IMPORTANT (512-D features)
resnet.eval()

# -----------------------------
# LOAD LSTM MODEL
# -----------------------------
lstm = LSTMModel()
lstm.load_state_dict(
    torch.load("models/violence_lstm.pt", map_location=device)
)
lstm.eval()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# BUFFERS
# -----------------------------
frame_buffer = deque(maxlen=SEQ_LEN)
prediction_buffer = deque(maxlen=PREDICTION_SMOOTHING)

# -----------------------------
# VIDEO SOURCE
# -----------------------------
# Option 1: Webcam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Option 2: Video file (better demo)
cap = cv2.VideoCapture(r"data\raw_videos\Normal\5968254-uhd_3840_2160_30fps.mp4")

# -----------------------------
# MAIN LOOP
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)

    # Extract CNN features
    with torch.no_grad():
        features = resnet(img).squeeze().numpy()  # (512,)
        frame_buffer.append(features)

    label = "WARMING UP..."
    color = (255, 255, 0)

    # Run LSTM when enough frames
    if len(frame_buffer) == SEQ_LEN:
        seq_np = np.array(frame_buffer)                # (16, 512)
        seq = torch.from_numpy(seq_np).unsqueeze(0).float()  # (1,16,512)

        with torch.no_grad():
            output = lstm(seq)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        pred = pred.item()
        confidence = confidence.item()

        # Confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            pred = 0  # force NORMAL

        # Temporal smoothing
        prediction_buffer.append(pred)
        final_pred = max(set(prediction_buffer),
                         key=prediction_buffer.count)

        if final_pred == 1:
            label = f"VIOLENCE ({confidence:.2f})"
            color = (0, 0, 255)
        else:
            label = f"NORMAL ({confidence:.2f})"
            color = (0, 255, 0)

    # Display
    cv2.putText(frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("CCTV Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
