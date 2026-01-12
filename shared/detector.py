import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import cv2
from collections import deque
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from baseline_model.models.lstm_model import LSTMModel

# -----------------------------
# Load CNN (Feature Extractor)
# -----------------------------
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()   # 512-D features
resnet.eval()

# -----------------------------
# Load LSTM
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "baseline_model",
    "models",
    "violence_lstm.pt"
)

lstm = LSTMModel()
lstm.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
lstm.eval()

# -----------------------------
# Buffers (TEMP: one global buffer)
# -----------------------------
SEQ_LEN = 16
frame_buffer = deque(maxlen=SEQ_LEN)

# -----------------------------
# Image Transform
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
# Inference Function
# -----------------------------
def process_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = resnet(img).squeeze().numpy()
        frame_buffer.append(feat)

    if len(frame_buffer) < SEQ_LEN:
        return {"status": "warming_up"}

    seq = np.array(frame_buffer)
    seq = torch.from_numpy(seq).unsqueeze(0).float()

    with torch.no_grad():
        out = lstm(seq)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    label = "VIOLENCE" if pred.item() == 1 else "NORMAL"

    return {
        "label": label,
        "confidence": float(conf.item())
    }
