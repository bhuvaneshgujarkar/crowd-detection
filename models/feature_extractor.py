import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# Paths
FRAME_DIR = "data/frames"
FEATURE_DIR = "features"

# Device (CPU is fine)
device = torch.device("cpu")

# Load pretrained ResNet18
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()  # remove classifier
resnet = resnet.to(device)
resnet.eval()

# Image preprocessing (ImageNet standard)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(category):
    input_dir = os.path.join(FRAME_DIR, category)
    output_dir = os.path.join(FEATURE_DIR, category)
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if not img_name.endswith(".jpg"):
            continue

        img_path = os.path.join(input_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = resnet(image)

        feature_path = os.path.join(
            output_dir, img_name.replace(".jpg", ".npy")
        )
        np.save(feature_path, features.cpu().numpy())

    print(f"[INFO] Features extracted for {category}")

if __name__ == "__main__":
    extract_features("normal")
    extract_features("violence")
