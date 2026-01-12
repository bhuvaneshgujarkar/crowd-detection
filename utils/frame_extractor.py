import cv2
import os

# Paths
RAW_VIDEO_DIR = "data/raw_videos"
FRAME_OUTPUT_DIR = "data/frames"

FRAME_INTERVAL = 5   # extract every 5th frame
IMG_SIZE = (224, 224)

def extract_frames(category):
    video_dir = os.path.join(RAW_VIDEO_DIR, category)
    save_dir = os.path.join(FRAME_OUTPUT_DIR, category)
    os.makedirs(save_dir, exist_ok=True)

    for video_file in os.listdir(video_dir):
        if not video_file.lower().endswith((".mp4",".avi")):
            continue

        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        saved_count = 0
        video_name = video_file.split(".")[0]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_INTERVAL == 0:
                frame = cv2.resize(frame, IMG_SIZE)
                frame_name = f"{video_name}_frame_{saved_count}.jpg"
                cv2.imwrite(os.path.join(save_dir, frame_name), frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"[INFO] Extracted {saved_count} frames from {video_file}")

if __name__ == "__main__":
    extract_frames("normal")
    extract_frames("violence")
