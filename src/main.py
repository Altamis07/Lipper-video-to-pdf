import cv2
from pathlib import  Path
import config
from config import SAMPLE_VIDEOS_DIR

video_path = SAMPLE_VIDEOS_DIR/"test.mp4"
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Could not open video file")
    exit(0)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps

print("✅ Video Loaded")
print(f"FPS: {fps}")
print(f"Frames: {frame_count}")
print(f"Duration: {duration:.2f} seconds")