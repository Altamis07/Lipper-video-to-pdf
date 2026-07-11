from logging import currentframe

import cv2
import config

#Load videos
video_path = config.SAMPLE_VIDEOS_DIR / "test.mp4" #name of video file in string
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

#Extract Frames

saved_frames = 0
current_frame = 0

frame_interval = int(fps * 0.5)

while True:
    success, frame = cap.read()
    if not success:
        break

    if current_frame % frame_interval == 0:
        frames_dir = config.FRAMES_DIR/f"frame_{current_frame:04d}.jpg" # directory of saving and name the file
        cv2.imwrite(frames_dir, frame)# save current working frame
        saved_frames += 1

    current_frame += 1

cap.release()

print(f"Total saved frames: {saved_frames}")