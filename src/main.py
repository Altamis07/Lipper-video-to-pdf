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
Threshold = 30
previous_frame = None

motion = False
stable_count = 0
stable_cooldown = 2 #2 frames cooldown
frame_save_count = 5

frame_interval = int(fps * 0.2)

while True:
    success, frame = cap.read()
    if not success:
        break

    if current_frame % frame_interval == 0: #Cut down initial huge frames by frame interval

        save_frame = False

        if previous_frame is None: #if there is no previous frame then save it ofc
            save_frame = True

        else:
            difference = cv2.absdiff(frame, previous_frame)
            change_score = difference.mean()

            # Output of compared frames
            if change_score > Threshold:
                #Video is moving Motion detect logic
                motion = True
                stable_count = 0
            else:
                #video is still
                #previous frame was true so motion just stopped
                if motion:
                    stable_count += 1
                    if stable_count > stable_cooldown and frame_save_count != 0:
                        # Take img
                        save_frame = True
                        frame_save_count -= 1 # one is saved
                        #reset
                        if frame_save_count == 0:
                            motion = False
                            stable_count = 0
                            frame_save_count = 5



        if save_frame:
            print(f"Saving frame: {current_frame}")
            frames_dir = config.FRAMES_DIR/f"frame_{current_frame:04d}.jpg" # directory of saving and name the file
            cv2.imwrite(str(frames_dir), frame)# save current working frame

            saved_frames += 1
        previous_frame = frame.copy() # stores copy of saved frame to compare with next one

    current_frame += 1



cap.release()

print(f"Total saved frames: {saved_frames}")