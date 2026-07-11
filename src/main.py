from logging import currentframe

import cv2

import config

#Load videos
video_path = config.SAMPLE_VIDEOS_DIR / "test.MOV" #name of video file in string
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

def sharpness_score(frame):#Sharpness score function
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def save_image(frame_name):
    frames_dir = config.FRAMES_DIR / f"frame_{current_frame:04d}.jpg"  # directory of saving and name the file
    cv2.imwrite(str(frames_dir), frame_name)  # save current working frame


saved_frames = 0
current_frame = 0
Threshold = 15
previous_frame = None

motion = False
stable_count = 0
stable_cooldown = 2 #2 frames cooldown
frame_save_count = 3

candidate_frames = []

frame_interval = int(fps * 0.2)

while True:
    success, frame = cap.read()
    if not success:# no frames left
        break

    if current_frame % frame_interval == 0: #Cut down initial huge frames by frame interval

        save_frame = False

        if previous_frame is None: #if there is no previous frame then save it ofc
            save_image(frame)
            previous_frame = frame.copy()
            current_frame += 1
            continue

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
                        # STABLE , Take img
                        # save until frame_save_count is 0
                        save_frame = True
                        frame_save_count -= 1 # one is saved
                        #reset
                        if frame_save_count == 0:
                            motion = False
                            stable_count = 0
                            frame_save_count = 3 # initial save count
    #NEXT: compare images, crop it  and add effects and convert to pdf file


        if save_frame:
            #print(f"Saving frame: {current_frame}")
            #frames_dir = config.FRAMES_DIR/f"frame_{current_frame:04d}.jpg" # directory of saving and name the file
            #cv2.imwrite(str(frames_dir), frame)# save current working frame
            candidate_frames.append(frame.copy())

            if len(candidate_frames) == 3 :
                best_frame = None
                best_score = -1

                for candidate in candidate_frames:

                    score = sharpness_score(candidate)
                    print(score)

                    if score > best_score:
                        best_score = score
                        best_frame = candidate

                print(f"Best score: {best_score:.2f}")
                save_image(best_frame)#save this frame
                saved_frames += 1

                candidate_frames.clear()

        previous_frame = frame.copy() # stores copy of saved frame to compare with next one

    current_frame += 1



cap.release()

print(f"Total saved frames: {saved_frames}")