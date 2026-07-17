import cv2
import numpy as np
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

#---Extract Frames---

def sharpness_score(in_frame):#Sharpness score function
    gray = cv2.cvtColor(in_frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def save_image(frame_name, frame_number):
    frames_dir = config.FRAMES_DIR / f"frame_{frame_number:04d}.jpg"  # directory of saving and name the file
    cv2.imwrite(str(frames_dir), frame_name)  # save current working frame


def get_sharpest_frame(candidate_frames_grp):
    best_frame = None
    best_score = -1

    for candidate in candidate_frames_grp:

        score = sharpness_score(candidate)
        print(score)

        if score > best_score:
            best_score = score
            best_frame = candidate

    return best_frame,best_score

def calculate_change_score(now_frame, before_frame):
    difference = cv2.absdiff(now_frame, before_frame)
    return np.mean(difference)


def scan_detection(image):
    height, width = image.shape[:2]
    # Default fallback: entire image boundaries
    document_contour = np.array([[[0, 0]], [[width, 0]], [[width, height]], [[0, height]]], dtype=np.int32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    edges = cv2.Canny(blur, 30, 120)
    combined = cv2.bitwise_or(thresh, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=4)
    dilated = cv2.dilate(closing, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    img_area = width * height

    # Store the final 4 clean corners explicitly for rendering straight lines
    final_corners = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > (img_area * 0.20):
            hull = cv2.convexHull(contour)
            peri = cv2.arcLength(hull, True)

            approx = hull
            found_four_points = False
            for epsilon_factor in np.linspace(0.01, 0.15, 30):
                candidate_approx = cv2.approxPolyDP(hull, epsilon_factor * peri, True)
                if len(candidate_approx) == 4:
                    approx = candidate_approx
                    found_four_points = True
                    break
                elif len(candidate_approx) < 4:
                    break

            if area > max_area:
                if found_four_points:
                    pts = approx.squeeze()
                else:
                    rect = cv2.minAreaRect(hull)
                    box = cv2.boxPoints(rect)
                    pts = np.int32(box)

                # If squeezed array drops to a single dimension error, patch it safely
                if len(pts.shape) == 2 and pts.shape[0] >= 3:
                    s = pts.sum(axis=1)
                    diff = np.diff(pts, axis=1)

                    tl = tuple(pts[np.argmin(s)])
                    br = tuple(pts[np.argmax(s)])
                    tr = tuple(pts[np.argmin(diff)])
                    bl = tuple(pts[np.argmax(diff)])

                    final_corners = [tl, tr, br, bl]
                    document_contour = np.array([[tl], [tr], [br], [bl]], dtype=np.int32)
                    max_area = area
                    break

    annotated_image = image.copy()

    # FIX: If we extracted 4 distinct corners, draw 4 dead-straight lines connecting them
    if final_corners is not None:
        tl, tr, br, bl = final_corners
        cv2.line(annotated_image, tl, tr, (0, 255, 0), 3)
        cv2.line(annotated_image, tr, br, (0, 255, 0), 3)
        cv2.line(annotated_image, br, bl, (0, 255, 0), 3)
        cv2.line(annotated_image, bl, tl, (0, 255, 0), 3)
    else:
        # Fallback tracking display
        cv2.drawContours(annotated_image, [document_contour], -1, (0, 255, 0), 3)

    return document_contour, annotated_image

saved_frames = 0
current_frame = 0
threshold = 15
previous_frame = None

motion = False
stable_count = 0
stable_cooldown = 2 #2 frames cooldown
frame_save_count = 3

candidate_frames = []

frame_interval = int(fps * 0.2)

# Create a resizable display window
cv2.namedWindow("Detected Page", cv2.WINDOW_NORMAL)


#process Video
while True:
    success, frame = cap.read()
    if not success:# no frames left
        break

    if current_frame % frame_interval == 0: #Cut down initial huge frames by frame interval

        should_save_frame = False

        if previous_frame is None: #-----if there is no previous frame then save it ofc
            first_frame = frame.copy()
            contour, first_out = scan_detection(first_frame)
            save_image(first_out, current_frame)
            previous_frame = frame.copy()
            current_frame += 1
            continue

        else:
            #-----MOTION DETECTION-----
            change_score = calculate_change_score(previous_frame, frame)
            # Output of compared frames
            if change_score > threshold:
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
                        should_save_frame = True
                        frame_save_count -= 1 # one is saved
                        #reset
                        if frame_save_count == 0:
                            motion = False
                            stable_count = 0
                            frame_save_count = 3 # initial save count

    #NEXT: compare images, crop it  and add effects and convert to pdf file

        #SAVING FRAMES Sharpness -> Page detect
        if should_save_frame:
            candidate_frames.append(frame.copy())

            if len(candidate_frames) == 3 :
                #-get sharpest frame-
                final_frame,best_frame_score = get_sharpest_frame(candidate_frames)
                print(f"Best score: {best_frame_score:.2f}")

                #-- CONTINUEEEEEEEEE HEREEEEEEEEE
                doc_contour, visual_frame = scan_detection(final_frame)



                cv2.imshow("Detected Page", visual_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                save_image(visual_frame, current_frame)#save this frame
                saved_frames += 1

                candidate_frames.clear()

        previous_frame = frame.copy() # stores copy of saved frame to compare with next one

    current_frame += 1



cap.release()

print(f"Total saved frames: {saved_frames}")