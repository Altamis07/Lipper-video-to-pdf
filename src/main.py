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
    cv2.imwrite(str(frames_dir), frame_name, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # save current working frame


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
    document_contour = np.array([[[0, 0]], [[width, 0]], [[width, height]], [[0, height]]], dtype=np.int32)

    # Keep original frame clean so text stays crisp
    annotated_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Isolate bilateral filter processing strictly to this variable
    gray_processed = cv2.bilateralFilter(gray, 9, 75, 75)

    thresh = cv2.adaptiveThreshold(gray_processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    edges = cv2.Canny(gray_processed, 30, 120)
    combined = cv2.bitwise_or(thresh, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=4)
    dilated = cv2.dilate(closing, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    img_area = width * height
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

    if final_corners is not None:
        tl, tr, br, bl = final_corners
        cv2.line(annotated_image, tl, tr, (0, 255, 0), 3)
        cv2.line(annotated_image, tr, br, (0, 255, 0), 3)
        cv2.line(annotated_image, br, bl, (0, 255, 0), 3)
        cv2.line(annotated_image, bl, tl, (0, 255, 0), 3)
    else:
        cv2.drawContours(annotated_image, [document_contour], -1, (0, 255, 0), 3)

    return document_contour, annotated_image


def perspective_warp(image, target_contour):
    pts = target_contour.squeeze()
    (tl, tr, br, bl) = pts

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
    return cv2.warpPerspective(image, transform_matrix, (max_width, max_height))


def enhance_document(warped_image):
    # Convert to grayscale
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

    # 1. Base normalization stretch (the one that works well)
    normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 2. Smoothly clip the extreme highs/lows to make the background uniform white
    # and ink solid black, while letting midtones smoothly transition.
    #Change 2nd and 3rd val of xp for fine tune , 2nd for text and 3rd for bg
    xp = [0, 50, 245, 255]
    fp = [0, 0, 255, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')

    # Apply the clean map
    popped_text = cv2.LUT(normalized, table)

    return popped_text

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



#cv2.namedWindow("Detected Page", cv2.WINDOW_NORMAL)
#cv2.namedWindow("Flattened Document", cv2.WINDOW_NORMAL)
#cv2.namedWindow("final Document", cv2.WINDOW_NORMAL)


#process Video
while True:


    success, frame = cap.read()
    if not success:# no frames left
        break



    if current_frame % frame_interval == 0: #Cut down initial huge frames by frame interval

        should_save_frame = False

        if previous_frame is None: #-----if there is no previous frame then save it ofc
            first_frame = frame.copy()

            # Detect corners
            doc_contour, visual_frame = scan_detection(first_frame)

            # WRAP Frame
            flattened_page = perspective_warp(first_frame, doc_contour)

            # Filters
            first_image = enhance_document(flattened_page)
            save_image(first_image, current_frame)

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



                #Detect corners
                doc_contour, visual_frame = scan_detection(final_frame)

                #WRAP Frame
                flattened_page = perspective_warp(final_frame, doc_contour)

                #Filters
                final_image = enhance_document(flattened_page)


                #cv2.imshow("Detected Page", visual_frame)
                #cv2.imshow("Flattened Document", flattened_page)
                #cv2.imshow("final Document", final_image)



                cv2.waitKey(0)
                cv2.destroyAllWindows()
                save_image(final_image, current_frame)#save this frame
                saved_frames += 1

                candidate_frames.clear()

        previous_frame = frame.copy() # stores copy of saved frame to compare with next one

    current_frame += 1



cap.release()

print(f"Total saved frames: {saved_frames}")