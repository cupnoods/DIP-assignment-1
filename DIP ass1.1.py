import cv2
import numpy as np
import os

# === CONFIGURATION ===
VIDEO_PATH = 'street.mp4'
TALKING_VIDEO_PATH = 'talking.mp4'
ENDSCREEN_PATH = 'endscreen.mp4'
WATERMARK1_PATH = 'watermark1.png'
WATERMARK2_PATH = 'watermark2.png'
FACE_CASCADE_PATH = 'face_detector.xml'
OUTPUT_PATH = 'task_a_output.avi'
FPS = 30

# === FUNCTION DEFINITIONS ===
def is_night_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < 80  # threshold for night

def increase_brightness(frame, value=50):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def blur_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        face_region = cv2.GaussianBlur(face_region, (51, 51), 30)
        frame[y:y+h, x:x+w] = face_region
    return frame

def overlay_video(base_frame, overlay_frame):
    h_b, w_b = base_frame.shape[:2]
    h_o, w_o = overlay_frame.shape[:2]
    scale = 0.25  # 25% of main video height
    new_h = int(h_b * scale)
    aspect_ratio = w_o / h_o
    new_w = int(new_h * aspect_ratio)
    small_overlay = cv2.resize(overlay_frame, (new_w, new_h))
    if new_h + 10 < h_b and new_w + 10 < w_b:
        base_frame[10:10+new_h, 10:10+new_w] = small_overlay
    return base_frame

def add_watermark(frame, watermark):
    # Resize watermark to match frame size
    h_f, w_f = frame.shape[:2]
    watermark_resized = cv2.resize(watermark, (w_f, h_f))

    if watermark_resized.shape[2] == 4:  # has alpha channel
        overlay = watermark_resized[:, :, :3]  # BGR
        mask = watermark_resized[:, :, 3] / 255.0  # Alpha mask
        for c in range(3):
            frame[:, :, c] = (1.0 - mask) * frame[:, :, c] + mask * overlay[:, :, c]
    else:
        frame = cv2.addWeighted(frame, 1, watermark_resized, 0.3, 0)
    return frame

def read_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, output_path, fps):
    if not frames:
        print("No frames to save.")
        return
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

# === MAIN PROCESSING ===
print("Loading face cascade...")
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

print("Loading main video...")
cap = cv2.VideoCapture(VIDEO_PATH)
overlay_frames = read_all_frames(TALKING_VIDEO_PATH)
endscreen_frames = read_all_frames(ENDSCREEN_PATH)
watermark1 = cv2.imread(WATERMARK1_PATH, cv2.IMREAD_UNCHANGED)
watermark2 = cv2.imread(WATERMARK2_PATH, cv2.IMREAD_UNCHANGED)

processed_frames = []
frame_idx = 0
night_mode = False
video_frames_temp = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_frames_temp.append(frame)

cap.release()
total_frames = len(video_frames_temp)

for idx, frame in enumerate(video_frames_temp):
    # Step 1: Night detection and brightness increase
    if idx == 0:
        night_mode = is_night_frame(frame)
    if night_mode:
        frame = increase_brightness(frame, 50)

    # Step 2: Face blurring
    frame = blur_faces(frame, face_cascade)

    # Step 3: Overlay talking video
    if idx < len(overlay_frames):
        frame = overlay_video(frame, overlay_frames[idx])

    # Step 4: Add watermark
    if idx < total_frames // 2:
        frame = add_watermark(frame, watermark1)
    else:
        frame = add_watermark(frame, watermark2)

    processed_frames.append(frame)

# Step 5: Add end screen video
processed_frames.extend(endscreen_frames)

# Save result
print("Saving processed video to:", OUTPUT_PATH)
save_video(processed_frames, OUTPUT_PATH, FPS)
print("Done!")
