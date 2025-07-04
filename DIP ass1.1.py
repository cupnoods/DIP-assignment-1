import cv2
import numpy as np

# === CONFIGURATION ===
VIDEO_LIST = ["alley.mp4", "office.mp4", "singapore.mp4", "traffic.mp4"]
TALKING_VIDEO_PATH = "talking.mp4"
ENDSCREEN_PATH = "endscreen.mp4"
WATERMARK1_PATH = "watermark1.png"
WATERMARK2_PATH = "watermark2.png"
FACE_CASCADE_PATH = "face_detector.xml"
FPS = 30

# === HELPER FUNCTIONS ===
def is_night_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < 80

def increase_brightness(frame, value=50):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def blur_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        region = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(region, (51, 51), 30)
    return frame

def overlay_talking_video(base_frame, overlay_frame, scale=0.25):
    h_base, w_base = base_frame.shape[:2]
    h_overlay, w_overlay = overlay_frame.shape[:2]
    new_w = int(w_base * scale)
    new_h = int(new_w * (h_overlay / w_overlay))
    overlay_resized = cv2.resize(overlay_frame, (new_w, new_h))
    if new_h + 10 < h_base and new_w + 10 < w_base:
        base_frame[10:10+new_h, 10:10+new_w] = overlay_resized
    return base_frame

def add_watermark(frame, watermark):
    h_f, w_f = frame.shape[:2]
    watermark_resized = cv2.resize(watermark, (w_f, h_f))
    mask = cv2.inRange(watermark_resized, (1, 1, 1), (255, 255, 255))
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(watermark_resized, watermark_resized, mask=mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    return cv2.add(bg, fg)

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
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

# === LOAD FIXED ASSETS ===
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
talking_frames = read_all_frames(TALKING_VIDEO_PATH)
endscreen_frames = read_all_frames(ENDSCREEN_PATH)
watermark1 = cv2.imread(WATERMARK1_PATH)
watermark2 = cv2.imread(WATERMARK2_PATH)

# === PROCESS ALL VIDEOS ===
for video_file in VIDEO_LIST:
    print(f"🔄 Processing {video_file}...")
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = []
    frame_idx = 0
    night_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == 0:
            night_mode = is_night_frame(frame)
        if night_mode:
            frame = increase_brightness(frame)
        frame = blur_faces(frame, face_cascade)
        if frame_idx < len(talking_frames):
            frame = overlay_talking_video(frame, talking_frames[frame_idx])
        if frame_idx < frame_count // 2:
            frame = add_watermark(frame, watermark1)
        else:
            frame = add_watermark(frame, watermark2)
        processed_frames.append(frame)
        frame_idx += 1

    cap.release()
    processed_frames.extend(endscreen_frames)

    output_file = video_file.replace(".mp4", "_edited.mp4")
    print(f"💾 Saving to {output_file}...")
    save_video(processed_frames, output_file, FPS)

print("✅ All videos processed!")
