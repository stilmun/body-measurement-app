import cv2
import numpy as np
import os
import openai
import json
import mediapipe as mp
import subprocess

# 🔐 Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 🧩 Resize + Trim video with ffmpeg
def preprocess_video(input_path, output_path="processed_video.mp4", max_duration=10):
    try:
        command = [
            "ffmpeg", "-i", input_path,
            "-vf", "scale=-2:360",
            "-t", str(max_duration),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-b:a", "128k",
            "-y", output_path
        ]
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print("❌ Error processing video with ffmpeg:", e)
        return None

# BlazePose keypoint detector
def detect_pose_blazepose(frame):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark
        return np.array([[l.y, l.x, l.visibility] for l in landmarks])  # shape: (33, 3)

# Draw keypoints
def draw_keypoints(frame, keypoints, threshold=0.8):
    h, w, _ = frame.shape
    for i, (y, x, c) in enumerate(keypoints):
        cx, cy = int(x * w), int(y * h)
        color = (0, 255, 0) if c >= threshold else (0, 0, 255)
        radius = 4 if c >= threshold else 2
        cv2.circle(frame, (cx, cy), radius, color, -1)
    return frame

# Extract frames
def extract_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((i, frame_rgb))
    cap.release()
    return frames

# ChatGPT filter
def analyze_pose_with_chatgpt(frame_data):
    prompt = f"""
You are a pose analysis assistant. Here are the keypoints:

{json.dumps(frame_data, indent=2)}

Tell me which body parts are reliable for measurement (confidence > 0.8).
Only return a JSON with keys: "view", "reliable_parts", "notes"
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a pose analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
        )
        return json.loads(response.choices[0].message["content"])
    except Exception as e:
        print("❌ OpenAI API Error:", e)
        return {"view": "unknown", "reliable_parts": [], "notes": "API error"}

# Helpers
def pixel_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def convert_to_pixel_coords(keypoints, image_shape):
    h, w = image_shape[:2]
    return [(int(x * w), int(y * h), c) for y, x, c in keypoints]

# 🧠 Main Flow
def process_video_and_measure(video_path, height_cm):
    processed_path = preprocess_video(video_path)
    if not processed_path:
        return {"error": "Video preprocessing failed"}, None

    os.makedirs("annotated_frames", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    frames = extract_frames(processed_path, num_frames=30)
    frame_data = []

    for idx, frame in frames:
        blazepose_kps = detect_pose_blazepose(frame)
        if blazepose_kps is None:
            continue

        avg_conf = np.mean(blazepose_kps[:, 2])
        frame_data.append((avg_conf, idx, frame, blazepose_kps))

        frame_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        debug_img = draw_keypoints(frame_bgr.copy(), blazepose_kps)
        cv2.imwrite(f"annotated_frames/frame_{idx}_blazepose.jpg", debug_img)

    top = sorted(frame_data, reverse=True)[:10]
    if not top:
        return {"error": "No valid frames found"}, None

    all_keypoints = [kp for _, _, _, kp in top]
    keypoints_stack = np.stack(all_keypoints)
    weights = keypoints_stack[:, :, 2]
    weights_sum = np.sum(weights, axis=0, keepdims=True)
    weights_normalized = weights / (weights_sum + 1e-6)

    y_avg = np.sum(keypoints_stack[:, :, 0] * weights_normalized, axis=0)
    x_avg = np.sum(keypoints_stack[:, :, 1] * weights_normalized, axis=0)
    conf_avg = np.mean(keypoints_stack[:, :, 2], axis=0)
    averaged_kps = np.stack([y_avg, x_avg, conf_avg], axis=-1)

    input_data = {
        "frame_index": -1,
        "keypoints": averaged_kps.tolist()
    }
    chat_result = analyze_pose_with_chatgpt(input_data)
    reliable = chat_result["reliable_parts"]

    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if not all(r in reliable for r in required):
        return {"error": "ChatGPT rejected the average frame"}, None

    frame = top[0][2]
    pixel_coords = convert_to_pixel_coords(averaged_kps, frame.shape)

    def get_xy(name):
        i = KEYPOINT_NAMES.index(name)
        return pixel_coords[i][:2]

    shoulder_px = pixel_distance(get_xy("left_shoulder"), get_xy("right_shoulder"))
    hip_px = pixel_distance(get_xy("left_hip"), get_xy("right_hip"))
    left_leg_px = pixel_distance(get_xy("left_hip"), get_xy("left_ankle"))
    right_leg_px = pixel_distance(get_xy("right_hip"), get_xy("right_ankle"))
    left_arm_px = pixel_distance(get_xy("left_shoulder"), get_xy("left_wrist"))
    right_arm_px = pixel_distance(get_xy("right_shoulder"), get_xy("right_wrist"))

    head_px = get_xy("nose")[1]
    feet_px = max(get_xy("left_ankle")[1], get_xy("right_ankle")[1])
    full_body_px = feet_px - head_px
    cm_per_px = height_cm / full_body_px

    measurements = {
        "Shoulder Width": round(shoulder_px * cm_per_px, 2),
        "Hip Width": round(hip_px * cm_per_px, 2),
        "Leg Length": round(((left_leg_px + right_leg_px) / 2) * cm_per_px, 2),
        "Arm Length": round(((left_arm_px + right_arm_px) / 2) * cm_per_px, 2),
    }

    for i, (x, y, c) in enumerate(pixel_coords):
        label = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"kp_{i}"
        color = (0, 255, 0) if c >= 0.8 else (0, 0, 255)
        cv2.circle(frame, (x, y), 4, color, -1)
        cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    output_path = "static/annotated.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return measurements, output_path
