
import cv2
import numpy as np
import mediapipe as mp
import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

def detect_pose_blazepose(frame):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark
        keypoints = np.array([[l.y, l.x, l.visibility] for l in landmarks])
        return keypoints

def draw_keypoints(frame, keypoints, threshold=0.3):
    height, width, _ = frame.shape
    for idx, (y, x, c) in enumerate(keypoints):
        if c > threshold:
            cx, cy = int(x * width), int(y * height)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
    return frame

def extract_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frames.append((i, frame))
    cap.release()
    return frames

def analyze_pose_with_chatgpt(frame_data):
    prompt = (
        "You are a pose analysis assistant. Given this set of BlazePose keypoints:\n\n"
        f"{json.dumps(frame_data, indent=2)}\n\n"
        "Label the view (front, side, back), and indicate which body parts are reliable "
        "for measurement (confidence > 0.8). Mention any symmetry or alignment issues.\n"
        "Return only a JSON with keys: \"view\", \"reliable_parts\", \"notes\"."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a pose analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return json.loads(response.choices[0].message["content"])

def process_video(video_path, num_frames=30, save_dir="annotated_frames"):
    print("Extracting frames from video...")
    frames = extract_frames(video_path, num_frames=num_frames)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Running BlazePose on extracted frames...")
    top_frames = []
    for idx, frame in frames:
        keypoints = detect_pose_blazepose(frame)
        if keypoints is None:
            continue
        avg_conf = np.mean(keypoints[:, 2])
        top_frames.append((avg_conf, idx, frame, keypoints))

    top_frames.sort(reverse=True)
    selected = top_frames[:10]

    for avg_conf, idx, frame, keypoints in selected:
        frame_data = {
            "frame_index": idx,
            "keypoints": keypoints.tolist()
        }
        try:
            analysis = analyze_pose_with_chatgpt(frame_data)
            print(f"\n📸 Frame {idx} - ChatGPT Response:")
            print(json.dumps(analysis, indent=2))

            annotated = draw_keypoints(frame.copy(), keypoints)
            cv2.imwrite(os.path.join(save_dir, f"frame_{idx}_blazepose.jpg"), annotated)
        except Exception as e:
            print(f"\n⚠️ Error analyzing frame {idx}: {e}")

if __name__ == "__main__":
    video_path = "your_video.mp4"  # replace with your video path
    process_video(video_path)
