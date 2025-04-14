import cv2
import mediapipe as mp
import math

import os

# Debug: Print current working directory and check if file exists
print("Current working directory:", os.getcwd())
print("Looking for image:", os.path.abspath("test_image.jpg"))
print("File exists?", os.path.exists("test_image.jpg"))


# Step 1: Function to let user click two points on an image
def get_reference_points(image):
    points = []

    # Resize image if too tall for screen
    screen_height = 800  # Adjust this if needed
    height, width = image.shape[:2]
    scale = 1.0
    if height > screen_height:
        scale = screen_height / height
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            # Rescale back to original coordinates
            real_x = int(x / scale)
            real_y = int(y / scale)
            points.append((real_x, real_y))
            print(f"Point {len(points)}: {real_x}, {real_y}")

    cv2.imshow("Click on the reference object (2 points)", image)
    cv2.setMouseCallback("Click on the reference object (2 points)", mouse_callback)

    while len(points) < 2:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return points



# Step 2: Load image and detect landmarks
def measure_body_part(image_path, reference_length_cm):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    # Ask user to mark reference object
    print("Click on both ends of the known-length reference object in the image.")
    ref_points = get_reference_points(image.copy())
    ref_pixel_distance = math.dist(ref_points[0], ref_points[1])
    print(f"Reference object pixel distance: {ref_pixel_distance:.2f}")

    # Calculate conversion factor
    cm_per_pixel = reference_length_cm / ref_pixel_distance
    print(f"Conversion factor: {cm_per_pixel:.4f} cm/pixel")

    # Detect pose
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("No landmarks detected.")
        return

    height, width, _ = image.shape
    landmarks = results.pose_landmarks.landmark

    # Get shoulder landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Convert normalized to pixel coordinates
    left_shoulder_px = (int(left_shoulder.x * width), int(left_shoulder.y * height))
    right_shoulder_px = (int(right_shoulder.x * width), int(right_shoulder.y * height))

    # Calculate pixel and real distance
    pixel_distance = math.dist(left_shoulder_px, right_shoulder_px)
    real_distance_cm = pixel_distance * cm_per_pixel

    print(f"Shoulder width: {pixel_distance:.2f} pixels ≈ {real_distance_cm:.2f} cm")

    # Draw and display
    cv2.line(image, left_shoulder_px, right_shoulder_px, (0, 255, 0), 2)
    cv2.circle(image, left_shoulder_px, 5, (255, 0, 0), -1)
    cv2.circle(image, right_shoulder_px, 5, (0, 0, 255), -1)

    # Resize output if too tall for screen
    screen_height = 800
    height, width = image.shape[:2]
    if height > screen_height:
        scale = screen_height / height
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    cv2.imshow("Measurement", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# --------- HOW TO USE IT ----------
# 1. Change this to the path of your image
image_path = "test_image.jpg"

# 2. Type in the **real-world length (in cm)** of the object you click on
reference_length_cm = 8.5  # for example, a credit card is ~8.5 cm wide

measure_body_part(image_path, reference_length_cm)
