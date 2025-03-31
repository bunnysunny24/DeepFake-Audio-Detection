import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Define 19 Feature Classes
FEATURES = {
    "skin": [0, 10, 338, 297, 332],
    "left_brow": [70, 63, 105, 66, 107],
    "right_brow": [300, 293, 334, 296, 336],
    "left_eye": [133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye": [362, 398, 384, 385, 386, 387, 388, 466],
    "eye_glass": [],
    "left_ear": [234, 127, 139, 218],
    "right_ear": [454, 356, 366, 447],
    "ear_ring": [],
    "nose": [1, 2, 97, 168, 6],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "upper_lip": [0, 13, 14, 15, 16],
    "lower_lip": [17, 18, 19, 20, 21],
    "neck": [],
    "necklace": [],
    "cloth": [],
    "hair": [10, 338, 297, 332, 284, 251],
    "hat": [],
    "background": []
}

# Extract Features from Image
def extract_features(image_path, output_dir, image_id):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to read: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print(f"[WARNING] No face detected in {image_path}")
        return

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape

    for feature_name, landmark_ids in FEATURES.items():
        mask = np.zeros((h, w), dtype=np.uint8)

        for landmark_id in landmark_ids:
            x = int(landmarks[landmark_id].x * w)
            y = int(landmarks[landmark_id].y * h)
            cv2.circle(mask, (x, y), 5, 255, -1)  # Draw keypoints

        output_path = os.path.join(output_dir, f"{image_id}_{feature_name}.png")
        cv2.imwrite(output_path, mask)
        print(f"[INFO] Saved: {output_path}")

# Process Only "Validation/Fake" Images
def process_validation_fake(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_folder = os.path.join(input_dir, "validation", "real")
    output_folder = os.path.join(output_dir, "validation", "real")
    os.makedirs(output_folder, exist_ok=True)

    image_id = 1  # Unique ID for each image
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        extract_features(image_path, output_folder, f"image-{image_id}")
        image_id += 1  # Increment ID for next image

if __name__ == "__main__":
    input_dir = "D:/Bunny/Deepfake/backend/image_data/image-dataset-7"
    output_dir = "D:/Bunny/Deepfake/backend/image_data/segmented_mediapipe"
    process_validation_fake(input_dir, output_dir)
