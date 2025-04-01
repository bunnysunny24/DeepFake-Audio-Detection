import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Define Feature Classes with Example Landmark IDs
FEATURES = {
    "skin": [0, 10, 338, 297, 332, 284, 251],
    "left_brow": [70, 63, 105, 66, 107],
    "right_brow": [300, 293, 334, 296, 336],
    "left_eye": [133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye": [362, 398, 384, 385, 386, 387, 388, 466],
    "nose": [1, 2, 97, 168, 6, 195, 5, 4],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "upper_lip": [0, 13, 14, 15, 16],
    "lower_lip": [17, 18, 19, 20, 21],
    "hair": [10, 338, 297, 332, 284, 251],
}

# Extract Features from Image
def extract_features(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Unable to load image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Extract filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Detect face
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        print(f"[WARNING] No face detected in {image_path}")
        return

    landmarks = results.multi_face_landmarks[0].landmark

    for feature_name, landmark_ids in FEATURES.items():
        points = []
        for l in landmark_ids:
            if l < len(landmarks):  # Ensure the landmark exists
                x = int(landmarks[l].x * w)
                y = int(landmarks[l].y * h)
                points.append((x, y))

        if len(points) < 3:
            print(f"[WARNING] Skipping {feature_name} in {image_path} (Not enough landmarks)")
            continue  # Skip if not enough points for a valid shape

        # Ensure the points are inside the image bounds
        valid_points = [(x, y) for x, y in points if 0 <= x < w and 0 <= y < h]
        if len(valid_points) < 3:
            print(f"[WARNING] Skipping {feature_name} in {image_path} (Invalid points)")
            continue  # Skip if invalid points

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(valid_points)], 255)  # Fill shape

        # Extract feature
        extracted_feature = cv2.bitwise_and(image, image, mask=mask)

        # Crop bounding box
        x, y, bbox_w, bbox_h = cv2.boundingRect(np.array(valid_points))
        cropped_feature = extracted_feature[y:y+bbox_h, x:x+bbox_w]

        if cropped_feature.size == 0:
            print(f"[WARNING] Skipping empty {feature_name} in {image_path}")
            continue  # Skip if crop is empty

        # Save cropped feature with original filename + feature name
        output_path = os.path.join(output_dir, f"{base_filename}_{feature_name}.png")
        cv2.imwrite(output_path, cropped_feature)

        print(f"[INFO] Saved: {output_path}")

# Process Only "Validation/Fake" Images
def process_validation_fake(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_folder = os.path.join(input_dir, "validation", "real")
    output_folder = os.path.join(output_dir, "validation", "real")
    os.makedirs(output_folder, exist_ok=True)

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        extract_features(image_path, output_folder)

if __name__ == "__main__":
    input_dir = "D:/Bunny/Deepfake/backend/image_data/image-dataset-7"
    output_dir = "D:/Bunny/Deepfake/backend/image_data/extracted_features-3"
    process_validation_fake(input_dir, output_dir)
