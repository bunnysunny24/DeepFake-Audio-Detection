import cv2
import os
from mtcnn import MTCNN

# Path where videos are stored
video_folder = r"D:\Bunny\Deepfake\backend\image_data\FF++\fake"

# Path to save extracted faces
output_folder = r"D:\Bunny\Deepfake\backend\image_data\image-dataset-6"
os.makedirs(output_folder, exist_ok=True)

# Initialize MTCNN face detector
detector = MTCNN()

# Get the first dataset (first video file)
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
if not video_files:
    print("No MP4 files found in the directory.")
    exit()

first_video = video_files[0]  # Select only the first dataset
video_path = os.path.join(video_folder, first_video)

print(f"Processing: {first_video}")

# Open the video file
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces
    faces = detector.detect_faces(frame)
    for idx, face in enumerate(faces):
        x, y, w, h = face['box']

        # Validate bounding box
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            print(f"Invalid bounding box in {first_video}: {x}, {y}, {w}, {h}")
            continue

        # Extract face region
        face_img = frame[y:y+h, x:x+w]

        # Debugging - Display extracted face
        cv2.imshow("Detected Face", face_img)
        cv2.waitKey(1)  # Change to 0 to pause at each face

        # Save face image
        face_filename = f"{os.path.splitext(first_video)[0]}_frame{frame_count}_face{idx}.jpg"
        face_path = os.path.join(output_folder, face_filename)
        cv2.imwrite(face_path, face_img)

        print(f"Saved: {face_path}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Processing complete.")
