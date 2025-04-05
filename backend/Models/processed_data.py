import json
import os
import cv2
import librosa
import numpy as np
import torch
import torchaudio.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from moviepy.video.io.VideoFileClip import VideoFileClip
import tempfile
import mediapipe as mp

# ============================ #
#       Path Configuration     #
# ============================ #

metadata_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"
data_dir = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF"
save_path = r"D:\Bunny\Deepfake\backend\combined_data\processed_data.json"

# ============================ #
#   Mediapipe FaceMesh Setup   #
# ============================ #

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# ============================ #
#    Step 1: Extract Frames    #
# ============================ #

def extract_frames(video_path, max_frames=32, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    cap.release()
    return np.array(frames) if frames else None

# ============================ #
#    Normalize Face Poses      #
# ============================ #

def normalize_faces(frames):
    normalized = []
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for frame in frames:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                left_eye_coords = np.array([left_eye.x * w, left_eye.y * h])
                right_eye_coords = np.array([right_eye.x * w, right_eye.y * h])
                dx, dy = right_eye_coords - left_eye_coords
                angle = np.degrees(np.arctan2(dy, dx))
                center = tuple(np.mean([left_eye_coords, right_eye_coords], axis=0).astype(float))
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned = cv2.warpAffine(frame, M, (w, h))
                normalized.append(aligned)
            else:
                normalized.append(frame)
    return normalized

# ============================ #
# Step 2: Extract Audio Feature #
# ============================ #

def extract_audio_features(video_path, sr=16000):
    try:
        print(f"🔊 Extracting audio from: {video_path}")
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(temp_audio.name, logger=None)
        audio, _ = librosa.load(temp_audio.name, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        print(f"✅ Mel spectrogram extracted successfully.")
        return log_mel_spec
    except Exception as e:
        print(f"❌ Error processing {video_path}: {type(e).__name__} - {e}")
        return None
    finally:
        try:
            os.remove(temp_audio.name)
        except Exception as cleanup_error:
            print(f"⚠️ Failed to delete temp audio file: {cleanup_error}")

# ============================ #
#   Step 3: Define Augmenters  #
# ============================ #

video_augment = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.HorizontalFlip(p=0.5),
    ToTensorV2()
])

def augment_frames(frames):
    return [video_augment(image=frame)['image'].numpy() for frame in frames]

def augment_audio(audio_tensor):
    vol_transform = T.Vol(0.5)
    freq_mask = T.FrequencyMasking(freq_mask_param=15)
    time_mask = T.TimeMasking(time_mask_param=35)
    audio_tensor = vol_transform(audio_tensor)
    audio_tensor = freq_mask(audio_tensor)
    audio_tensor = time_mask(audio_tensor)
    return audio_tensor.numpy()

# ============================ #
#   Deepfake Detection Cues    #
# ============================ #

# ✂️ Keep all previous imports and path definitions as-is

# ============================ #
#   Updated Detection Functions #
# ============================ #

def detect_eye_blinking(frames):
    EAR_THRESHOLD = 0.2
    blink_count = 0
    frame_with_face_count = 0

    def eye_aspect_ratio(landmarks, indices):
        a = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[5]])
        b = np.linalg.norm(landmarks[indices[2]] - landmarks[indices[4]])
        c = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])
        return (a + b) / (2.0 * c)

    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for frame in frames:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results and results.multi_face_landmarks:
                frame_with_face_count += 1
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                coords = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
                left_ear = eye_aspect_ratio(coords, left_eye_indices)
                right_ear = eye_aspect_ratio(coords, right_eye_indices)
                avg_ear = (left_ear + right_ear) / 2.0
                if avg_ear < EAR_THRESHOLD:
                    blink_count += 1

    blink_rate = blink_count / frame_with_face_count if frame_with_face_count else 0
    return blink_rate < 0.05

def detect_lip_sync_issue(frames, audio_spec):
    if audio_spec is None:
        return True

    mouth_indices = [13, 14]
    mouth_movement = []

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for frame in frames:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results and results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                top = np.array([landmarks[mouth_indices[0]].x * w, landmarks[mouth_indices[0]].y * h])
                bottom = np.array([landmarks[mouth_indices[1]].x * w, landmarks[mouth_indices[1]].y * h])
                dist = np.linalg.norm(top - bottom)
                mouth_movement.append(dist)
            else:
                mouth_movement.append(0)

    mouth_std = np.std(mouth_movement)
    audio_energy = np.mean(audio_spec)
    return mouth_std < 1.0 and audio_energy > 0.5

def detect_facial_inconsistencies(frames):
    prev_landmarks = None
    shifts = []

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for frame in frames:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results and results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                coords = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
                if prev_landmarks is not None:
                    shift = np.mean(np.linalg.norm(coords - prev_landmarks, axis=1))
                    shifts.append(shift)
                prev_landmarks = coords

    return np.mean(shifts) > 10 if shifts else False


# ============================ #
#     Step 4: Tag Hard Cases   #
# ============================ #

def analyze_challenging_conditions(frames, audio_spec):
    low_light = False
    occlusion = False
    extreme_pose = False
    distorted_audio = False

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 50:
            low_light = True
        black_pixels = np.sum(gray < 10)
        if black_pixels / gray.size > 0.3:
            occlusion = True

    extreme_pose = any(frame is None for frame in frames)

    if audio_spec is not None:
        std_dev = np.std(audio_spec)
        if std_dev < 0.1 or std_dev > 10:
            distorted_audio = True

    return {
        "low_light": bool(low_light),
        "occlusion": bool(occlusion),
        "extreme_pose": bool(extreme_pose),
        "distorted_audio": bool(distorted_audio),
        "eye_blinking_issue": bool(detect_eye_blinking(frames)),
        "lip_sync_issue": bool(detect_lip_sync_issue(frames, audio_spec)),
        "facial_inconsistency": bool(detect_facial_inconsistencies(frames))
    }

# ============================ #
#   Step 5: Process + Save     #
# ============================ #

# Load metadata
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

if os.path.exists(save_path):
    with open(save_path, "r", encoding="utf-8") as f:
        processed_data = json.load(f)
        processed_files = {item["file"] for item in processed_data}
else:
    processed_data = []
    processed_files = set()

for entry in metadata:
    if entry["file"] in processed_files:
        print(f"⏩ Already processed: {entry['file']}")
        continue

    video_path = os.path.join(data_dir, entry["file"])
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        continue

    print(f"\n🔄 Processing: {entry['file']}")

    frames = extract_frames(video_path)
    if frames is None:
        print(f"⚠️ Skipping {entry['file']} due to missing frames.")
        continue

    normalized_frames = normalize_faces(frames)
    augmented_frames = augment_frames(normalized_frames)
    audio_features = extract_audio_features(video_path)

    if audio_features is not None:
        augmented_audio = augment_audio(torch.tensor(audio_features))
    else:
        print(f"⚠️ Skipping audio for {entry['file']} due to extraction error.")
        augmented_audio = None

    challenge_tags = analyze_challenging_conditions(frames, audio_features)

    try:
        video_data = {
            "file": entry["file"],
            "label": 1 if entry.get("n_fakes", 0) > 0 else 0,
            "frames": [frame.tolist() if isinstance(frame, np.ndarray) else None for frame in augmented_frames],
            "audio": augmented_audio.tolist() if isinstance(augmented_audio, np.ndarray) else None,
            "tags": challenge_tags
        }
        processed_data.append(video_data)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=4)

        print(f"✅ Saved progress after processing: {entry['file']}")

    except Exception as e:
        print(f"💥 Failed to process {entry['file']}: {type(e).__name__} - {e}")
