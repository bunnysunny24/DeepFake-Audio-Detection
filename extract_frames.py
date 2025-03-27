import os
import cv2
from tqdm import tqdm

def extract_frames_from_videos(video_folder, output_folder, frame_interval=5):
    """
    Extracts frames from videos and saves them as images.
    
    - video_folder: Path to MP4 videos.
    - output_folder: Where extracted frames will be stored.
    - frame_interval: Extract every nth frame.
    """
    os.makedirs(output_folder, exist_ok=True)

    for video_file in tqdm(os.listdir(video_folder), desc="Extracting frames"):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_folder, video_file)
            vidcap = cv2.VideoCapture(video_path)

            frame_count = 0
            success, image = vidcap.read()
            while success:
                if frame_count % frame_interval == 0:  # Save every nth frame
                    filename = f"{os.path.splitext(video_file)[0]}_frame{frame_count}.jpg"
                    frame_path = os.path.join(output_folder, filename)
                    cv2.imwrite(frame_path, image)
                success, image = vidcap.read()
                frame_count += 1

            vidcap.release()

# Paths
video_folder_processed = "../image_data/downloaded_celebvhq/processed"
video_folder_raw = "../image_data/downloaded_celebvhq/raw"
output_frames_processed = "../image_data/frames/processed"
output_frames_raw = "../image_data/frames/raw"

# Extract frames from processed and raw videos
extract_frames_from_videos(video_folder_processed, output_frames_processed)
extract_frames_from_videos(video_folder_raw, output_frames_raw)
