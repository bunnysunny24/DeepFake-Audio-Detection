import os
import cv2
from tqdm import tqdm

def extract_frames(video_path, save_dir, frame_interval=5):
    """ Extract frames from a video and save them in a directory """
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return 0

    frame_count, saved_frames = 0, 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            frame_filename = f"{os.path.basename(video_path)}_{frame_count}.jpg"
            frame_path = os.path.join(save_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frames += 1

        frame_count += 1

    cap.release()
    
    if saved_frames == 0:
        print(f"⚠️ Warning: No frames were saved for {video_path}")
    else:
        print(f"✅ Extracted {saved_frames} frames from {video_path}")

    return saved_frames

def process_videos(video_folder, output_folder):
    """ Process videos: Extract frames and store them in the output folder """
    if not os.path.exists(video_folder):
        print(f"❌ Error: Folder {video_folder} does not exist!")
        return

    video_files = [os.path.join(video_folder, vid) for vid in os.listdir(video_folder) if vid.endswith(".mp4")]
    
    if not video_files:
        print(f"⚠️ Warning: No .mp4 videos found in {video_folder}")
        return

    print(f"📽️ Found {len(video_files)} videos in {video_folder}")

    frame_dir = os.path.join(output_folder, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    print("🔄 Extracting frames from videos...")
    for video in tqdm(video_files, desc="Processing Videos"):
        extract_frames(video, frame_dir)

# Paths
celeb_synthesis_videos = "../image_data/Celeb_DF/Celeb-synthesis"
celeb_synthesis_output = "../image_data/image_dataset-4/Celeb-synthesis"

print("🚀 Extracting frames for Celeb-synthesis...")
process_videos(celeb_synthesis_videos, celeb_synthesis_output)

print("✅ Frame extraction complete!")
