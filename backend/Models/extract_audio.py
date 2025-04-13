import os
import subprocess
import json

# Paths
json_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"
data_dir = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF"

# Load metadata
with open(json_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Extract audio from videos
for entry in metadata:
    video_file = os.path.join(data_dir, entry['file'])
    audio_file = os.path.splitext(video_file)[0] + ".wav"
    if not os.path.exists(audio_file):  # Skip if already extracted
        try:
            print(f"Extracting audio from {video_file}...")
            subprocess.run(
                ["ffmpeg", "-i", video_file, "-ar", "16000", "-ac", "1", audio_file],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from {video_file}: {e}")