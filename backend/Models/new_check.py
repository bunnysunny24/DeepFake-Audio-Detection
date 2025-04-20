import json
import os
import torchaudio
import cv2

def sanity_check(json_path, data_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    missing_files = []
    failed_videos = []
    failed_audios = []
    total = len(data)

    print(f"Total samples to check: {total}")

    for i, sample in enumerate(data):  # Process all samples
        print(f"Checking sample {i + 1}/{total}")

        video_path = os.path.join(data_dir, sample['file'])
        audio_path = video_path.replace('.mp4', '.wav')

        # Optional: original file check
        original_video_path = os.path.join(data_dir, sample['original']) if 'original' in sample and sample['original'] else None
        original_audio_path = original_video_path.replace('.mp4', '.wav') if original_video_path else None

        # Check main files
        if not os.path.exists(video_path):
            missing_files.append(video_path)
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                failed_videos.append(video_path)
            cap.release()

        if not os.path.exists(audio_path):
            missing_files.append(audio_path)
        else:
            try:
                torchaudio.load(audio_path)
            except Exception as e:
                failed_audios.append((audio_path, str(e)))

        # Check original if needed
        if sample.get('modify_video', False) and original_video_path:
            if not os.path.exists(original_video_path):
                missing_files.append(original_video_path)
            else:
                cap = cv2.VideoCapture(original_video_path)
                if not cap.isOpened():
                    failed_videos.append(original_video_path)
                cap.release()

        if sample.get('modify_audio', False) and original_audio_path:
            if not os.path.exists(original_audio_path):
                missing_files.append(original_audio_path)
            else:
                try:
                    torchaudio.load(original_audio_path)
                except Exception as e:
                    failed_audios.append((original_audio_path, str(e)))

    print("\n✅ Check complete!")
    print(f"Total samples: {total}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Unreadable videos: {len(failed_videos)}")
    print(f"Unreadable audios: {len(failed_audios)}")

    if missing_files:
        print("\n📁 Missing files:")
        for path in missing_files:
            print(f" - {path}")

    if failed_videos:
        print("\n🎞 Failed to read videos:")
        for path in failed_videos:
            print(f" - {path}")

    if failed_audios:
        print("\n🔊 Failed to read audios:")
        for path, error in failed_audios:
            print(f" - {path}: {error}")


# 👇 Your customized paths
json_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"
data_dir = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF"

sanity_check(json_path, data_dir)
