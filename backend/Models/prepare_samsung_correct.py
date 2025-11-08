"""
CORRECT Samsung FakeAVCeleb metadata preparation
Handles ALL 3 types of videos:
1. real/idXXXXX/NNNNN.mp4 - REAL video (n_fakes=0)
2. real/idXXXXX/NNNNN_fake.mp4 - FAKE video (n_fakes=1, original=NNNNN.mp4)
3. fake/idXXXXX/NNNNN_idYYYYY_*.mp4 - FAKE video (n_fakes=1, original=NNNNN.mp4)
"""
import os
import json
import subprocess
from tqdm import tqdm
from pathlib import Path

samsung_root = r"F:\deepfake\backup\SAMSUNG\fakeavceleb"
output_file = r"F:\deepfake\backup\Models\samsung_metadata_correct.json"

def extract_audio(video_path, audio_path):
    """Extract audio from video using ffmpeg."""
    if os.path.exists(audio_path):
        return True
    
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            audio_path, '-y', '-loglevel', 'error'
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    except:
        return False

def get_video_duration(video_path):
    """Get video duration using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except:
        return 0.0

metadata = []

# ============================================================================
# Process REAL folder (contains both real and fake videos)
# ============================================================================
print("="*70)
print("Processing REAL folder")
print("="*70)

real_folder = os.path.join(samsung_root, "real")
person_folders = [d for d in os.listdir(real_folder) if os.path.isdir(os.path.join(real_folder, d))]

real_count = 0
fake_in_real_count = 0

for person_id in tqdm(person_folders, desc="Real folder"):
    person_path = os.path.join(real_folder, person_id)
    videos = [f for f in os.listdir(person_path) if f.endswith('.mp4')]
    
    for video_file in videos:
        video_path = os.path.join(person_path, video_file)
        audio_path = video_path.replace('.mp4', '.wav')
        
        # Extract audio
        extract_audio(video_path, audio_path)
        
        # Get duration
        duration = get_video_duration(video_path)
        
        # Determine if real or fake
        is_fake = '_fake.mp4' in video_file
        
        # For fake videos in real folder, the original is the base video
        original_file = None
        if is_fake:
            # 00206_fake.mp4 -> original is 00206.mp4
            base_num = video_file.replace('_fake.mp4', '.mp4')
            original_file = f"{person_id}\\{base_num}"
            fake_in_real_count += 1
        else:
            real_count += 1
        
        # Create metadata entry
        entry = {
            'file': f"{person_id}\\{video_file}",
            'n_fakes': 1 if is_fake else 0,
            'duration': duration,
            'split': 'train',  # Will be assigned later
            'video_frames': int(duration * 25),  # Assuming 25 FPS
            'audio_channels': 1,
            'audio_frames': int(duration * 16000),  # 16kHz
            'fake_periods': [[0, duration]] if is_fake else [],
            'modify_audio': is_fake,
            'modify_video': is_fake,
            'original': original_file,
            'timestamps': [],
            'transcript': ""
        }
        
        metadata.append(entry)

print(f"\n✅ Real folder processed:")
print(f"   Real videos: {real_count}")
print(f"   Fake videos (_fake.mp4): {fake_in_real_count}")

# ============================================================================
# Process FAKE folder (all are fake videos, face-swapped)
# ============================================================================
print("\n" + "="*70)
print("Processing FAKE folder")
print("="*70)

fake_folder = os.path.join(samsung_root, "fake")
person_folders = [d for d in os.listdir(fake_folder) if os.path.isdir(os.path.join(fake_folder, d))]

fake_count = 0

for person_id in tqdm(person_folders, desc="Fake folder"):
    person_path = os.path.join(fake_folder, person_id)
    videos = [f for f in os.listdir(person_path) if f.endswith('.mp4')]
    
    for video_file in videos:
        video_path = os.path.join(person_path, video_file)
        audio_path = video_path.replace('.mp4', '.wav')
        
        # Extract audio
        extract_audio(video_path, audio_path)
        
        # Get duration
        duration = get_video_duration(video_path)
        
        # Parse filename to find original
        # Example: 00206_id00029_wavtolip.mp4 -> original is 00206.mp4
        base_num = video_file.split('_')[0]  # 00206
        original_file = f"{person_id}\\{base_num}.mp4"
        
        fake_count += 1
        
        # Create metadata entry
        entry = {
            'file': f"{person_id}\\{video_file}",
            'n_fakes': 1,
            'duration': duration,
            'split': 'train',  # Will be assigned later
            'video_frames': int(duration * 25),
            'audio_channels': 1,
            'audio_frames': int(duration * 16000),
            'fake_periods': [[0, duration]],
            'modify_audio': True,
            'modify_video': True,
            'original': original_file,
            'timestamps': [],
            'transcript': ""
        }
        
        metadata.append(entry)

print(f"\n✅ Fake folder processed:")
print(f"   Fake videos: {fake_count}")

# ============================================================================
# Save metadata
# ============================================================================
print("\n" + "="*70)
print("Saving metadata")
print("="*70)

with open(output_file, 'w') as f:
    json.dump(metadata, f, indent=2)

total = len(metadata)
real = sum(1 for x in metadata if x['n_fakes'] == 0)
fake = sum(1 for x in metadata if x['n_fakes'] == 1)
with_original = sum(1 for x in metadata if x.get('original'))

print(f"\n✅ Metadata saved to: {output_file}")
print(f"\nSummary:")
print(f"  Total entries: {total:,}")
print(f"  Real videos: {real:,}")
print(f"  Fake videos: {fake:,}")
print(f"  Fake videos with original reference: {with_original:,} ({with_original/fake*100:.1f}%)")
print("="*70)
