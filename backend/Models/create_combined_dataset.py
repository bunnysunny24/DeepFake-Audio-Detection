"""
Create a new combined dataset folder with LAV-DF + Samsung FakeAVCeleb.
Organizes into train/test/dev splits with real and fake videos.
Creates symlinks to avoid duplicating files.
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def create_combined_dataset(output_root, lavdf_root, samsung_root):
    """
    Create combined dataset structure:
    COMBINED_DATASET/
        train/
            real/
            fake/
        test/
            real/
            fake/
        dev/
            real/
            fake/
        metadata.json
    """
    output_root = Path(output_root)
    lavdf_root = Path(lavdf_root)
    samsung_root = Path(samsung_root)
    
    print("\n" + "="*70)
    print("CREATING COMBINED LAV-DF + SAMSUNG DATASET")
    print("="*70)
    
    # Create directory structure
    print("\n📁 Creating folder structure...")
    for split in ['train', 'test', 'dev']:
        for label in ['real', 'fake']:
            (output_root / split / label).mkdir(parents=True, exist_ok=True)
    
    # Load LAV-DF metadata
    print(f"\n📂 Loading LAV-DF metadata...")
    lavdf_meta_path = lavdf_root / "metadata.json"
    with open(lavdf_meta_path, 'r') as f:
        lavdf_data = json.load(f)
    
    print(f"   ✅ {len(lavdf_data)} LAV-DF videos")
    
    combined_metadata = []
    copied_files = 0
    
    # Process LAV-DF videos
    print(f"\n🔗 Copying LAV-DF videos...")
    for entry in tqdm(lavdf_data, desc="LAV-DF"):
        try:
            src_file = lavdf_root / entry['file']
            src_audio = src_file.with_suffix('.wav')
            
            if not src_file.exists():
                continue
            
            # Determine label folder
            label = 'real' if entry['n_fakes'] == 0 else 'fake'
            split = entry.get('split', 'train')
            
            # Create destination paths
            dest_folder = output_root / split / label
            dest_file = dest_folder / src_file.name
            dest_audio = dest_folder / src_audio.name
            
            # Copy video file
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
            
            # Copy audio file if exists
            if src_audio.exists() and not dest_audio.exists():
                shutil.copy2(src_audio, dest_audio)
            
            # Update metadata with new path
            new_entry = entry.copy()
            new_entry['file'] = f"{split}/{label}/{src_file.name}"
            new_entry['source_dataset'] = 'LAV-DF'
            combined_metadata.append(new_entry)
            copied_files += 1
            
        except Exception as e:
            print(f"\n⚠️  Error processing {entry['file']}: {e}")
    
    print(f"\n   ✅ Copied {copied_files} LAV-DF videos")
    
    # Process Samsung FakeAVCeleb videos
    print(f"\n📂 Processing Samsung FakeAVCeleb videos...")
    
    samsung_fake_dir = samsung_root / "fake"
    samsung_real_dir = samsung_root / "real"
    
    # Process fake videos
    fake_videos = list(samsung_fake_dir.rglob("*.mp4"))
    print(f"   Found {len(fake_videos)} fake videos")
    
    samsung_copied = 0
    for idx, src_file in enumerate(tqdm(fake_videos[:5000], desc="Samsung Fake")):  # Limit to 5000 for now
        try:
            src_audio = src_file.with_suffix('.wav')
            
            # 80% train, 10% test, 10% dev
            if idx % 10 < 8:
                split = 'train'
            elif idx % 10 == 8:
                split = 'test'
            else:
                split = 'dev'
            
            dest_folder = output_root / split / 'fake'
            
            # Create unique filename to avoid conflicts
            dest_file = dest_folder / f"samsung_fake_{idx:06d}.mp4"
            dest_audio = dest_folder / f"samsung_fake_{idx:06d}.wav"
            
            # Copy video
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
            
            # Extract or copy audio
            if src_audio.exists() and not dest_audio.exists():
                shutil.copy2(src_audio, dest_audio)
            elif not dest_audio.exists():
                # Extract audio
                import subprocess
                subprocess.run([
                    'ffmpeg', '-i', str(src_file),
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '16000', '-ac', '1',
                    str(dest_audio), '-y', '-loglevel', 'error'
                ], check=False)
            
            # Get duration
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(src_file)
            ], capture_output=True, text=True)
            
            duration = float(result.stdout.strip()) if result.returncode == 0 else 5.0
            
            # Add to metadata
            combined_metadata.append({
                'file': f"{split}/fake/samsung_fake_{idx:06d}.mp4",
                'n_fakes': 1,
                'duration': duration,
                'split': split,
                'video_frames': int(duration * 25),
                'audio_channels': 1,
                'audio_frames': 0,
                'fake_periods': [[0, duration]],
                'modify_audio': False,
                'modify_video': True,
                'original': None,
                'timestamps': [],
                'transcript': '',
                'source_dataset': 'Samsung-FakeAVCeleb'
            })
            samsung_copied += 1
            
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
    
    # Process real videos
    real_videos = list(samsung_real_dir.rglob("*.mp4"))
    print(f"\n   Found {len(real_videos)} real videos")
    
    for idx, src_file in enumerate(tqdm(real_videos[:500], desc="Samsung Real")):  # Limit to 500
        try:
            src_audio = src_file.with_suffix('.wav')
            
            # 80% train, 10% test, 10% dev
            if idx % 10 < 8:
                split = 'train'
            elif idx % 10 == 8:
                split = 'test'
            else:
                split = 'dev'
            
            dest_folder = output_root / split / 'real'
            dest_file = dest_folder / f"samsung_real_{idx:06d}.mp4"
            dest_audio = dest_folder / f"samsung_real_{idx:06d}.wav"
            
            # Copy video
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
            
            # Copy/extract audio
            if src_audio.exists() and not dest_audio.exists():
                shutil.copy2(src_audio, dest_audio)
            elif not dest_audio.exists():
                import subprocess
                subprocess.run([
                    'ffmpeg', '-i', str(src_file),
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '16000', '-ac', '1',
                    str(dest_audio), '-y', '-loglevel', 'error'
                ], check=False)
            
            # Get duration
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(src_file)
            ], capture_output=True, text=True)
            
            duration = float(result.stdout.strip()) if result.returncode == 0 else 5.0
            
            combined_metadata.append({
                'file': f"{split}/real/samsung_real_{idx:06d}.mp4",
                'n_fakes': 0,
                'duration': duration,
                'split': split,
                'video_frames': int(duration * 25),
                'audio_channels': 1,
                'audio_frames': 0,
                'fake_periods': [],
                'modify_audio': False,
                'modify_video': False,
                'original': None,
                'timestamps': [],
                'transcript': '',
                'source_dataset': 'Samsung-FakeAVCeleb'
            })
            samsung_copied += 1
            
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
    
    print(f"\n   ✅ Copied {samsung_copied} Samsung videos")
    
    # Save combined metadata
    metadata_path = output_root / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    
    # Print statistics
    total_real = sum(1 for m in combined_metadata if m['n_fakes'] == 0)
    total_fake = sum(1 for m in combined_metadata if m['n_fakes'] == 1)
    
    print("\n" + "="*70)
    print("✅ COMBINED DATASET CREATED")
    print("="*70)
    print(f"📊 Total videos: {len(combined_metadata):,}")
    print(f"   Real: {total_real:,} ({total_real/len(combined_metadata)*100:.1f}%)")
    print(f"   Fake: {total_fake:,} ({total_fake/len(combined_metadata)*100:.1f}%)")
    print(f"\n📁 Location: {output_root.absolute()}")
    print(f"💾 Metadata: {metadata_path.absolute()}")
    print("\n📝 To train:")
    print(f'   Update train_enhanced_model.ps1:')
    print(f'   --json_path "{metadata_path.absolute()}"')
    print(f'   --data_dir "{output_root.absolute()}"')
    print("="*70)

if __name__ == "__main__":
    output_root = r"F:\deepfake\backup\COMBINED_DATASET"
    lavdf_root = r"F:\deepfake\backup\LAV-DF"
    samsung_root = r"F:\deepfake\backup\SAMSUNG\fakeavceleb"
    
    create_combined_dataset(output_root, lavdf_root, samsung_root)
