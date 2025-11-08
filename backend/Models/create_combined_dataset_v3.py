import json
import shutil
import os
from pathlib import Path
from tqdm import tqdm

print("=" * 70)
print("Creating COMBINED_DATASET v3 (LAV-DF + Samsung Corrected)")
print("=" * 70)

# Paths
LAVDF_BASE = Path(r"F:\deepfake\backup\LAV-DF")
SAMSUNG_BASE = Path(r"F:\deepfake\backup\SAMSUNG\fakeavceleb")
OUTPUT_BASE = Path(r"F:\deepfake\backup\COMBINED_DATASET")
LAVDF_METADATA = LAVDF_BASE / "metadata.json"
SAMSUNG_METADATA = Path(r"F:\deepfake\backup\Models\samsung_metadata_correct.json")

# Load metadata
print("\nLoading metadata...")
with open(LAVDF_METADATA, 'r') as f:
    lavdf_data = json.load(f)
print(f"✅ LAV-DF: {len(lavdf_data)} entries")

with open(SAMSUNG_METADATA, 'r') as f:
    samsung_data = json.load(f)
print(f"✅ Samsung: {len(samsung_data)} entries")

# Create output structure
OUTPUT_BASE.mkdir(exist_ok=True)

# Process LAV-DF
print("\n" + "=" * 70)
print("Processing LAV-DF Dataset")
print("=" * 70)

lavdf_combined = []
lavdf_real_count = 0
lavdf_fake_count = 0
lavdf_with_original = 0

for entry in tqdm(lavdf_data, desc="LAV-DF"):
    video_path = entry['file']
    # Get full paths
    src_video = LAVDF_BASE / video_path
    dst_video = OUTPUT_BASE / "LAVDF" / video_path
    
    # Copy video if exists
    if src_video.exists():
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        if not dst_video.exists():
            shutil.copy2(src_video, dst_video)
        
        # Copy audio if exists
        src_audio = src_video.with_suffix('.wav')
        dst_audio = dst_video.with_suffix('.wav')
        if src_audio.exists() and not dst_audio.exists():
            shutil.copy2(src_audio, dst_audio)
        
        # Copy original video if exists
        if entry.get('original'):
            src_original = LAVDF_BASE / entry['original']
            dst_original = OUTPUT_BASE / "LAVDF" / entry['original']
            if src_original.exists():
                dst_original.parent.mkdir(parents=True, exist_ok=True)
                if not dst_original.exists():
                    shutil.copy2(src_original, dst_original)
                
                # Copy original audio
                src_original_audio = src_original.with_suffix('.wav')
                dst_original_audio = dst_original.with_suffix('.wav')
                if src_original_audio.exists() and not dst_original_audio.exists():
                    shutil.copy2(src_original_audio, dst_original_audio)
                
                lavdf_with_original += 1
        
        # Add to combined metadata
        combined_entry = entry.copy()
        combined_entry['file'] = f"LAVDF\\{video_path}"
        if entry.get('original'):
            combined_entry['original'] = f"LAVDF\\{entry['original']}"
        lavdf_combined.append(combined_entry)
        
        if entry['n_fakes'] == 0:
            lavdf_real_count += 1
        else:
            lavdf_fake_count += 1

print(f"\n✅ LAV-DF processed:")
print(f"   Real videos: {lavdf_real_count}")
print(f"   Fake videos: {lavdf_fake_count}")
print(f"   Fakes with original: {lavdf_with_original} ({lavdf_with_original/lavdf_fake_count*100:.1f}%)")

# Process Samsung
print("\n" + "=" * 70)
print("Processing Samsung Dataset")
print("=" * 70)

samsung_combined = []
samsung_real_count = 0
samsung_fake_count = 0
samsung_with_original = 0

for entry in tqdm(samsung_data, desc="Samsung"):
    video_path = entry['file']
    
    # Samsung files are in real/ or fake/ folders
    # Determine source folder based on video type
    if '_fake.mp4' in video_path or entry['n_fakes'] == 0:
        # Files in real folder (both real and _fake.mp4)
        src_video = SAMSUNG_BASE / "real" / video_path
    else:
        # Files in fake folder (face-swaps)
        src_video = SAMSUNG_BASE / "fake" / video_path
    
    dst_video = OUTPUT_BASE / "SAMSUNG" / video_path
    
    # Copy video if exists
    if src_video.exists():
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        if not dst_video.exists():
            shutil.copy2(src_video, dst_video)
        
        # Copy audio if exists
        src_audio = src_video.with_suffix('.wav')
        dst_audio = dst_video.with_suffix('.wav')
        if src_audio.exists() and not dst_audio.exists():
            shutil.copy2(src_audio, dst_audio)
        
        # Copy original video if exists
        if entry.get('original'):
            # Originals are always in real folder
            src_original = SAMSUNG_BASE / "real" / entry['original']
            dst_original = OUTPUT_BASE / "SAMSUNG" / entry['original']
            if src_original.exists():
                dst_original.parent.mkdir(parents=True, exist_ok=True)
                if not dst_original.exists():
                    shutil.copy2(src_original, dst_original)
                
                # Copy original audio
                src_original_audio = src_original.with_suffix('.wav')
                dst_original_audio = dst_original.with_suffix('.wav')
                if src_original_audio.exists() and not dst_original_audio.exists():
                    shutil.copy2(src_original_audio, dst_original_audio)
                
                samsung_with_original += 1
        
        # Add to combined metadata
        combined_entry = entry.copy()
        combined_entry['file'] = f"SAMSUNG\\{video_path}"
        if entry.get('original'):
            combined_entry['original'] = f"SAMSUNG\\{entry['original']}"
        samsung_combined.append(combined_entry)
        
        if entry['n_fakes'] == 0:
            samsung_real_count += 1
        else:
            samsung_fake_count += 1

print(f"\n✅ Samsung processed:")
print(f"   Real videos: {samsung_real_count}")
print(f"   Fake videos: {samsung_fake_count}")
if samsung_fake_count > 0:
    print(f"   Fakes with original: {samsung_with_original} ({samsung_with_original/samsung_fake_count*100:.1f}%)")
else:
    print(f"   Fakes with original: 0 (no fakes processed)")

# Combine and save
print("\n" + "=" * 70)
print("Saving Combined Metadata")
print("=" * 70)

combined_metadata = lavdf_combined + samsung_combined
output_metadata = OUTPUT_BASE / "metadata.json"

with open(output_metadata, 'w') as f:
    json.dump(combined_metadata, f, indent=2)

print(f"\n✅ Metadata saved: {output_metadata}")

# Final statistics
total_real = lavdf_real_count + samsung_real_count
total_fake = lavdf_fake_count + samsung_fake_count
total_with_original = lavdf_with_original + samsung_with_original

print("\n" + "=" * 70)
print("COMBINED DATASET v3 - Summary")
print("=" * 70)
print(f"\nTotal entries: {len(combined_metadata)}")
print(f"  LAV-DF: {len(lavdf_combined)}")
print(f"  Samsung: {len(samsung_combined)}")
print(f"\nReal videos: {total_real}")
print(f"  LAV-DF: {lavdf_real_count}")
print(f"  Samsung: {samsung_real_count}")
print(f"\nFake videos: {total_fake}")
print(f"  LAV-DF: {lavdf_fake_count}")
print(f"  Samsung: {samsung_fake_count}")
print(f"\nFakes with original reference: {total_with_original}/{total_fake} ({total_with_original/total_fake*100:.1f}%)")
print(f"  LAV-DF: {lavdf_with_original}/{lavdf_fake_count} ({lavdf_with_original/lavdf_fake_count*100:.1f}%)")
print(f"  Samsung: {samsung_with_original}/{samsung_fake_count} ({samsung_with_original/samsung_fake_count*100:.1f}%)")
print(f"\nClass imbalance: {total_fake/total_real:.2f}:1 (fake:real)")
print("=" * 70)
