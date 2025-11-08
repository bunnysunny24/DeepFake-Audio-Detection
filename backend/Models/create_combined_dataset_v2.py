"""
Create COMBINED_DATASET with proper original file mapping for temporal consistency.
This version correctly handles:
- Samsung: Fake videos reference their original real videos
- LAV-DF: Fake videos already have 'original' field in metadata
"""
import os
import json
import shutil
from tqdm import tqdm
from pathlib import Path

# Paths
lavdf_root = r"F:\deepfake\backup\LAV-DF"
samsung_root = r"F:\deepfake\backup\SAMSUNG\fakeavceleb"
combined_root = r"F:\deepfake\backup\COMBINED_DATASET"

# Load metadata
print("Loading metadata files...")
lavdf_metadata = json.load(open(os.path.join(lavdf_root, 'metadata.json')))
samsung_metadata = json.load(open(r'F:\deepfake\backup\Models\samsung_metadata.json'))

print(f"LAV-DF: {len(lavdf_metadata):,} videos")
print(f"Samsung: {len(samsung_metadata):,} videos")

# Create combined dataset folder structure
print("\nCreating folder structure...")
for split in ['train', 'test', 'dev']:
    for label in ['real', 'fake']:
        os.makedirs(os.path.join(combined_root, split, label), exist_ok=True)

combined_metadata = []

# ============================================================================
# PART 1: PROCESS LAV-DF (keep original file references)
# ============================================================================
print("\n" + "="*70)
print("PROCESSING LAV-DF VIDEOS")
print("="*70)

for entry in tqdm(lavdf_metadata, desc="LAV-DF"):
    src_video = os.path.join(lavdf_root, entry['file'])
    src_audio = src_video.replace('.mp4', '.wav')
    
    # Determine destination based on split and n_fakes
    split = entry['split']
    label = 'fake' if entry['n_fakes'] > 0 else 'real'
    filename = os.path.basename(entry['file'])
    
    dst_video = os.path.join(combined_root, split, label, filename)
    dst_audio = dst_video.replace('.mp4', '.wav')
    
    # Copy files
    if os.path.exists(src_video) and not os.path.exists(dst_video):
        shutil.copy2(src_video, dst_video)
    if os.path.exists(src_audio) and not os.path.exists(dst_audio):
        shutil.copy2(src_audio, dst_audio)
    
    # Update metadata with new path
    new_entry = entry.copy()
    new_entry['file'] = f"{split}/{label}/{filename}"
    
    # Update original path if it exists
    if 'original' in entry and entry['original']:
        original_filename = os.path.basename(entry['original'])
        # Original videos are always in 'real' folder
        original_split = entry['original'].split('/')[0]  # Get split from original path
        new_entry['original'] = f"{original_split}/real/{original_filename}"
        
        # Copy original if it's a fake video and original exists
        if entry['n_fakes'] > 0:
            src_original = os.path.join(lavdf_root, entry['original'])
            dst_original = os.path.join(combined_root, original_split, 'real', original_filename)
            
            if os.path.exists(src_original) and not os.path.exists(dst_original):
                shutil.copy2(src_original, dst_original)
                # Copy original audio too
                src_original_audio = src_original.replace('.mp4', '.wav')
                dst_original_audio = dst_original.replace('.mp4', '.wav')
                if os.path.exists(src_original_audio) and not os.path.exists(dst_original_audio):
                    shutil.copy2(src_original_audio, dst_original_audio)
    
    combined_metadata.append(new_entry)

print(f"✅ Copied {len(lavdf_metadata):,} LAV-DF videos")

# ============================================================================
# PART 2: PROCESS SAMSUNG (parse filename to find original)
# ============================================================================
print("\n" + "="*70)
print("PROCESSING SAMSUNG VIDEOS")
print("="*70)

samsung_counter = 0

for entry in tqdm(samsung_metadata, desc="Samsung"):
    # Parse Samsung file path
    # Example: id00018\00181_id00029_wavtolip.mp4
    file_path = entry['file']
    person_id = file_path.split('\\')[0]  # id00018
    filename = file_path.split('\\')[1]   # 00181_id00029_wavtolip.mp4
    
    # Determine if fake or real
    is_fake = entry['n_fakes'] > 0
    
    # Source paths
    if is_fake:
        src_video = os.path.join(samsung_root, 'fake', file_path.replace('\\', '/'))
        label = 'fake'
        
        # Extract original filename from fake filename
        # 00181_id00029_wavtolip.mp4 -> 00181.mp4
        original_num = filename.split('_')[0]  # 00181
        original_filename = f"{original_num}.mp4"
        original_path = f"{person_id}\\{original_filename}"
        
    else:
        src_video = os.path.join(samsung_root, 'real', file_path.replace('\\', '/'))
        label = 'real'
        original_path = None
    
    src_audio = src_video.replace('.mp4', '.wav')
    
    # Assign to split (80% train, 10% test, 10% dev)
    split_num = samsung_counter % 10
    if split_num < 8:
        split = 'train'
    elif split_num == 8:
        split = 'test'
    else:
        split = 'dev'
    
    samsung_counter += 1
    
    # Rename with samsung_ prefix to avoid conflicts
    new_filename = f"samsung_{label}_{samsung_counter:06d}.mp4"
    dst_video = os.path.join(combined_root, split, label, new_filename)
    dst_audio = dst_video.replace('.mp4', '.wav')
    
    # Copy files
    if os.path.exists(src_video) and not os.path.exists(dst_video):
        shutil.copy2(src_video, dst_video)
    if os.path.exists(src_audio) and not os.path.exists(dst_audio):
        shutil.copy2(src_audio, dst_audio)
    
    # Create metadata entry
    new_entry = entry.copy()
    new_entry['file'] = f"{split}/{label}/{new_filename}"
    
    # For fake videos, set original path
    if is_fake and original_path:
        # The original real video path in Samsung dataset
        src_original = os.path.join(samsung_root, 'real', original_path.replace('\\', '/'))
        
        # Copy original if it exists and we haven't already
        if os.path.exists(src_original):
            # Find a unique name for the original
            # Use the same counter but for real videos
            original_new_filename = f"samsung_real_{person_id}_{original_num}.mp4"
            original_dst_video = os.path.join(combined_root, split, 'real', original_new_filename)
            original_dst_audio = original_dst_video.replace('.mp4', '.wav')
            
            if not os.path.exists(original_dst_video):
                shutil.copy2(src_original, original_dst_video)
                src_original_audio = src_original.replace('.mp4', '.wav')
                if os.path.exists(src_original_audio):
                    shutil.copy2(src_original_audio, original_dst_audio)
            
            # Set original reference in metadata
            new_entry['original'] = f"{split}/real/{original_new_filename}"
        else:
            # Original doesn't exist, leave as None
            new_entry['original'] = None
    else:
        new_entry['original'] = None
    
    combined_metadata.append(new_entry)

print(f"✅ Copied {len(samsung_metadata):,} Samsung videos")

# ============================================================================
# SAVE COMBINED METADATA
# ============================================================================
print("\n" + "="*70)
print("SAVING COMBINED METADATA")
print("="*70)

output_file = os.path.join(combined_root, 'metadata.json')
with open(output_file, 'w') as f:
    json.dump(combined_metadata, f, indent=2)

print(f"✅ Saved {len(combined_metadata):,} entries to {output_file}")

# ============================================================================
# STATISTICS
# ============================================================================
print("\n" + "="*70)
print("COMBINED DATASET STATISTICS")
print("="*70)

total = len(combined_metadata)
fake_count = sum(1 for x in combined_metadata if x['n_fakes'] > 0)
real_count = sum(1 for x in combined_metadata if x['n_fakes'] == 0)
with_original = sum(1 for x in combined_metadata if x.get('original'))

train_count = sum(1 for x in combined_metadata if x['split'] == 'train')
test_count = sum(1 for x in combined_metadata if x['split'] == 'test')
dev_count = sum(1 for x in combined_metadata if x['split'] == 'dev')

print(f"Total videos: {total:,}")
print(f"  Fake: {fake_count:,} ({fake_count/total*100:.1f}%)")
print(f"  Real: {real_count:,} ({real_count/total*100:.1f}%)")
print(f"  With original reference: {with_original:,} ({with_original/total*100:.1f}%)")
print()
print(f"By split:")
print(f"  Train: {train_count:,} ({train_count/total*100:.1f}%)")
print(f"  Test: {test_count:,} ({test_count/total*100:.1f}%)")
print(f"  Dev: {dev_count:,} ({dev_count/total*100:.1f}%)")
print()
print("="*70)
print("✅ COMBINED DATASET CREATION COMPLETE!")
print("="*70)
