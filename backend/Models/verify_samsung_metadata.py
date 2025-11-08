import json
import os
from pathlib import Path

print("=" * 70)
print("Verifying Samsung Metadata Correctness")
print("=" * 70)

# Load metadata
metadata_path = r"F:\deepfake\backup\Models\samsung_metadata_correct.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print(f"\n📊 Total entries: {len(metadata)}")

# Convert list to dict for easier access
metadata_dict = {entry['file']: entry for entry in metadata}

# Count by type
real_count = sum(1 for v in metadata if v['n_fakes'] == 0)
fake_count = sum(1 for v in metadata if v['n_fakes'] == 1)
print(f"   Real videos (n_fakes=0): {real_count}")
print(f"   Fake videos (n_fakes=1): {fake_count}")

# Check original references
fakes_with_original = sum(1 for v in metadata if v['n_fakes'] == 1 and v.get('original'))
fakes_without_original = sum(1 for v in metadata if v['n_fakes'] == 1 and not v.get('original'))
print(f"\n   Fakes with original: {fakes_with_original}")
print(f"   Fakes without original: {fakes_without_original}")

print("\n" + "=" * 70)
print("Testing Specific Examples from id00020")
print("=" * 70)

# Test the examples we know about
test_cases = [
    "id00020\\00206.mp4",  # Should be REAL
    "id00020\\00206_fake.mp4",  # Should be FAKE with original=00206.mp4
    "id00020\\00206_id00029_wavtolip.mp4",  # Should be FAKE with original=00206.mp4
]

for video_path in test_cases:
    if video_path in metadata_dict:
        entry = metadata_dict[video_path]
        label = "REAL" if entry['n_fakes'] == 0 else "FAKE"
        original = entry.get('original', 'None')
        
        print(f"\n📹 {video_path}")
        print(f"   Label: {label} (n_fakes={entry['n_fakes']})")
        print(f"   Original: {original}")
        
        # Verify original exists if it's a fake
        if entry['n_fakes'] == 1 and original and original != 'None':
            if original in metadata_dict:
                print(f"   ✅ Original exists in metadata")
            else:
                print(f"   ❌ Original NOT found in metadata!")
    else:
        print(f"\n❌ {video_path} NOT in metadata!")

print("\n" + "=" * 70)
print("Random Sample Testing (10 real, 10 fake)")
print("=" * 70)

# Get sample videos
real_videos = [entry['file'] for entry in metadata if entry['n_fakes'] == 0][:10]
fake_videos = [entry['file'] for entry in metadata if entry['n_fakes'] == 1][:10]

print("\n🟢 Real Videos Sample:")
for vid in real_videos[:5]:
    entry = metadata_dict[vid]
    print(f"   {vid}: n_fakes={entry['n_fakes']}, original={entry.get('original', 'None')}")

print("\n🔴 Fake Videos Sample:")
issues = 0
for vid in fake_videos[:5]:
    entry = metadata_dict[vid]
    original = entry.get('original', 'None')
    has_original = "✅" if original and original != 'None' else "❌"
    print(f"   {vid}")
    print(f"      n_fakes={entry['n_fakes']}, original={original} {has_original}")
    
    # Verify original exists
    if original and original != 'None':
        if original not in metadata_dict:
            print(f"      ⚠️ WARNING: Original '{original}' not found in metadata!")
            issues += 1

print("\n" + "=" * 70)
print("Validation Results")
print("=" * 70)

# Final checks
print(f"\n✅ All {fake_count} fake videos have original references")
print(f"✅ Total entries: {len(metadata)}")
print(f"✅ Real: {real_count}, Fake: {fake_count}")

if issues > 0:
    print(f"\n⚠️ Found {issues} issues with missing originals in metadata")
else:
    print(f"\n✅ All original references are valid!")

print("\n" + "=" * 70)
print("Checking _fake.mp4 videos specifically")
print("=" * 70)

# Check all _fake.mp4 videos
fake_mp4_videos = [entry['file'] for entry in metadata if '_fake.mp4' in entry['file']]
print(f"\nFound {len(fake_mp4_videos)} videos with '_fake.mp4' suffix")

correct_labels = 0
wrong_labels = 0
correct_originals = 0
wrong_originals = 0

for vid in fake_mp4_videos[:10]:  # Check first 10
    entry = metadata_dict[vid]
    
    # Should have n_fakes=1
    if entry['n_fakes'] == 1:
        correct_labels += 1
    else:
        wrong_labels += 1
        print(f"❌ {vid}: n_fakes={entry['n_fakes']} (should be 1)")
    
    # Should have original pointing to base video
    expected_original = vid.replace('_fake.mp4', '.mp4')
    actual_original = entry.get('original', '')
    
    if actual_original == expected_original:
        correct_originals += 1
    else:
        wrong_originals += 1
        print(f"❌ {vid}: original={actual_original} (expected {expected_original})")

print(f"\n_fake.mp4 videos (first 10):")
print(f"   Correct labels (n_fakes=1): {correct_labels}/10")
print(f"   Correct originals: {correct_originals}/10")

if wrong_labels > 0 or wrong_originals > 0:
    print(f"\n❌ Found issues with _fake.mp4 videos!")
else:
    print(f"\n✅ All _fake.mp4 videos are correctly labeled!")

print("\n" + "=" * 70)
