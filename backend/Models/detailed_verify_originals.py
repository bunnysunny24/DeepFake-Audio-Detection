import json
import os

# Load metadata
metadata = json.load(open('F:/deepfake/backup/COMBINED_DATASET/metadata.json'))

print("="*80)
print("DETAILED VERIFICATION OF ORIGINAL FILE MATCHING")
print("="*80)

# ============================================================================
# LAV-DF VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("LAV-DF: Checking if fake videos correctly reference their original real videos")
print("="*80)

lavdf_fakes = [x for x in metadata if x['n_fakes'] > 0 and 'samsung' not in x['file']]
print(f"\nTotal LAV-DF fake videos: {len(lavdf_fakes):,}")

# Check 10 random examples
import random
samples = random.sample(lavdf_fakes, min(10, len(lavdf_fakes)))

for i, fake_entry in enumerate(samples, 1):
    fake_file = fake_entry['file']
    original_file = fake_entry.get('original', 'N/A')
    
    # Full paths
    fake_path = os.path.join('F:/deepfake/backup/COMBINED_DATASET', fake_file)
    original_path = os.path.join('F:/deepfake/backup/COMBINED_DATASET', original_file) if original_file != 'N/A' else None
    
    # Check existence
    fake_exists = os.path.exists(fake_path)
    original_exists = os.path.exists(original_path) if original_path else False
    
    # Find the original entry in metadata
    original_entry = None
    if original_file != 'N/A':
        original_entry = next((x for x in metadata if x['file'] == original_file), None)
    
    print(f"\nExample {i}:")
    print(f"  Fake video: {fake_file}")
    print(f"    - Exists: {'✅' if fake_exists else '❌'}")
    print(f"    - n_fakes: {fake_entry['n_fakes']}")
    
    if original_file != 'N/A':
        print(f"  Original video: {original_file}")
        print(f"    - Exists: {'✅' if original_exists else '❌'}")
        if original_entry:
            print(f"    - n_fakes: {original_entry['n_fakes']} {'✅ (should be 0)' if original_entry['n_fakes'] == 0 else '❌ (ERROR: should be 0!)'}")
        else:
            print(f"    - ❌ ERROR: Original not found in metadata!")
    else:
        print(f"  Original: ❌ NOT SET")

# ============================================================================
# SAMSUNG VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("SAMSUNG: Checking if fake videos correctly reference their original real videos")
print("="*80)

samsung_fakes = [x for x in metadata if x['n_fakes'] > 0 and 'samsung' in x['file']]
print(f"\nTotal Samsung fake videos: {len(samsung_fakes):,}")

# Load original Samsung metadata to verify filename parsing
samsung_original_meta = json.load(open('F:/deepfake/backup/Models/samsung_metadata.json'))

# Check 10 random examples
samples = random.sample(samsung_fakes, min(10, len(samsung_fakes)))

for i, fake_entry in enumerate(samples, 1):
    fake_file = fake_entry['file']
    original_file = fake_entry.get('original', 'N/A')
    
    # Full paths
    fake_path = os.path.join('F:/deepfake/backup/COMBINED_DATASET', fake_file)
    original_path = os.path.join('F:/deepfake/backup/COMBINED_DATASET', original_file) if original_file != 'N/A' else None
    
    # Check existence
    fake_exists = os.path.exists(fake_path)
    original_exists = os.path.exists(original_path) if original_path else False
    
    # Find the original entry in metadata
    original_entry = None
    if original_file != 'N/A':
        original_entry = next((x for x in metadata if x['file'] == original_file), None)
    
    # Try to find the original Samsung metadata entry to verify the relationship
    # Extract the original Samsung filename from combined metadata
    samsung_original_filename = None
    if 'samsung_metadata_original' in fake_entry:
        samsung_original_filename = fake_entry['samsung_metadata_original']
    
    print(f"\nExample {i}:")
    print(f"  Fake video: {os.path.basename(fake_file)}")
    print(f"    - Exists: {'✅' if fake_exists else '❌'}")
    print(f"    - n_fakes: {fake_entry['n_fakes']}")
    
    if original_file != 'N/A':
        print(f"  Original video: {os.path.basename(original_file)}")
        print(f"    - Exists: {'✅' if original_exists else '❌'}")
        if original_entry:
            print(f"    - n_fakes: {original_entry['n_fakes']} {'✅ (should be 0)' if original_entry['n_fakes'] == 0 else '❌ (ERROR: should be 0!)'}")
        else:
            print(f"    - ❌ ERROR: Original not found in metadata!")
    else:
        print(f"  Original: ❌ NOT SET")

# ============================================================================
# OVERALL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)

# Count correct mappings
lavdf_correct = sum(1 for x in lavdf_fakes if x.get('original') and 
                    any(y['file'] == x['original'] and y['n_fakes'] == 0 for y in metadata))
samsung_correct = sum(1 for x in samsung_fakes if x.get('original') and 
                      any(y['file'] == x['original'] and y['n_fakes'] == 0 for y in metadata))

print(f"\nLAV-DF:")
print(f"  Total fake videos: {len(lavdf_fakes):,}")
print(f"  With valid original reference: {lavdf_correct:,} ({lavdf_correct/len(lavdf_fakes)*100:.1f}%)")

print(f"\nSamsung:")
print(f"  Total fake videos: {len(samsung_fakes):,}")
print(f"  With valid original reference: {samsung_correct:,} ({samsung_correct/len(samsung_fakes)*100:.1f}%)")

total_fakes = len(lavdf_fakes) + len(samsung_fakes)
total_correct = lavdf_correct + samsung_correct

print(f"\nOVERALL:")
print(f"  Total fake videos: {total_fakes:,}")
print(f"  Correctly mapped to originals: {total_correct:,} ({total_correct/total_fakes*100:.1f}%)")

if total_correct/total_fakes > 0.95:
    print(f"\n✅ EXCELLENT: 95%+ of fake videos have correct original references!")
elif total_correct/total_fakes > 0.80:
    print(f"\n⚠️ GOOD: 80%+ mapping, but some improvements possible")
else:
    print(f"\n❌ WARNING: Less than 80% correctly mapped!")

print("="*80)
