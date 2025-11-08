import json
import os

# Load metadata
metadata = json.load(open('F:/deepfake/backup/COMBINED_DATASET/metadata.json'))

print("="*70)
print("VERIFYING ORIGINAL FILE MAPPINGS")
print("="*70)
print()

# Check LAV-DF fake videos
lavdf_fakes_with_original = [x for x in metadata if x['n_fakes'] > 0 and 'samsung' not in x['file'] and x.get('original')]
print(f"LAV-DF fake videos with original: {len(lavdf_fakes_with_original):,}")

# Check a few examples
print("\nLAV-DF Examples:")
for i, entry in enumerate(lavdf_fakes_with_original[:3]):
    fake_path = os.path.join('F:/deepfake/backup/COMBINED_DATASET', entry['file'])
    original_path = os.path.join('F:/deepfake/backup/COMBINED_DATASET', entry['original'])
    
    fake_exists = os.path.exists(fake_path)
    original_exists = os.path.exists(original_path)
    
    print(f"\n  Example {i+1}:")
    print(f"    Fake: {entry['file']} {'✅' if fake_exists else '❌'}")
    print(f"    Original: {entry['original']} {'✅' if original_exists else '❌'}")

# Check Samsung fake videos
samsung_fakes_with_original = [x for x in metadata if x['n_fakes'] > 0 and 'samsung' in x['file'] and x.get('original')]
samsung_fakes_total = [x for x in metadata if x['n_fakes'] > 0 and 'samsung' in x['file']]

print(f"\n{'='*70}")
print(f"Samsung fake videos total: {len(samsung_fakes_total):,}")
print(f"Samsung fake videos with original: {len(samsung_fakes_with_original):,}")
print(f"Match rate: {len(samsung_fakes_with_original)/len(samsung_fakes_total)*100:.1f}%")

# Check a few Samsung examples
print("\nSamsung Examples:")
for i, entry in enumerate(samsung_fakes_with_original[:5]):
    fake_path = os.path.join('F:/deepfake/backup/COMBINED_DATASET', entry['file'])
    original_path = os.path.join('F:/deepfake/backup/COMBINED_DATASET', entry['original']) if entry.get('original') else None
    
    fake_exists = os.path.exists(fake_path)
    original_exists = os.path.exists(original_path) if original_path else False
    
    print(f"\n  Example {i+1}:")
    print(f"    Fake: {os.path.basename(entry['file'])} {'✅' if fake_exists else '❌'}")
    if entry.get('original'):
        print(f"    Original: {os.path.basename(entry['original'])} {'✅' if original_exists else '❌'}")
    else:
        print(f"    Original: None ❌")

# Overall statistics
total_fakes = len([x for x in metadata if x['n_fakes'] > 0])
fakes_with_original = len([x for x in metadata if x['n_fakes'] > 0 and x.get('original')])

print(f"\n{'='*70}")
print("OVERALL STATISTICS")
print("="*70)
print(f"Total videos: {len(metadata):,}")
print(f"Total fake videos: {total_fakes:,}")
print(f"Fake videos with original reference: {fakes_with_original:,} ({fakes_with_original/total_fakes*100:.1f}%)")
print(f"Real videos: {len([x for x in metadata if x['n_fakes'] == 0]):,}")
print()
print("✅ Original file mapping is working!" if fakes_with_original/total_fakes > 0.7 else "⚠️ Some originals are missing")
print("="*70)
