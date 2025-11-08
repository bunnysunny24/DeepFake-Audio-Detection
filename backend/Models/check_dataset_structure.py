import json

# Check LAV-DF structure
lavdf_data = json.load(open('F:/deepfake/backup/LAV-DF/metadata.json'))
fake_sample = next(x for x in lavdf_data if x['n_fakes'] == 1)

print("=" * 70)
print("LAV-DF STRUCTURE")
print("=" * 70)
print(f"Fake file: {fake_sample['file']}")
print(f"Original: {fake_sample.get('original', 'N/A')}")
print(f"n_fakes: {fake_sample['n_fakes']}")
print()

# Check Samsung metadata
samsung_data = json.load(open('F:/deepfake/backup/Models/samsung_metadata.json'))
fake_samsung = next(x for x in samsung_data if x['n_fakes'] == 1)
real_samsung = next(x for x in samsung_data if x['n_fakes'] == 0)

print("=" * 70)
print("SAMSUNG STRUCTURE")
print("=" * 70)
print(f"Fake file: {fake_samsung['file']}")
print(f"Original: {fake_samsung.get('original', 'N/A')}")
print()
print(f"Real file: {real_samsung['file']}")
print(f"Original: {real_samsung.get('original', 'N/A')}")
print()

print("=" * 70)
print("UNDERSTANDING THE STRUCTURE")
print("=" * 70)
print()
print("SAMSUNG:")
print("  real/id00018/00181.mp4 = REAL original video")
print("  fake/id00018/00181_id00029_wavtolip.mp4 = FAKE of 00181.mp4")
print("  -> The fake name starts with original file number (00181)")
print()
print("LAV-DF:")
print("  train/000001.mp4 = REAL original")
print("  train/fake/000002.mp4 = FAKE of 000001.mp4")
print("  -> Metadata has 'original' field pointing to real video")
print()
