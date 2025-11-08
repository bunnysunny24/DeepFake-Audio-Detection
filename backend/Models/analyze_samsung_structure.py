import os
import json

samsung_root = r"F:\deepfake\backup\SAMSUNG\fakeavceleb"

print("="*80)
print("SAMSUNG DATASET STRUCTURE ANALYSIS")
print("="*80)

# Check one person's folder
test_person = "id00020"
test_num = "00206"

real_folder = os.path.join(samsung_root, "real", test_person)
fake_folder = os.path.join(samsung_root, "fake", test_person)

print(f"\nAnalyzing person: {test_person}, video: {test_num}")
print("="*80)

# Files in real folder
print(f"\nIn real/{test_person}/:")
real_files = [f for f in os.listdir(real_folder) if test_num in f]
for f in sorted(real_files):
    print(f"  - {f}")

# Files in fake folder
print(f"\nIn fake/{test_person}/:")
fake_files = [f for f in os.listdir(fake_folder) if f.startswith(test_num)]
print(f"  Found {len(fake_files)} fake videos starting with {test_num}")
for f in sorted(fake_files[:5]):
    print(f"  - {f}")
if len(fake_files) > 5:
    print(f"  ... and {len(fake_files)-5} more")

print("\n" + "="*80)
print("UNDERSTANDING THE STRUCTURE")
print("="*80)

print(f"""
REAL FOLDER (real/{test_person}/):
  1. {test_num}.mp4          <- REAL video (original, n_fakes=0)
  2. {test_num}_fake.mp4     <- FAKE video (manipulated version, n_fakes=1)
                                Original: {test_num}.mp4
  3. {test_num}_text.txt     <- Transcript

FAKE FOLDER (fake/{test_person}/):
  - {test_num}_id00029_wavtolip.mp4  <- FAKE (person {test_num}'s face on id00029)
                                        Original: {test_num}.mp4
  - {test_num}_id00049_wavtolip.mp4  <- FAKE (person {test_num}'s face on id00049)
                                        Original: {test_num}.mp4
  ... etc
  
So for person {test_person}, video {test_num}:
  - 1 REAL video ({test_num}.mp4)
  - 1 FAKE in real folder ({test_num}_fake.mp4)
  - {len(fake_files)} FAKE in fake folder (face-swaps)
  - Total: 1 real + {1 + len(fake_files)} fakes
""")

# Check what our metadata has
print("="*80)
print("CHECKING OUR SAMSUNG METADATA")
print("="*80)

samsung_meta = json.load(open(r'F:\deepfake\backup\Models\samsung_metadata.json'))

# Find entries for this person
entries_206_real = [x for x in samsung_meta if '00206.mp4' in x['file'] and 'fake' not in x['file'].lower()]
entries_206_fake_in_real = [x for x in samsung_meta if '00206_fake.mp4' in x['file']]
entries_206_in_fake_folder = [x for x in samsung_meta if 'id00020' in x['file'] and x['file'].startswith('id00020\\00206_id')]

print(f"\nEntries for 00206.mp4 (should be REAL, n_fakes=0):")
for entry in entries_206_real[:1]:
    print(f"  File: {entry['file']}")
    print(f"  n_fakes: {entry['n_fakes']}")
    print(f"  modify_video: {entry.get('modify_video', 'N/A')}")

print(f"\nEntries for 00206_fake.mp4 (should be FAKE, n_fakes=1):")
for entry in entries_206_fake_in_real[:1]:
    print(f"  File: {entry['file']}")
    print(f"  n_fakes: {entry['n_fakes']}")
    print(f"  modify_video: {entry.get('modify_video', 'N/A')}")
    print(f"  original: {entry.get('original', 'N/A')}")

print(f"\nEntries in fake folder (should be FAKE, n_fakes=1):")
print(f"  Total found: {len(entries_206_in_fake_folder)}")
if entries_206_in_fake_folder:
    entry = entries_206_in_fake_folder[0]
    print(f"  Example:")
    print(f"    File: {entry['file']}")
    print(f"    n_fakes: {entry['n_fakes']}")
    print(f"    modify_video: {entry.get('modify_video', 'N/A')}")
    print(f"    original: {entry.get('original', 'N/A')}")

print("\n" + "="*80)
print("CORRECT ORIGINAL MAPPINGS SHOULD BE:")
print("="*80)
print(f"""
1. real/id00020/00206.mp4 (REAL)
   - n_fakes: 0
   - original: None

2. real/id00020/00206_fake.mp4 (FAKE)
   - n_fakes: 1
   - original: id00020/00206.mp4 (the real video in same folder)

3. fake/id00020/00206_id00029_wavtolip.mp4 (FAKE)
   - n_fakes: 1
   - original: id00020/00206.mp4 (the real video from real folder)
""")
