from dataset_loader import get_data_loaders, MultiModalDeepfakeDataset
import json
import os
import time

# Print basic info
print("Testing dataset loading...")
json_path = "D:/Bunny/Deepfake/backend/combined_data/LAV-DF/metadata.json"
data_dir = "D:/Bunny/Deepfake/backend/combined_data/LAV-DF"

# Check if files exist
print(f"JSON file exists: {os.path.exists(json_path)}")
print(f"Data directory exists: {os.path.exists(data_dir)}")

# Try to load JSON
try:
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Successfully loaded JSON with {len(data)} entries")
    print(f"First entry: {data[0]}")
except Exception as e:
    print(f"Error loading JSON: {e}")

# Monkey patch the MultiModalDeepfakeDataset._validate_dataset method to print debug info
original_validate = MultiModalDeepfakeDataset._validate_dataset

def debug_validate(self):
    print("Starting dataset validation...")
    print("This might take a long time for large datasets!")
    start_time = time.time()
    
    # For debugging, only use first 5 entries
    max_entries = min(5, len(self.data))
    print(f"DEBUG: Using first {max_entries} entries only")
    valid_indices = []
    
    for idx in range(max_entries):
        print(f"Validating sample {idx+1}/{max_entries}...")
        try:
            item = self.data[idx]
            file_path = os.path.join(self.data_dir, item['file'])
            print(f"  Checking file: {file_path}")
            print(f"  File exists: {os.path.exists(file_path)}")
            
            # Basic validation
            if os.path.exists(file_path):
                valid_indices.append(idx)
            
        except Exception as e:
            print(f"  Error validating sample {idx}: {e}")
    
    elapsed = time.time() - start_time
    print(f"Validation completed in {elapsed:.2f} seconds")
    print(f"Found {len(valid_indices)} valid samples out of {max_entries} checked")
    return valid_indices

# Apply monkey patch
MultiModalDeepfakeDataset._validate_dataset = debug_validate

# Now try with correct parameters
try:
    print("\n===== Starting dataset loading with correct parameters =====")
    
    # First try creating just the dataset object - check the actual parameters in your file
    print("Creating dataset object...")
    dataset = MultiModalDeepfakeDataset(
        json_path=json_path,
        data_dir=data_dir,
        phase='train',
        detect_faces=False,
        compute_spectrograms=False,
        temporal_features=False
        # Removed max_samples as it's not accepted by the class directly
    )
    print(f"Dataset created with {len(dataset)} samples")
    
    # Now try the full dataloader setup - this should handle max_samples correctly
    print("\nCreating data loaders with max_samples=5...")
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        json_path=json_path,
        data_dir=data_dir,
        batch_size=2,
        validation_split=0.2,
        test_split=0.1,
        shuffle=True,
        num_workers=0,
        max_samples=5,  # This should work at the get_data_loaders level
        detect_faces=False,
        compute_spectrograms=False,
        temporal_features=False
    )
    print(f"Success! Created loaders with {len(train_loader)} training batches")
        
except Exception as e:
    print(f"Error loading dataset: {e}")
    import traceback
    traceback.print_exc()