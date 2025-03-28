import os

train_dir =r"hybrid_model_data/train/images"
val_dir = r"hybrid_model_data/validation/images"

print("Train Image Count:", len(os.listdir(train_dir)))
print("Validation Image Count:", len(os.listdir(val_dir)))
