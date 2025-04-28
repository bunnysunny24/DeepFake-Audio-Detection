import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch

# Define dataset path (modify according to your directory)
DATASET_PATH = r"D:\Bunny\Deepfake\backend\voice_data\The Fake-or-Real (FoR) Dataset"
AUDIO_DURATION = 2  # Using 2-second clips
SAMPLE_RATE = 16000  # Standard for ASR models
N_MELS = 80  # Number of Mel bands

# Augmentations (can be turned on/off)
APPLY_AUGMENTATION = True
audio_augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
])

# Function to process and extract Mel-Spectrograms
def process_audio(file_path, save_path, apply_augment=False):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Ensure 2-second duration
        if len(y) < SAMPLE_RATE * AUDIO_DURATION:
            y = np.pad(y, (0, SAMPLE_RATE * AUDIO_DURATION - len(y)), mode='constant')

        # Apply augmentation
        if apply_augment:
            y = audio_augmenter(samples=y, sample_rate=SAMPLE_RATE)

        # Compute Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=sr//2)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Save Mel-Spectrogram as an image
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel")
        plt.axis("off")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

# Process all files in dataset
def process_dataset():
    for mode in ["training", "testing", "validation"]:
        for category in ["real", "fake"]:
            folder_path = os.path.join(DATASET_PATH, "for-norm", mode, category)
            save_dir = os.path.join(DATASET_PATH, "processed_spectrograms", mode, category)

            # ✅ Check if the folder exists before processing
            if not os.path.exists(folder_path):
                print(f"⚠️ Warning: Skipping missing folder {folder_path}")
                continue

            os.makedirs(save_dir, exist_ok=True)

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(folder_path, file_name)
                    save_path = os.path.join(save_dir, f"{file_name}.png")

                    process_audio(file_path, save_path, apply_augment=APPLY_AUGMENTATION)
                    print(f"✅ Processed: {file_name}")

if __name__ == "__main__":
    process_dataset()

