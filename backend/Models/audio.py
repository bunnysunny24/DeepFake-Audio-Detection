import os

def check_mp4_to_wav(directory):
    """
    Checks if all .mp4 files in the given directory have corresponding .wav files.
    
    Args:
    - directory (str): Path to the directory to check.

    Returns:
    - missing_files (list): List of .mp4 files without corresponding .wav files.
    """
    # Get all files in the directory
    all_files = os.listdir(directory)

    # Separate .mp4 and .wav files
    mp4_files = {os.path.splitext(file)[0] for file in all_files if file.endswith('.mp4')}
    wav_files = {os.path.splitext(file)[0] for file in all_files if file.endswith('.wav')}

    # Check for missing .wav files
    missing_files = [f"{file}.mp4" for file in mp4_files if file not in wav_files]

    if missing_files:
        print(f"⚠️ The following .mp4 files in '{directory}' are missing corresponding .wav files:")
        for missing in missing_files:
            print(f" - {missing}")
    else:
        print(f"✅ All .mp4 files in '{directory}' have corresponding .wav files.")

    return missing_files

if __name__ == "__main__":
    # Define the directories to check
    base_directory = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF"
    folders = ["dev", "test", "train"]

    all_missing_files = {}

    for folder in folders:
        directory = os.path.join(base_directory, folder)
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"\nChecking folder: {directory}")
            missing_files = check_mp4_to_wav(directory)
            if missing_files:
                all_missing_files[folder] = missing_files
        else:
            print(f"❌ The folder '{directory}' does not exist or is not a directory.")

    if all_missing_files:
        print("\n⚠️ Summary of missing .wav files:")
        for folder, missing in all_missing_files.items():
            print(f" - Folder '{folder}':")
            for file in missing:
                print(f"   - {file}")
        print("\nPlease ensure the missing .wav files are generated.")
    else:
        print("\n🎉 All .mp4 files in all folders have corresponding .wav files.")