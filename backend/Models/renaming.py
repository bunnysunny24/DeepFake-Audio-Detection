import os

# Specify the directory where the files are located
directory = "D:/Bunny/Deepfake/backend/image_data/segmented_mediapipe"

# Loop through all the files in the directory
for filename in os.listdir(directory):
    # Check if the file has '_seg' in its name and ends with '.jpg'
    if '_seg' in filename and filename.endswith('.jpg'):
        # Construct the new file name by removing '_seg' and changing the extension to '.png'
        new_filename = filename.replace('_seg', '').replace('.jpg', '.png')
        
        # Get the full paths of the old and new file names
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')
