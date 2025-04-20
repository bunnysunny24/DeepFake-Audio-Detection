@echo off
echo Installing PyTorch ecosystem (CPU version)...
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

echo Installing other Machine Learning libraries...
pip install transformers
pip install timm
pip install scikit-learn

echo Installing Audio Processing libraries...
pip install librosa
pip install audiomentations
pip install soundfile

echo Installing Image Processing libraries...
pip install opencv-python
pip install albumentations
pip install mediapipe
pip install facenet-pytorch
pip install pillow

echo Installing Data Handling libraries...
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn

echo Installing Utility libraries...
pip install tqdm
pip install pyyaml
pip install wandb
pip install python-dotenv

echo Installing Tracking tools...
pip install tensorboard

echo Installing Video Processing libraries...
pip install imageio
pip install imageio-ffmpeg
pip install moviepy

echo Installation complete!