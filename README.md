# Deepfake Audio Detection

## Preprocessing

1. Preprocessing the data using `ff.py`

Run the following command to preprocess the data and divide it into training and validation sets:

```bash
python ff.py
```

You will get two folders with the following structure:

```
-- Image-dataset-7
    |-- train
        |-- fake
        |-- real
    |-- validation
        |-- fake
        |-- real
```

## Generating Heatmaps

2. Generating heatmaps using `mediapipe_ff.py`

Run the following command to convert the data into heatmaps:

```bash
python mediapipe_ff.py
```

You will get the following folder structure:

```
-- landmark_heatmaps
    |-- train
        |-- fake
        |-- real
    |-- validation
        |-- fake
        |-- real
```
