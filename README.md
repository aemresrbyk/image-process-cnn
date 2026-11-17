# Facial Emotion Recognition (Image Processing + CNN)

**Real-time emotion recognition from a webcam.** The program finds faces in the live video, cleans each face image with classic image processing, and a small convolutional neural network (CNN) labels it as **happy, sad, angry or neutral**, with a confidence score and a colour overlay on the face.

## What it is for

Reading emotion from a face is useful wherever a system should react to the person in front of it: driver drowsiness and stress monitoring, user-experience testing, assistive tools, smart kiosks and human–robot interaction. This project shows the full chain on a normal laptop with no GPU at run time:

**camera → face detection → image preprocessing → CNN → label + overlay**

## How it works

1. **Face detection.** Each frame is converted to grayscale and OpenCV's Haar cascade finds the face regions.
2. **Image preprocessing** (`utils.py`). The face is resized to 48×48, then:
   - **histogram equalization** stretches the contrast, so the model works under dark or uneven light;
   - a **3×3 Gaussian blur** removes sensor noise that could confuse the model;
   - pixels are scaled to 0–1.
3. **Classification** (`train_final.py`). A **Mini-Xception** CNN: depthwise-separable convolutions with residual connections and global average pooling instead of large dense layers. This keeps the model under 1 MB (`emotion_model_final.keras`) and fast enough for real time on a CPU.
4. **Output** (`main.py`). The predicted emotion and its confidence are drawn over the face, with a colour tint per emotion (yellow happy, blue sad, red angry, grey neutral).

## Training

- **Data:** FER-style 48×48 grayscale face images ([Kaggle: emotion-detection-fer](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)), limited to 4 classes for a more reliable model.
- **Augmentation:** small rotations, shifts, zoom and horizontal flips, so the model does not memorise the training faces.
- **Callbacks:** best-model checkpoint on validation accuracy, learning-rate reduction on plateau, early stopping.
- Earlier experiments (a plain CNN, transfer learning) were replaced by Mini-Xception, which gave a much smaller model for similar accuracy.

## Run it

```bash
pip install -r requirements.txt tensorflow
python main.py            # webcam demo, press q to quit
```

To retrain:

```bash
python download_data.py   # fetches the dataset into dataset/
python train_final.py     # writes emotion_model_final.keras
```

`colab_training.py` trains the earlier plain-CNN baseline on Google Colab (free GPU).

## Files

| File | Purpose |
|------|---------|
| `main.py` | Live webcam demo |
| `utils.py` | Preprocessing (equalization, blur) and the colour overlay |
| `train_final.py` | Mini-Xception model and training |
| `evaluate_cnn.py`, `quick_eval.py` | Accuracy and per-class report on the test set |
| `download_data.py` | Dataset download |

## Limits

- Trained on posed, low-resolution faces; accuracy drops on side views, strong shadows or partly covered faces.
- Haar cascade detection is fast but misses faces that a modern detector would find.
