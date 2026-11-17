
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Configuration
TEST_DIR = "dataset/test"
IMG_SIZE = 48
BATCH_SIZE = 64
MODEL_PATH = "emotion_model_final.keras"
CLASSES = ["happy", "sad", "angry", "neutral"]  # must match train_final.py order

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    if not os.path.exists(TEST_DIR):
        print(f"Error: Test dataset not found at {TEST_DIR}")
        return

    print("Loading Model...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Preparing Test Data...")
    test_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False  # Important for confusion matrix
    )

    print("Evaluating Model...")
    # Get predictions
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = validation_generator.classes
    
    # Class labels
    class_labels = list(validation_generator.class_indices.keys())

    # Calculate Accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

if __name__ == "__main__":
    evaluate_model()
