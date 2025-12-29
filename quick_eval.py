import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Configuration
TEST_DIR = "dataset/test"
IMG_SIZE = 48
BATCH_SIZE = 64
MODEL_PATH = "emotion_model_cnn.keras"

def evaluate_model_quick():
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

    print("Preparing Test Data (Quick Mode)...")
    test_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True # Shuffle to get a random sample
    )

    print("Evaluating Model on 10 batches...")
    # Evaluate on 10 batches
    results = model.evaluate(validation_generator, steps=10)
    
    print(f"\nQuick Test Accuracy (approx): {results[1] * 100:.2f}%")
    print(f"Test Loss: {results[0]:.4f}")

if __name__ == "__main__":
    evaluate_model_quick()
