import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight

# --- CONFIGURATION ---
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
IMG_SIZE = 128  # Upscale for EfficientNet (min 32, but >96 recommended)
BATCH_SIZE = 32
EPOCHS_HEAD = 10   # Epochs for training top layers
EPOCHS_FINE = 30   # Epochs for fine-tuning
MODEL_NAME = "emotion_model_efficientnet.keras"

def train_advanced_model():
    print(f"TensorFlow Version: {tf.__version__}")
    print("Initializing Data Generators...")
    
    # 1. Advanced Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print(f"Loading Images (Upscaling to {IMG_SIZE}x{IMG_SIZE})...")
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='rgb',   # EfficientNet expects 3 channels
        class_mode='categorical',
        shuffle=True
    )
    
    # Validation data
    if os.path.exists(TEST_DIR):
        validation_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            color_mode='rgb',
            class_mode='categorical',
            shuffle=False
        )
    else:
        # Fallback to splitting train data if test dir doesn't exist (unlikely here)
        print("Warning: Test directory not found, using subset of training??")
        validation_generator = None 

    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")

    # --- PHASE 1: Build Model & Train Head ---
    print("\n--- PHASE 1: Training Head (Frozen Base) ---")
    
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(7, activation='softmax')
    ])
    
    # Compile with Label Smoothing
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    model.summary()
    
    checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    
    # Train Head
    history_head = model.fit(
        train_generator,
        epochs=EPOCHS_HEAD,
        validation_data=validation_generator,
        callbacks=[checkpoint],
        class_weight=class_weights_dict
    )
    
    # --- PHASE 2: Fine-Tuning ---
    print("\n--- PHASE 2: Fine-Tuning (Unfreezing Top Layers) ---")
    
    # Unfreeze the last N layers of the base model
    base_model.trainable = True
    
    # Let's freeze the bottom layers and only train the top 50
    # EfficientNetB0 has ~237 layers. 
    for layer in base_model.layers[:-50]:
        layer.trainable = False
        
    print("Re-compiling with low learning rate...")
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5), # Very low learning rate
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
    
    history_fine = model.fit(
        train_generator,
        epochs=EPOCHS_FINE,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights_dict
    )
    
    print(f"Training Complete. Best model saved to {MODEL_NAME}")

if __name__ == "__main__":
    train_advanced_model()
