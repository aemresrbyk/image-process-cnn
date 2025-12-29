
# ==========================================
# RUN THIS SCRIPT IN GOOGLE COLAB
# ==========================================
# Instructions:
# 1. Go to https://colab.research.google.com/
# 2. Create a New Notebook.
# 3. Go to "Runtime" -> "Change runtime type" -> Select "T4 GPU".
# 4. In the "Files" tab (left sidebar), upload your 'dataset.zip' file.
# 5. Copy and Paste ALL this code into a code cell and run it.
# ==========================================

import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight

# 1. Setup & Unzip
if not os.path.exists("dataset"):
    if os.path.exists("dataset.zip"):
        print("Found dataset.zip. Unzipping...")
        with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Unzipped successfully!")
    else:
        print("ERROR: Please upload 'dataset.zip' to the Colab Files tab first!")
        exit()

# 2. Configuration
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50 

def train_cnn_model():
    print("Initializing Data Generators...")
    
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Loading Images...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical'
    )
    
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False
    )
    
    # Calculate Class Weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")
    
    # Build CNN Architecture
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 2
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 3
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Flatten & Dense
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(7, activation='softmax'))
    
    # Compile
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    checkpoint = ModelCheckpoint("emotion_model_cnn.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
    
    print("Starting Training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights_dict
    )
    
    print("Training Complete. Saving model...")
    model.save("emotion_model_cnn.keras")
    
    # Trigger Download
    try:
        from google.colab import files
        files.download('emotion_model_cnn.keras')
        print("Download started!")
    except ImportError:
        print("Not running in Colab? Model saved locally.")

if __name__ == "__main__":
    train_cnn_model()
