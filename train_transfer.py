import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- AYARLAR ---
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
# MobileNetV2 en iyi 96x96 ve üzeri renkli (3 kanal) resimlerle çalışır.
# Bizim resimler 48x48 ve gri. Kod içinde bunları otomatik büyüteceğiz.
IMG_SIZE = 96 
BATCH_SIZE = 32
EPOCHS = 20

def train_transfer_learning():
    print("Veri Hazırlanıyor (Transfer Learning Modu)...")
    
    # 1. Veri Üreteçleri (Data Generators)
    # MobileNetV2 renkli resim ister (RGB). 'color_mode="rgb"' diyerek
    # gri resimleri otomatik olarak 3 kanallı hale getiriyoruz.
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Resimler yükleniyor ve 96x96 boyutuna getiriliyor...")
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='rgb', # ÖNEMLİ: Gri resmi RGB yapar
        class_mode='categorical',
        shuffle=True
    )
    
    # Test klasörü kontrolü
    if not os.path.exists(TEST_DIR):
        validation_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            color_mode='rgb',
            class_mode='categorical',
            subset='validation' # Eğer split yapılıyorsa
        )
    else:
        validation_generator = val_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            color_mode='rgb',
            class_mode='categorical',
            shuffle=False
        )

    # --- TRANSFER LEARNING MODELİ (MobileNetV2) ---
    print("MobileNetV2 Yükleniyor (ImageNet ağırlıklarıyla)...")
    
    # Base Model: Google'ın eğittiği model (Kafasını kesiyoruz: include_top=False)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Base modelin ağırlıklarını dondur (Öğrenilenleri unutmasın)
    # İstersen son 20 katmanı eğitime açabilirsin (Fine-tuning) ama şimdilik donduralım.
    base_model.trainable = False 
    
    # Kendi kafamızı ekliyoruz (Custom Head)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax') # 7 Duygu
    ])
    
    # Derleme
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.summary()
    
    # Callbackler
    checkpoint = ModelCheckpoint("emotion_model_transfer.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Eğitim Başlıyor...")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    print("Transfer Learning Tamamlandı! Model: emotion_model_transfer.keras")

if __name__ == "__main__":
    train_transfer_learning()