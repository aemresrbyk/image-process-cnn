import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# --- AYARLAR ---
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 30
SELECTED_CLASSES = ['happy', 'sad', 'angry', 'neutral']

print(f"🚀 Mini-Xception Başlatılıyor. Hedef Sınıflar: {SELECTED_CLASSES}")

# 1. Veri Hazırlığı
data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

# Train Generator
train_generator = data_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=SELECTED_CLASSES,
    shuffle=True
)

# Validation Generator
if os.path.exists(TEST_DIR):
    validation_generator = val_gen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=SELECTED_CLASSES,
        shuffle=False
    )
else:
    validation_generator = data_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=SELECTED_CLASSES,
        subset='validation'
    )

# 2. Mini-Xception Mimarisi
def mini_xception(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    
    # Giriş
    x = Conv2D(8, (3, 3), padding='same', use_bias=False)(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Bloklar
    for filters in [16, 32, 64, 128]:
        residual = Conv2D(filters, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        
        x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = Add()([x, residual])
    
    # Çıkış
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

model = mini_xception((IMG_SIZE, IMG_SIZE, 1), len(SELECTED_CLASSES))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Adım Sayılarını Hesapla (HATA ÇÖZÜMÜ BURADA)
train_steps = train_generator.samples // BATCH_SIZE
val_steps = validation_generator.samples // BATCH_SIZE

print(f"Eğitim Adımları: {train_steps}, Doğrulama Adımları: {val_steps}")

# Callbacks
checkpoint = ModelCheckpoint("emotion_model_final.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Eğitim
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,   # Keras'a sınırı söylüyoruz
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=val_steps,    # Keras'a sınırı söylüyoruz
    callbacks=[checkpoint, reduce_lr, early_stop]
)

print("✅ Eğitim Tamamlandı! Dosya: emotion_model_final.keras")