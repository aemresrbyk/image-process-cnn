import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- AYARLAR ---
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
IMG_SIZE = 96 
BATCH_SIZE = 32
EPOCHS = 20 # İnce ayar için 20 tur daha

def run_fine_tuning():
    print("Fine-Tuning (İnce Ayar) Modu Başlatılıyor...")
    
    # 1. Mevcut Modeli Yükle (Az önce eğittiğimiz model)
    try:
        model = load_model("emotion_model_transfer.keras")
        print("Mevcut model yüklendi, üzerine eğitim yapılacak.")
    except:
        print("HATA: 'emotion_model_transfer.keras' bulunamadı! Önce train_transfer.py çalıştırılmalı.")
        return

    # 2. Modelin İçindeki Kilidi Aç (Base Model'e ulaş)
    # Modelin ilk katmanı (katman 0) MobileNetV2'dir.
    base_model = model.layers[0]
    base_model.trainable = True # Kilidi açıyoruz!
    
    # Ancak her şeyi açarsak bilgiler bozulabilir. 
    # Sadece son 50 katmanı eğitime açalım, gerisi (temel çizgiler) sabit kalsın.
    fine_tune_at = 100 # MobileNetV2 toplam 155 katmanlıdır.
    
    # İlk 100 katmanı dondur, sonrasını serbest bırak
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    print(f"Toplam katman: {len(base_model.layers)}")
    print(f"{fine_tune_at}. katmandan sonrası eğitime açıldı.")

    # 3. Yeniden Derle (ÇOK ÖNEMLİ: Learning Rate çok düşük olmalı)
    # Çok hızlı öğrenmeye çalışırsa bildiklerini unutur. O yüzden 10 kat yavaş (1e-5) öğrenecek.
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-5), # Çok düşük hız
                  metrics=['accuracy'])
    
    model.summary()

    # 4. Veri Hazırlığı (Aynısı)
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
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, color_mode='rgb', class_mode='categorical', shuffle=True
    )
    
    # Test seti
    if os.path.exists(TEST_DIR):
        validation_generator = val_datagen.flow_from_directory(
            TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, color_mode='rgb', class_mode='categorical', shuffle=False
        )
    else:
        validation_generator = train_datagen.flow_from_directory(
            TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, color_mode='rgb', class_mode='categorical', subset='validation'
        )

    # 5. Eğitimi Başlat
    checkpoint = ModelCheckpoint("emotion_model_transfer_tuned.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    
    print("Fine-Tuning Başlıyor... (Bu işlem doğruluğu artıracak)")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    print("Fine-Tuning Tamamlandı! Yeni dosya: emotion_model_transfer_tuned.keras")

if __name__ == "__main__":
    run_fine_tuning()