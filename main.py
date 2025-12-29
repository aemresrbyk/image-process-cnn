import cv2
import numpy as np
import pickle
import tensorflow as tf
from skimage.feature import local_binary_pattern

# utils.py dosyasından fonksiyonları çekiyoruz
# (Bu dosyanın proje klasöründe olduğundan emin ol)
from utils import apply_dip_preprocessing, apply_post_processing_filters

# --- AYARLAR ---
# True:  Yeni eğittiğin Transfer Learning (MobileNetV2) modelini kullanır. (Tavsiye Edilen)
# False: Eski Klasik (LBP + SVM) modelini kullanır.
USE_TRANSFER_LEARNING = True 

# Model Girdi Boyutları
IMG_SIZE_TL = 96   # MobileNetV2 için (96x96)
IMG_SIZE_SVM = 48  # SVM/LBP için (48x48)

# Duygu Etiketleri
EMOTIONS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
            4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# --- MODEL YÜKLEME ---
model = None

if USE_TRANSFER_LEARNING:
    print("MOD: Transfer Learning (MobileNetV2) başlatılıyor...")
    model_path ="emotion_model_transfer_tuned.keras"
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"BAŞARILI: {model_path} yüklendi.")
    except Exception as e:
        print(f"HATA: {model_path} yüklenemedi!")
        print("Lütfen 'python train_transfer.py' kodunu çalıştırıp modeli eğittiğinden emin ol.")
        exit()
else:
    print("MOD: Klasik Yöntem (LBP + SVM) başlatılıyor...")
    model_path = "emotion_model.pkl"
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"BAŞARILI: {model_path} yüklendi.")
    except Exception as e:
        print(f"HATA: {model_path} yüklenemedi!")
        print("Lütfen 'python train_model.py' kodunu çalıştırıp modeli eğittiğinden emin ol.")
        exit()

# --- YARDIMCI FONKSİYON (Sadece Klasik Mod İçin) ---
def extract_lbp_feature_single(image):
    """Tek bir resimden LBP histogramı çıkarır (SVM tahmini için)"""
    radius = 3
    n_points = 24
    # LBP hesapla
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    # Histogram oluştur
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7) # Normalize et
    return hist.reshape(1, -1) # SVM tek satırlık veri bekler

# --- ANA KAMERA DÖNGÜSÜ ---
# Yüz Tespiti için Haar Cascade yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

print("\n--- KAMERA AÇILIYOR ---")
print("Çıkış için 'q' tuşuna basınız.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı!")
        break

    # Yüz tespiti gri görüntüde yapılır
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    current_emotion = "Neutral" # Varsayılan

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w] # Yüz bölgesini (Renkli) kes
        
        # --- TAHMİN BÖLÜMÜ (AI) ---
        
        if USE_TRANSFER_LEARNING:
            # 1. Transfer Learning (MobileNetV2) için Hazırlık
            # Model RGB ve 96x96 boyut istiyor.
            # OpenCV BGR okur, RGB'ye çevirelim (Daha doğru renk analizi için)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, (IMG_SIZE_TL, IMG_SIZE_TL))
            
            # Normalizasyon (0-1 arası) ve Boyut Ekleme (Batch dimension)
            img_pixels = np.expand_dims(roi_resized, axis=0)
            img_pixels = img_pixels / 255.0
            
            # Tahmin
            preds = model.predict(img_pixels, verbose=0)
            max_index = np.argmax(preds[0])
            current_emotion = EMOTIONS[max_index]
            confidence = preds[0][max_index] # Güven oranı (isteğe bağlı kullanılabilir)
            
        else:
            # 2. Klasik Yöntem (LBP + SVM) için Hazırlık
            # Model 48x48 ve Gri istiyor.
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (IMG_SIZE_SVM, IMG_SIZE_SVM))
            
            # DIP Preprocessing (utils.py'den geliyor - Histogram Eşitleme vb.)
            processed_roi = apply_dip_preprocessing(roi_resized)
            
            # LBP Özelliği Çıkar
            lbp_hist = extract_lbp_feature_single(processed_roi)
            
            # Tahmin
            pred_idx = model.predict(lbp_hist)[0]
            current_emotion = EMOTIONS[pred_idx]

        # --- GÖRSELLEŞTİRME ---
        # Yüzün etrafına kutu çiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Duyguyu yaz
        label_text = f"{current_emotion}"
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # --- POST-PROCESSING (Görsel Filtreler) ---
        # Duyguya göre ekranı renklendir veya efekt ver (utils.py)
        frame = apply_post_processing_filters(frame, (x, y, w, h), current_emotion)

    # Ekrana bas
    cv2.imshow('Final Project: DIP + AI Emotion Recognition', frame)

    # 'q' ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()