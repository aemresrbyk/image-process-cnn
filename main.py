import cv2
import os
import numpy as np

# ---- DIP FUNCTION ----
from skimage.feature import local_binary_pattern
import joblib

# Global SVM model
svm_model = None
MODEL_PATH = "emotion_model.pkl"

try:
    if os.path.exists(MODEL_PATH):
        svm_model = joblib.load(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

def extract_lbp_texture(img):
    # LBP Parameters
    radius = 3
    n_points = 8 * radius
    method = 'uniform'
    
    # Grayscale dönüşümü (Doku analizi için)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # LBP Hesaplama
    lbp = local_binary_pattern(gray, n_points, radius, method)
    
    # SVM için özellik vektörü (Histogram)
    # Uniform LBP için olası değer sayısı P + 2 dir.
    # Bu sayıyı sabit tutmalıyız ki vektör boyutu her resim için aynı olsun.
    n_bins = int(n_points + 2)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return lbp, hist

def preprocess_frame(img):
    # 1. Gürültü Azaltma (Noise Reduction)
    smoothed = cv2.GaussianBlur(img, (5, 5), 0)

    # 2. Histogram Eşitleme (Histogram Equalization)
    img_yuv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    return result

# ---- FACE DETECTION ----
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Face Detection
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        processed_face_display = np.zeros((200, 200, 3), dtype=np.uint8)
        lbp_display_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        ai_result = "No Face"

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            
            # Process Face
            cleaned_face = preprocess_frame(face_roi)
            processed_face_display = cv2.resize(cleaned_face, (200, 200))
            lbp_map, lbp_hist = extract_lbp_texture(cleaned_face)
            
            # LBP Visualization
            lbp_norm = (lbp_map / lbp_map.max() * 255).astype(np.uint8)
            lbp_display_bgr = cv2.cvtColor(lbp_norm, cv2.COLOR_GRAY2BGR)
            lbp_display_bgr = cv2.resize(lbp_display_bgr, (200, 200))
            
            # AI Prediction
            if svm_model:
                # Reshape for single sample
                prediction = svm_model.predict([lbp_hist])
                ai_result = prediction[0]
            else:
                ai_result = "Model Not Loaded"
                
            break

        # ---- VISUALIZATION ----
        # Ana ekrana AI sonucunu yaz
        cv2.putText(frame, f"Emotion: {ai_result}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Yan paneli oluştur (Bulunan yüz işlemleri)
        # processed_face_display ve lbp_display_bgr alt alta birleştir
        side_panel = np.vstack((processed_face_display, lbp_display_bgr))
        
        # Ana pencereye sığdırmak için height eşitleme yapılabilir ama şimdilik ayrı bir pencere veya yanına ekleme yapalım
        # Basitlik için: Ana frame + Yan panel (yükseklik uyuşmazlığını dolgu ile çözmek yerine basitçe resize yapalım)
        
        # Ana frame'i biraz küçültelim ki sığsın
        frame_resized = cv2.resize(frame, (640, 480))
        side_panel_resized = cv2.resize(side_panel, (200, 480))
        
        final_view = np.hstack((frame_resized, side_panel_resized))

        cv2.imshow("Emotion Recognition Pipeline", final_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


