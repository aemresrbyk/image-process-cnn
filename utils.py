import cv2
import numpy as np

def apply_dip_preprocessing(image):
    """
    DIP (Digital Image Processing) Ön İşleme Adımları
    """
    # 1. Griye çevir (Eğer renkliyse)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 2. Histogram Eşitleme (Kontrast artırma)
    equ = cv2.equalizeHist(gray)
    
    # 3. Gaussian Blur (Gürültü azaltma)
    blurred = cv2.GaussianBlur(equ, (3, 3), 0)
    
    return blurred

def apply_post_processing_filters(frame, face_coords, emotion):
    """
    Görsel Efektler (YAZILAR KALDIRILDI)
    Sadece görüntüye renk/efekt verir, yazı yazmaz.
    """
    x, y, w, h = [int(c) for c in face_coords]
    
    # Koordinat kontrolü (Ekran dışına taşma engelleme)
    rows, cols, _ = frame.shape
    if x < 0 or y < 0 or x+w > cols or y+h > rows:
        return frame

    roi = frame[y:y+h, x:x+w]
    if roi.size == 0: return frame
    
    # --- SADECE GÖRSEL EFEKTLER (Yazı kodları silindi) ---

    if emotion == 'Happy':
        # Efekt: Sarı/Sıcak Ton
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 40)
        roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif emotion == 'Sad':
        # Efekt: Mavi Ton
        blue_tint = np.zeros_like(roi)
        blue_tint[:, :, 0] = 50 
        roi = cv2.add(roi, blue_tint)

    elif emotion == 'Angry':
        # Efekt: Kırmızı Ton + Keskinleştirme
        red_tint = np.zeros_like(roi)
        red_tint[:, :, 2] = 40 
        roi = cv2.add(roi, red_tint)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        roi = cv2.filter2D(roi, -1, kernel)
    
    elif emotion == 'Surprise':
        # Efekt: Negatif (Ters Renkler)
        roi = cv2.bitwise_not(roi)
    
    elif emotion == 'Fear':
        # Efekt: Mor/Karanlık
        roi = cv2.addWeighted(roi, 0.5, np.zeros_like(roi), 0.5, -50)
    
    else:
        # Neutral - Hiçbir şey yapma
        pass

    # İşlenmiş ROI'yi geri yapıştır
    try:
        frame[y:y+h, x:x+w] = roi
    except:
        pass

    return frame