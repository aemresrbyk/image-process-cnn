import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from utils import apply_dip_preprocessing, apply_post_processing_filters

# --- Configuration ---
MODEL_PATH = "emotion_model_final.keras"
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTIONS = ['happy', 'sad', 'angry', 'neutral'] # MUST match train_final.py order

def main():
    # 1. Load Model
    print("Loading model...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run 'python train_final.py' first to train the model.")
        return

    # 2. Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return
        
    # Load Face Detector
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    print("Starting video stream. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect Faces
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Region of Interest (ROI)
            roi_gray = gray_frame[y:y+h, x:x+w]
            
            # --- PREPROCESSING START ---
            
            # 1. Resize to 48x48 (Model input size)
            roi = cv2.resize(roi_gray, (48, 48))
            
            # 2. DIP Preprocessing (Contrast + Blur)
            # Pass numpy array directly
            roi = apply_dip_preprocessing(roi)
            
            # 3. Normalize (0-1)
            roi = roi.astype("float") / 255.0
            
            # 4. Prepare for Keras (Batch Dimension)
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # --- PREPROCESSING END ---
            
            # Predict
            preds = model.predict(roi, verbose=0)[0]
            label_idx = np.argmax(preds)
            label = EMOTIONS[label_idx]
            confidence = preds[label_idx]
            
            # --- POST-PROCESSING ---
            
            # Apply color filter
            frame = apply_post_processing_filters(frame, (x, y, w, h), label)
            
            # Draw Text (Emotion + Score)
            text_str = f"{label}: {confidence*100:.1f}%"
            cv2.putText(frame, text_str, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw Bounding Box (White)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
        cv2.imshow('Emotion Recognition (Mini-Xception)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()