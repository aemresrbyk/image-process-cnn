import cv2
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import local_binary_pattern

# Import functions from main.py
# (Ensure main.py is in the same directory)
from main import preprocess_frame, extract_lbp_texture

# Configuration
DATASET_PATH = "dataset/train"
MODEL_FILENAME = "emotion_model.pkl"
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
SAMPLES_PER_CLASS = 100000  # Use all data (max ~7000 in happy folder)

def train_svm_model():
    print("Loading dataset and extracting features...")
    
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    X = []
    y = []
    
    # Traverse through emotion folders (angry, happy, etc.)
    for emotion_label in os.listdir(DATASET_PATH):
        emotion_dir = os.path.join(DATASET_PATH, emotion_label)
        
        if not os.path.isdir(emotion_dir):
            continue
            
        print(f"Processing Class: {emotion_label}")
        
        count = 0
        for image_name in os.listdir(emotion_dir):
            if count >= SAMPLES_PER_CLASS:
                break
                
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            image_path = os.path.join(emotion_dir, image_name)
            
            # 1. Load Image (Grayscale by default for FER dataset)
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            # Note: FER images are already 48x48 cropped faces.
            # We DONT need to detect faces again (it will fail on 48x48).
            # We can use the image directly.
            
            # 2. Preprocessing & Texture
            # Resize slightly up for better Gaussian/LBP calculation if needed, 
            # but 48x48 is fine. LBP radius=3 might be too big for 48x48.
            # Let's resize to 100x100 for consistency with our pipeline.
            target_face = cv2.resize(img, (100, 100))
            
            cleaned_face = preprocess_frame(target_face)
            
            # 3. Feature Extraction (LBP)
            _, lbp_hist = extract_lbp_texture(cleaned_face)
            
            X.append(lbp_hist)
            y.append(emotion_label)
            count += 1

    X = np.array(X)
    y = np.array(y)
    
    print(f"Total samples: {len(X)}")
    print(f"Classes found: {np.unique(y)}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training SVM Model...")
    # Using RBF kernel because texture data is complex
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    # Save Model
    joblib.dump(svm, MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")

if __name__ == "__main__":
    train_svm_model()
