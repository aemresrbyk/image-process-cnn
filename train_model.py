import cv2
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import local_binary_pattern

# Import DIP preprocessing from utils
from utils import apply_dip_preprocessing

# Configuration
DATASET_PATH = "dataset/train"
MODEL_FILENAME = "emotion_model.pkl"
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
SAMPLES_PER_CLASS = 100000  # Use all data

def extract_lbp_features(image):
    """
    Extracts LBP features with radius=3, n_points=24.
    """
    radius = 3
    n_points = 24  # 8 * radius, keeping consistent with request
    method = 'uniform'
    
    # Check if image is color, if so convert to gray (though preprocessing should handle it)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    lbp = local_binary_pattern(gray, n_points, radius, method)
    
    # Calculate histogram
    # For 'uniform' method, number of bins = n_points + 2
    n_bins = int(n_points + 2)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

def train_svm_model():
    print("Loading dataset and extracting features...")
    
    X = []
    y = []
    
    # Traverse through emotion folders
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' does not exist.")
        return

    for emotion_label in os.listdir(DATASET_PATH):
        emotion_dir = os.path.join(DATASET_PATH, emotion_label)
        
        if not os.path.isdir(emotion_dir):
            continue
            
        print(f"Processing Class: {emotion_label}")
        
        # 1. Load all images for this class first
        class_images = []
        for image_name in os.listdir(emotion_dir):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(emotion_dir, image_name)
            img = cv2.imread(image_path)
            if img is not None:
                class_images.append(img)
        
        # Limit to SAMPLES_PER_CLASS if we have too many
        if len(class_images) > SAMPLES_PER_CLASS:
            class_images = class_images[:SAMPLES_PER_CLASS]
            
        # 2. Data Augmentation for Minority Classes
        # If we have fewer than MIN_SAMPLES, synthetically generate more!
        MIN_SAMPLES = 3000
        current_count = len(class_images)
        
        if current_count < MIN_SAMPLES and current_count > 0:
            print(f"  -> Augmenting {emotion_label} from {current_count} to {MIN_SAMPLES} samples...")
            diff = MIN_SAMPLES - current_count
            
            # Simple augmentation loop
            for i in range(diff):
                # Pick a random original image
                orig_img = class_images[i % current_count]
                
                # Apply Random Augmentations
                aug_img = orig_img.copy()
                
                # Random Flip
                if np.random.rand() > 0.5:
                    aug_img = cv2.flip(aug_img, 1) # Horizontal flip
                
                # Random Noise (Gaussian)
                if np.random.rand() > 0.5:
                    gauss = np.random.normal(0, 10, aug_img.shape).astype('uint8')
                    # Add noise with clipping
                    aug_img = cv2.add(aug_img, gauss)
                
                # Random Brightness
                if np.random.rand() > 0.5:
                    value = np.random.randint(-30, 30)
                    if value > 0:
                        aug_img = np.minimum(aug_img + value, 255).astype('uint8')
                    else:
                        aug_img = np.maximum(aug_img + value, 0).astype('uint8')
                        
                class_images.append(aug_img)

        # 3. Process All Images (Original + Augmented)
        count = 0
        for img in class_images:
            # Apply DIP Preprocessing
            processed_img = apply_dip_preprocessing(img)
            
            # Extract Features (LBP)
            lbp_hist = extract_lbp_features(processed_img)
            
            X.append(lbp_hist)
            y.append(emotion_label)
            count += 1
        
        print(f"  -> Features extracted: {count}")

    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("No images found. Check dataset path.")
        return

    print(f"Total samples: {len(X)}")
    print(f"Classes found: {np.unique(y)}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training LinearSVC Model...")
    # Using LinearSVC as requested
    svm = LinearSVC(random_state=42, max_iter=2000, class_weight='balanced') # Increased max_iter for convergence, added balanced weights
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
