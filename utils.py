import cv2
import numpy as np

def apply_dip_preprocessing(image):
    """
    Applies Digital Image Processing (DIP) techniques to preprocess the image for the model.
    Steps:
    1. Convert to Grayscale (if needed).
    2. Apply Histogram Equalization (to improve contrast).
    3. Apply Gaussian Blur (kernel size 3x3, to remove noise).
    
    Args:
        image: Input image (numpy array).
        
    Returns:
        Processed image.
    """
    # 1. Convert to Grayscale if the image has 3 channels
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Apply Histogram Equalization
    # Improves contrast by stretching the intensity range
    equ = cv2.equalizeHist(gray)

    # 3. Apply Gaussian Blur
    # Removes high-frequency noise which can confuse the model
    blurred = cv2.GaussianBlur(equ, (3, 3), 0)

    return blurred

def apply_post_processing_filters(frame, face_coords, emotion):
    """
    Applies visual effects (color filters) to the frame based on the detected emotion.
    
    Args:
        frame: The original video frame.
        face_coords: Tuple (x, y, w, h) of the detected face.
        emotion: The detected emotion string ('happy', 'sad', 'angry', 'neutral').
        
    Returns:
        The frame with the applied filter.
    """
    x, y, w, h = face_coords
    
    # Define colors for emotions (BGR format)
    colors = {
        'happy': (0, 255, 255),    # Yellow
        'sad': (255, 0, 0),        # Blue
        'angry': (0, 0, 255),      # Red
        'neutral': (128, 128, 128) # Gray
    }
    
    color = colors.get(emotion, (255, 255, 255)) # Default white
    
    # Create an overlay
    overlay = frame.copy()
    
    # Draw a colored rectangle over the face area
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    
    # Blend the overlay with the original frame (transparency)
    alpha = 0.3  # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame