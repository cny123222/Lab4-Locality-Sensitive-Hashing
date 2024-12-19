import cv2
import numpy as np

class Image:
    """
    Image class
    """

    def __init__(
            self,
            img_path: str
    ):
        self.path = img_path
        img = cv2.imread(img_path)
        self.feature_vec = generate_feature_vector(img)
        

def generate_feature_vector(
        img: np.ndarray
) -> np.ndarray:
    """
    Generate feature vector for the given image
    """
    H, W, C = img.shape
    half_H, half_W = H // 2, W // 2
    
    # Calculate the sum of RGB values in each quadrant
    rgb = []
    rgb.extend(np.sum(img[:half_H, :half_W], axis=(0, 1)))  # Top-left
    rgb.extend(np.sum(img[:half_H, half_W:], axis=(0, 1)))  # Top-right
    rgb.extend(np.sum(img[half_H:, :half_W], axis=(0, 1)))  # Bottom-left
    rgb.extend(np.sum(img[half_H:, half_W:], axis=(0, 1)))  # Bottom-right

    # Generate feature vector
    lb = min(rgb) + (max(rgb) - min(rgb)) / 3
    ub = max(rgb) - (max(rgb) - min(rgb)) / 3
    feature_vec = np.ones(12, dtype=np.uint8)
    feature_vec[rgb >= ub] = 2
    feature_vec[rgb <= lb] = 0
    return feature_vec