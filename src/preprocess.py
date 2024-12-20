import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image as PILImage


class Image:
    """
    Image class
    """

    def __init__(
            self,
            img_path: str,
            resnet: bool = False, 
            normalize: bool = True
    ):
        self.path = img_path
        img = cv2.imread(img_path)
        if not resnet:
            self.feature_vec = extract_color_feature(img, normalize)
        else:
            self.feature_vec = extract_resnet_feature(img)
        

def extract_color_feature(
        img: np.ndarray, 
        normalize: bool = True
) -> np.ndarray:
    """
    Generate color feature vector for the given image
    """
    H, W, C = img.shape 
    half_H, half_W = H // 2, W // 2
    
    # Calculate the sum of RGB values in each quadrant
    rgb = []
    for i in range(2):
        for j in range(2):
            quadrant = img[i * half_H: (i + 1) * half_H, j * half_W: (j + 1) * half_W]
            quadrant_sum = np.sum(quadrant, axis=(0, 1))
            if normalize:
                rgb.extend(quadrant_sum / np.sum(quadrant_sum))
            else:
                rgb.extend(quadrant_sum)
    rgb = np.array(rgb)

    # Generate feature vector
    lb = min(rgb) + (max(rgb) - min(rgb)) / 3
    ub = max(rgb) - (max(rgb) - min(rgb)) / 3
    feature_vec = np.ones(12, dtype=np.uint8)
    feature_vec[rgb >= ub] = 2
    feature_vec[rgb <= lb] = 0
    return feature_vec


def extract_resnet_feature(
        img: np.ndarray
) -> np.ndarray:
    """
    Generate feature vector for the given image using ResNet
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    img = preprocess_resnet(img)
    with torch.no_grad():
        features = model(img)
    features = features.squeeze()  # Remove the batch dimension
    feature_vector = features[:12]  # Extract the first 12 features
    return feature_vector.numpy()
    

def preprocess_resnet(
        img: np.ndarray
):
    """
    Preprocess the image for ResNet
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # Resize to 224x224
        transforms.ToTensor(),           # Convert to tensor
        transforms.Normalize(            # Normalize
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PILImage.fromarray(img)  # Convert to PIL image
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img