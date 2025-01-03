import numpy as np
from src.preprocess import Image


class KNN:
    """
    K-Nearest Neighbors (K = 1)
    """

    def __init__(
            self, 
            resnet: bool = False, 
            normalize: bool = True
    ):
        self.image_paths = []
        self.resnet = resnet
        self.normalize = normalize

    def add(
            self, 
            img_path: str
    ):
        """
        Add a dataset image
        """
        self.image_paths.append(img_path)

    def search(
            self, 
            img_path: str
    ):
        """
        Search for the most similar image in the dataset
        """
        image = Image(img_path, self.resnet, self.normalize)
        
        min_dist = float('inf')
        min_img = None

        for img_path in self.image_paths:
            img, dist = self._calc_dist(img_path, image)
            if dist < min_dist:
                min_dist = dist
                min_img = img

        return min_img.path
    
    def _calc_dist(
            self,
            img_path: str,
            target_img: Image
    ) -> float:
        """
        Calculate the distance between the target image and the image in the dataset
        """
        img = Image(img_path, self.resnet, self.normalize)
        dist = np.linalg.norm(img.feature_vec - target_img.feature_vec)
        return img, dist