import numpy as np
from typing import Union
from preprocess import Image

class LSH:
    """
    Locality Sensitive Hashing
    """

    def __init__(
            self, 
            indicators: Union[list, np.ndarray],
            resnet: bool = False,
            normalize: bool = True
    ):
        self.hash_tab = {}
        self.indicators = np.array(indicators) if isinstance(indicators, list) else indicators
        self.resnet = resnet
        self.normalize = normalize

    def add(
            self, 
            img_path: str
    ):
        """
        Add a dataset image to the hash table
        """
        img = Image(img_path, self.resnet, self.normalize)
        hash_val = self._projection(img.feature_vec, self.indicators)
        if hash_val not in self.hash_tab:
            self.hash_tab[hash_val] = [img_path]
        else:
            self.hash_tab[hash_val].append(img_path)

    def search(
            self, 
            img_path: str
    ):
        """
        Search for the most similar image in the dataset
        """
        image = Image(img_path, self.resnet, self.normalize)
        hash_val = self._projection(image.feature_vec, self.indicators)
        if hash_val not in self.hash_tab:
            return None
        
        candidates = self.hash_tab[hash_val]
        min_dist = float('inf')
        min_img = None

        for img_path in candidates:
            img, dist = self._calc_dist(img_path, image)
            if dist < min_dist:
                min_dist = dist
                min_img = img

        return min_img.path
    
    def _projection(
            self, 
            feature_vec: np.ndarray,
            indicators: np.ndarray
    ) -> np.ndarray:
        """
        Project the feature vector to the given projection set
        """
        proj_1 = (indicators - 1) // 2
        proj_2 = (indicators - 1) % 2 + 1
        hash_res = (proj_2 <= feature_vec[proj_1]).astype(np.uint8)
        return tuple(hash_res)
    
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
        

if __name__ == '__main__':
    pass