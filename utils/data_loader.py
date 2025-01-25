import numpy as np
import cv2
from pathlib import Path
from typing import List

from config import Config

def load_dataset(benign_dirs: List[Path], malign_dirs: List[Path]):
    """
    Load images from benign and malignant directories
    
    Args:
        benign_dirs (List[Path]): Paths to benign image directories
        malign_dirs (List[Path]): Paths to malignant image directories
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels
    """
    X, y = [], []
    
    # Load benign images (labeled as 0)
    for benign_dir in benign_dirs:
        for img_path in Path(benign_dir).glob('*.jpg'):
            image = cv2.imread(str(img_path))
            if image is not None:
                image = cv2.resize(image, Config.image_size)
                features = extract_features(image)
                X.append(features)
                y.append(0)
    
    # Load malignant images (labeled as 1)
    for malign_dir in malign_dirs:
        for img_path in Path(malign_dir).glob('*.jpg'):
            image = cv2.imread(str(img_path))
            if image is not None:
                image = cv2.resize(image, Config.image_size)
                features = extract_features(image)
                X.append(features)
                y.append(1)
    
    return np.array(X), np.array(y)

def extract_features(image: np.ndarray) -> np.ndarray:
    """Import extract_features from features.py"""
    from features import extract_features as _extract_features
    return _extract_features(image)
