import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from EDA.config import Config as config
from EDA.features import extract_features


def load_dataset(benign_dir:[Path], malign_dir: [Path]):
    """
        This function is used to load datas.
        Parameters:
            benign_dir([]): list of paths to benign datas.
            malign_dir (str): list of paths to malign datas.
        Returns:
            Returns a Tuple (X, y) where X is a data ( image representation ) and y it's class
    """
    data_dir = config.data_dir() if callable(config.data_dir) else config.data_dir

    if not isinstance(data_dir, Path):
        raise TypeError("Config.data_dir doit être un objet pathlib.Path")
    print(f"data_dir: {data_dir}")
    data_dir = data_dir.absolute()
    print(f"data_dir: {data_dir}")
    if data_dir.is_dir():
        print("It's exist !!!")
    valid_dirs = []
    valid_dirs.extend(benign_dir)
    valid_dirs.extend(malign_dir)
    all_files = []
    for dir_name in valid_dirs:
        print(f"data_dir: {data_dir.absolute()}; dir_name: {dir_name}")
        dir_path: Path = data_dir / dir_name
        if dir_path.exists():
            all_files.extend(list(dir_path.glob('*.[jp][pn][g]')))

    features = []
    labels = []

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = []
        for file_path in all_files:
            futures.append(
                executor.submit(_process_single_file, file_path)
            )

        for future in tqdm(futures):
            try:
                feature, label = future.result()
                features.append(feature)
                labels.append(1 if label == 'Benign' else 0)
            except Exception as e:
                logging.error(f"Erreur lors du traitement d'une image: {e}")
    features = np.array(features)
    labels = np.array(labels)
    title = [f"X{_}" for _ in range(features.shape[1])]
    df = pd.DataFrame(features, columns=title)
    df.insert(0, "y", labels)
    df.to_csv(config.blood_cell_processed_data, index=False)
    return np.array(features), np.array(labels)


def _process_single_file(file_path: Path) -> Tuple[np.ndarray, str]:
    """Traite une image unique"""
    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError(f"Impossible de charger l'image: {file_path}")

    image = cv2.resize(image, config.image_size)
    features = extract_features(image)
    label = file_path.parent.name

    return features, label


if __name__ == '__main__':

    benign_dir = [config.benign_dir.absolute()]
    malign_dirs = [config.malignant_pre_b_dir.absolute(), config.malignant_early_pre_b_dir.absolute(), config.malignant_pro_b_dir.absolute()]

    X_load, y_load = load_dataset(benign_dir, malign_dirs)
    titles = [f"x{_}" for _ in range(X_load.shape[1])]
    print(f"X_load = {X_load.shape} y_load = {y_load.shape}")