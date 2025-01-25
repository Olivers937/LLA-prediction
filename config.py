from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict


@dataclass
class Config:
    """Configuration pour la détection de la LLA"""
    current_file = Path(__file__)
    dataset_dir: Path = current_file.parent.parent / "dataset"
    data_dir: Path = dataset_dir /"Blood cell Cancer [ALL]"
    benign_dir: Path = data_dir / "Benign"
    malignant_early_pre_b_dir = data_dir / "[Malignant] early Pre-B"
    malignant_pre_b_dir = data_dir / "[Malignant] Pre-B"
    malignant_pro_b_dir = data_dir / "[Malignant] Pro-B"
    blood_cell_processed_data = dataset_dir / "processed_data" / "blood_cell_cancer.csv"
    c_nmc_processed_data = dataset_dir/ "processed_data" / "c_nmc_leukemia.csv"
    image_size: Tuple[int, int] = (128, 128)
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5
    max_workers: int = 8

    # Paramètres pour les différents modèles
    rf_params = {
        'n_estimators': [200, 300, 400],
        'max_depth': [None, 30, 50],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', 'second_class']
    }

    etc_params = {
        'n_estimators': [200, 300, 400],
        'max_depth': [None, 30, 50],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', 'second_class']
    }

    svm_params = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1],
        'kernel': ['rbf'],
        'class_weight': ['balanced', 'second_class']
    }

    lr_params = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'class_weight': ['balanced', {0: 1, 1: 10}],
        'max_iter': [1000]
    }
