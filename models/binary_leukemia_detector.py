import time
from dataclasses import dataclass
import numpy as np
import cv2
from pathlib import Path
import logging
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional

from pyct.cmd import DATA_DIR
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ALLConfig:
    """Configuration pour la détection de la LLA"""
    data_dir: Path = Path("/home/willy-watcho/PycharmProjects/LLA-prediction/dataset/C-NMC_Leukemia/training_data/fold_0/hem")
    image_size: Tuple[int, int] = (128, 128)
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5
    max_workers: int = 8

    
    # Paramètres pour les différents modèles
    rf_params: Dict = None
    etc_params: Dict = None
    svm_params: Dict = None
    lr_params: Dict = None
    
    def __post_init__(self):
        """Initialisation des grilles de paramètres optimisées pour LLA"""
        self.rf_params = {
            'n_estimators': [200, 300, 400],
            'max_depth': [None, 30, 50],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', 'second_class']
        }
        
        self.etc_params = {
            'n_estimators': [200, 300, 400],
            'max_depth': [None, 30, 50],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', 'second_class']
        }
        
        self.svm_params = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1],
            'kernel': ['rbf'],
            'class_weight': ['balanced', 'second_class']
        }
        
        self.lr_params = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'class_weight': ['balanced', {0:1, 1:10}],
            'max_iter': [1000]
        }

class ImageFeatureExtractor:
    """Extracteur de caractéristiques optimisé pour la détection de LLA"""
    
    def __init__(self, config: ALLConfig):
        self.config = config
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extrait les caractéristiques pertinentes pour la LLA"""
        features = []
        
        # 1. Caractéristiques morphologiques (spécifiques à la LLA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.extend(self._extract_morphological_features(gray))
        
        # 2. Caractéristiques de couleur (importantes pour la LLA)
        features.extend(self._extract_color_features(image))
        
        # 3. Caractéristiques de texture
        features.extend(self._extract_texture_features(gray))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_morphological_features(self, gray: np.ndarray) -> List[float]:
        """Extrait les caractéristiques morphologiques spécifiques à la LLA"""
        features = []
        
        # Segmentation des cellules
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Caractéristiques de forme des cellules
            areas = [cv2.contourArea(cnt) for cnt in contours]
            perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
            circularities = [(4 * np.pi * area) / (peri ** 2) if peri > 0 else 0 
                           for area, peri in zip(areas, perimeters)]
            
            features.extend([
                np.mean(areas),  # Taille moyenne des cellules
                np.std(areas),   # Variation de taille
                np.mean(circularities),  # Circularité moyenne
                np.std(circularities),   # Variation de forme
                len(contours)    # Nombre de cellules
            ])
        else:
            features.extend([0] * 5)
        
        return features
    
    def _extract_color_features(self, image: np.ndarray) -> List[float]:
        """Extrait les caractéristiques de couleur importantes pour la LLA"""
        features = []
        
        # Conversion en espace LAB (plus proche de la perception humaine)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        for channel in cv2.split(lab):
            features.extend([
                np.mean(channel),  # Moyenne
                np.std(channel),   # Écart-type
                np.percentile(channel, 25),  # Q1
                np.percentile(channel, 75),  # Q3
                skew(channel.ravel()),  # Asymétrie
                kurtosis(channel.ravel())  # Kurtosis
            ])
        
        return features
    
    def _extract_texture_features(self, gray: np.ndarray) -> List[float]:
        """Extrait les caractéristiques de texture pour la LLA"""
        features = []
        
        # GLCM amélioré
        glcm = graycomatrix(gray, [1], [0, 45, 90, 135], 
                           levels=256, symmetric=True, normed=True)
        
        # Caractéristiques de Haralick
        contrast = graycoprops(glcm, 'contrast').ravel()
        dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
        homogeneity = graycoprops(glcm, 'homogeneity').ravel()
        correlation = graycoprops(glcm, 'correlation').ravel()
        
        features.extend([
            np.mean(contrast),
            np.mean(dissimilarity),
            np.mean(homogeneity),
            np.mean(correlation)
        ])
        
        return features

class BaseModelWrapper:
    """Wrapper pour les modèles de classification binaire"""
    
    def __init__(self, model, param_grid: Dict, name: str, config: ALLConfig):
        self.model = model
        self.param_grid = param_grid
        self.name = name
        self.config = config
        self.best_model = None
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entraîne le modèle avec optimisation des hyperparamètres"""
        logging.info(f"Entraînement de {self.name}...")
        
        # Prétraitement
        X_scaled = self.scaler.fit_transform(X)
        
        # Recherche des meilleurs paramètres
        grid_search = GridSearchCV(
            self.model,
            self.param_grid,
            cv=self.config.cv_folds,
            n_jobs=self.config.n_jobs,
            scoring='roc_auc',  # Utilisation de ROC AUC pour l'optimisation
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        self.best_model = grid_search.best_estimator_
        
        logging.info(f"{self.name} - Meilleurs paramètres: {grid_search.best_params_}")
        logging.info(f"{self.name} - Score ROC AUC CV: {grid_search.best_score_:.3f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les classes (0: Normal, 1: LLA)"""
        if self.best_model is None:
           raise ValueError(f"{self.name} n'est pas encore entraîné")

        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prédit les probabilités"""
        if self.best_model is None:
            raise ValueError(f"{self.name} n'est pas encore entraîné")

        X_scaled = self.scaler.transform(X)
        return self.best_model.predict_proba(X_scaled)

    def to_string(self):
        return "BaseModelWrapper : {" + f"model = {self.model}; name = {self.name}; config = {self.config}; best_model = {self.best_model}; scaler = {self.scaler}" + "}"

class EnsembleALLDetector:
    """Détecteur de LLA basé sur un ensemble de modèles"""
    
    def __init__(self, config: ALLConfig):
        self.config = config
        self.feature_extractor = ImageFeatureExtractor(config)
        self.models = []
        self.results_dir = Path('../test/results') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.benign_dir = ['/home/willy-watcho/PycharmProjects/LLA-prediction/LLA-prediction/dataset/Blood cell Cancer [ALL]/Benign']
        self.malign_dir = ['/home/willy-watcho/PycharmProjects/LLA-prediction/LLA-prediction/dataset/Blood cell Cancer [ALL]/[Malignant] early Pre-B',
            '/home/willy-watcho/PycharmProjects/LLA-prediction/LLA-prediction/dataset/Blood cell Cancer [ALL]/[Malignant] Pre-B',
            '/home/willy-watcho/PycharmProjects/LLA-prediction/LLA-prediction/dataset/Blood cell Cancer [ALL]/[Malignant] Pro-B']
        self.benign_training_dir = ['/home/willy-watcho/PycharmProjects/LLA-prediction/LLA-prediction/dataset/Blood cell Cancer [ALL]/testing/benign']
        self.malign_training_dir = ['/home/willy-watcho/PycharmProjects/LLA-prediction/LLA-prediction/dataset/Blood cell Cancer [ALL]/testing/malign']
        self._setup_logging()
        self._initialize_models()
        
    def _setup_logging(self) -> None:
        """Configure le système de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'ensemble.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_models(self) -> None:
        """Initialise les modèles de l'ensemble"""
        self.models = [
            BaseModelWrapper(
                RandomForestClassifier(random_state=self.config.random_state),
                self.config.rf_params,
                "Random Forest",
                self.config
            ),
            BaseModelWrapper(
                SVC(probability=True, random_state=self.config.random_state),
                self.config.svm_params,
                "SVM",
                self.config
            ),
            BaseModelWrapper(
                ExtraTreesClassifier(random_state=self.config.random_state),
                self.config.etc_params,
                "Extra Trees",
                self.config
            ),
            BaseModelWrapper(
                LogisticRegression(random_state=self.config.random_state),
                self.config.lr_params,
                "Logistic Regression",
                self.config
            )
        ]
    
    def _load_dataset(self, benign_dir: [], malign_dir: []) -> Tuple[np.ndarray, np.ndarray]:
        # Ne prend que les dossiers 'normal' et 'all'
        #valid_dirs = [normal_dir, all_dir]
        #valid_dirs = [
        #    '/home/willy-watcho/PycharmProjects/LLA-prediction/LLA-prediction/dataset/Blood cell Cancer [ALL]/Benign',
        #    '/home/willy-watcho/PycharmProjects/LLA-prediction/LLA-prediction/dataset/Blood cell Cancer [ALL]/[Malignant] early Pre-B',
        #    '/home/willy-watcho/PycharmProjects/LLA-prediction/LLA-prediction/dataset/Blood cell Cancer [ALL]/[Malignant] Pre-B',
        #    '/home/willy-watcho/PycharmProjects/LLA-prediction/LLA-prediction/dataset/Blood cell Cancer [ALL]/[Malignant] Pro-B',
        #]
        valid_dirs = []
        valid_dirs.extend(benign_dir)
        valid_dirs.extend(malign_dir)
        all_files = []
        for dir_name in valid_dirs:
            print(f"dir_name = {dir_name}")
            dir_path = self.config.data_dir / dir_name
            print(f"dir_path = {dir_path}")
            if dir_path.exists():
                all_files.extend(list(dir_path.glob('*.[jp][pn][g]')))
        
        features = []
        labels = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for file_path in all_files:
                futures.append(
                    executor.submit(self._process_single_file, file_path)
                )

            for future in tqdm(futures):
                try:
                    feature, label = future.result()
                    features.append(feature)
                    labels.append(1 if label == 'Benign' else 0)
                except Exception as e:
                    logging.error(f"Erreur lors du traitement d'une image: {e}")
        
        return np.array(features), np.array(labels)
    
    def _process_single_file(self, file_path: Path) -> Tuple[np.ndarray, str]:
        """Traite une image unique"""
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {file_path}")

        image = cv2.resize(image, self.config.image_size)
        features = self.feature_extractor.extract_features(image)
        label = file_path.parent.name

        return features, label
    
    def train(self) -> Dict[str, Any]:
        """Entraîne l'ensemble et évalue ses performances"""
        logging.info("Chargement et traitement des données...")
        X, y = self._load_dataset(self.benign_dir, self.malign_dir)
        #print(f"\nprint loaded dataset\nX = {X.shape} y = {y.shape}")
        np.savetxt("../test/test.txt", y)
        
        # Division des données
        tick = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        print(f"After train_test_split X_test = {len(X_test)}")

        # Entraînement parallèle des modèles

        #with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
        #   futures = [executor.submit(model.fit, X_train, y_train)
        #            for model in self.models]
        #    for future in futures:
        #        tick = time.time()
        #        future.result()
        #        tock = time.time()
        #        print(f"time = {tock - tick}")

        for model in self.models:
            logging.info(f"start training {model.name}")
            tick = time.time()
            model.fit(X_train, y_train)
            tock = time.time()
            logging.info(f"end training {model.name}")
            print(f"time = {tock - tick} for model {model.name}")
            np.savetxt("../test/test.txt", y_train)

        #X_test, y_test = self._load_dataset(self.benign_training_dir, self.malign_training_dir)

        print(f"X_test= {X_test.shape} y_test = {y_test.shape}")

        result = self._evaluate_ensemble(X_test, y_test)
        print(f"result = {result}")
        #self._save_model()
        
        return result
    
    def _evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Évalue les performances de l'ensemble"""
        # Prédictions individuelles
        individual_preds = {model.name: model.predict(X_test) for model in self.models}
        individual_probas = {model.name: model.predict_proba(X_test)[:, 1]
                          for model in self.models}
        result = 0
        for model in self.models:
            self.plt_test_result(X_test, y_test, individual_preds[model.name], model.name)
            print(model.to_string())

        for t in range(len(y_test)):
            if individual_preds[self.models[0].name][t] == individual_preds[self.models[1].name][t] == individual_preds[self.models[2].name][t] == individual_preds[self.models[3].name][t] :
                result += 1
        data =  {
            "y_test": y_test,
            self.models[0].name: individual_preds[self.models[0].name],
            self.models[1].name: individual_preds[self.models[1].name],
            self.models[2].name: individual_preds[self.models[2].name],
            self.models[3].name: individual_preds[self.models[3].name],
            self.models[0].name+"_proba": individual_probas[self.models[0].name],
            self.models[1].name+"_proba": individual_probas[self.models[1].name],
            self.models[2].name+"_proba": individual_probas[self.models[2].name],
            self.models[3].name+"_proba": individual_probas[self.models[3].name],
        }
        df = pd.DataFrame(data)
        df.to_csv("results.csv")

        print(f"result = {result}")
        print(f"y_test = {len(y_test)}")


    def plt_test_result(self, X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, name: str):

        plt.figure(figsize=(10, 6))

        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', cmap='viridis', label='Réel', alpha=0.7)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', cmap='coolwarm', label='Prédit', alpha=0.7)

        plt.title(f"Comparaison entre étiquettes réelles et prédites : {name}", fontsize=14)
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.legend()
        plt.grid(True)

        plt.show()


if __name__ == '__main__':
    config = ALLConfig()
    detector = EnsembleALLDetector(config)
    results = detector.train()
        