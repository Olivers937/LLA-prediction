from dataclasses import dataclass
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class LeukemiaConfig:
    """Configuration pour la détection de leucémie par ensemble"""
    data_dir: Path = Path("path/to/training/set")
    image_size: Tuple[int, int] = (128, 128)
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5
    max_workers: int = 8
    
    # Grilles de paramètres pour chaque modèle
    rf_params: Dict = None
    etc_params: Dict = None
    svm_params: Dict = None
    lr_params: Dict = None
    
    def __post_init__(self):
        """Initialisation des grilles de paramètres par défaut"""
        self.rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 20, 40],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced']
        }
        
        self.etc_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 20, 40],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced']
        }
        
        self.svm_params = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'poly'],
            'class_weight': ['balanced']
        }
        
        self.lr_params = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'multi_class': ['multinomial'],
            'class_weight': ['balanced']
        }

class ImageFeatureExtractor:
    """Extracteur optimisé de caractéristiques d'images"""
    
    def __init__(self, config: LeukemiaConfig):
        self.config = config
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extrait les caractéristiques d'une image"""
        features = []
        
        # 1. Caractéristiques de couleur
        for channel in cv2.split(image):
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
                np.max(channel) - np.min(channel),
                self._calculate_entropy(channel)
            ])
        
        # 2. Caractéristiques de texture
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.extend(self._extract_texture_features(gray))
        
        # 3. Caractéristiques de forme
        features.extend(self._extract_shape_features(gray))
        
        # 4. Histogrammes de couleur
        features.extend(self._extract_color_histograms(image))
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_entropy(self, channel: np.ndarray) -> float:
        """Calcule l'entropie d'un canal"""
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist / float(hist.sum())
        return -np.sum(hist * np.log2(hist + 1e-7))
    
    def _extract_texture_features(self, gray: np.ndarray) -> List[float]:
        """Extrait les caractéristiques de texture"""
        features = []
        
        # GLCM (Gray Level Co-occurrence Matrix)
        glcm = cv2.resize(gray, (32, 32))
        features.extend([
            np.mean(glcm),
            np.std(glcm),
            np.var(glcm)
        ])
        
        # LBP (Local Binary Patterns) simplifié
        kernel = np.array([[1, 2, 4],
                          [8, 0, 16],
                          [32, 64, 128]], dtype=np.uint8)
        lbp = cv2.filter2D(gray, -1, kernel)
        features.extend([np.mean(lbp), np.std(lbp)])
        
        return features
    
    def _extract_shape_features(self, gray: np.ndarray) -> List[float]:
        """Extrait les caractéristiques de forme"""
        features = []
        
        # Détection de contours
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Caractéristiques basées sur les contours
        if contours:
            areas = [cv2.contourArea(cnt) for cnt in contours]
            perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
            features.extend([
                np.mean(areas),
                np.std(areas),
                np.mean(perimeters),
                np.std(perimeters),
                len(contours)  # Nombre de contours
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return features
    
    def _extract_color_histograms(self, image: np.ndarray, bins: int = 32) -> List[float]:
        """Extrait les histogrammes de couleur"""
        features = []
        
        for channel in cv2.split(image):
            hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalisation
            features.extend(hist)
        
        return features

class BaseModelWrapper:
    """Wrapper pour les modèles individuels avec optimisation"""
    
    def __init__(self, model, param_grid: Dict, name: str, config: LeukemiaConfig):
        self.model = model
        self.param_grid = param_grid
        self.name = name
        self.config = config
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entraîne le modèle avec recherche de paramètres"""
        logging.info(f"Entraînement de {self.name}...")
        
        # Prétraitement
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Recherche des meilleurs paramètres
        grid_search = GridSearchCV(
            self.model,
            self.param_grid,
            cv=self.config.cv_folds,
            n_jobs=self.config.n_jobs,
            scoring='accuracy',
            verbose=1
        )
        
        grid_search.fit(X_scaled, y_encoded)
        self.best_model = grid_search.best_estimator_
        
        logging.info(f"{self.name} - Meilleurs paramètres: {grid_search.best_params_}")
        logging.info(f"{self.name} - Score CV: {grid_search.best_score_:.3f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les classes"""
        if self.best_model is None:
            raise ValueError(f"{self.name} n'est pas encore entraîné")
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.best_model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prédit les probabilités"""
        if self.best_model is None:
            raise ValueError(f"{self.name} n'est pas encore entraîné")
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict_proba(X_scaled)

class EnsembleLeukemiaDetector:
    """Détecteur de leucémie basé sur un ensemble de modèles"""
    
    def __init__(self, config: LeukemiaConfig):
        self.config = config
        self.feature_extractor = ImageFeatureExtractor(config)
        self.models = []
        self.label_encoder = LabelEncoder()
        self.results_dir = Path('results') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir.mkdir(parents=True, exist_ok=True)
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
        """Initialise tous les modèles de l'ensemble"""
        self.models = [
            BaseModelWrapper(
                RandomForestClassifier(random_state=self.config.random_state),
                self.config.rf_params,
                "Random Forest",
                self.config
            ),
            BaseModelWrapper(
                ExtraTreesClassifier(random_state=self.config.random_state),
                self.config.etc_params,
                "Extra Trees",
                self.config
            ),
            BaseModelWrapper(
                SVC(probability=True, random_state=self.config.random_state),
                self.config.svm_params,
                "SVM",
                self.config
            ),
            BaseModelWrapper(
                LogisticRegression(random_state=self.config.random_state),
                self.config.lr_params,
                "Logistic Regression",
                self.config
            )
        ]
    
    def _load_and_process_image(self, image_path: Path) -> np.ndarray:
        """Charge et prétraite une image"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        image = cv2.resize(image, self.config.image_size)
        return image
    
    def _process_single_file(self, file_path: Path) -> Tuple[np.ndarray, str]:
        """Traite un fichier unique et retourne ses caractéristiques"""
        try:
            image = self._load_and_process_image(file_path)
            features = self.feature_extractor.extract_features(image)
            label = file_path.parent.name
            return features, label
        except Exception as e:
            logging.error(f"Erreur lors du traitement de {file_path}: {e}")
            raise
    
    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Charge et traite le dataset complet en parallèle"""
        all_files = list(self.config.data_dir.rglob('*.[jp][pn][g]'))
        logging.info(f"Chargement de {len(all_files)} images...")
        
        features = []
        labels = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Utilisation de tqdm pour afficher une barre de progression
            for result in tqdm(executor.map(self._process_single_file, all_files),
                             total=len(all_files)):
                feature, label = result
                features.append(feature)
                labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def train(self) -> Dict[str, Any]:
        """Entraîne l'ensemble de modèles"""
        logging.info("Chargement et traitement des données...")
        X, y = self._load_dataset()
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Entraînement parallèle des modèles
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = [executor.submit(model.fit, X_train, y_train) 
                      for model in self.models]
            
            for future in futures:
                future.result()  # Attend la fin de l'entraînement
        
        # Évaluation
        results = self._evaluate_ensemble(X_test, y_test)
        
        return results
    
    def _evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Évalue les performances de l'ensemble"""
        individual_predictions = {}
        individual_probas = {}
        
        # Prédictions individuelles
        for model in self.models:
            individual_predictions[model.name] =