from dataclasses import dataclass
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import warnings
warnings.filterwarnings('ignore')

@dataclass
class LeukemiaConfig:
    """Configuration pour la détection de leucémie"""
    data_dir: Path = Path("path/to/training/set")
    image_size: Tuple[int, int] = (128, 128)  # Taille optimale pour l'extraction de caractéristiques
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    max_workers: int = 8
    cv_folds: int = 5

class ImageFeatureExtractor:
    """Extracteur de caractéristiques d'images optimisé"""
    
    @staticmethod
    def extract_features(image: np.ndarray) -> np.ndarray:
        """Extrait les caractéristiques d'une image"""
        features = []
        
        # Caractéristiques de couleur
        for channel in cv2.split(image):
            features.extend([
                np.mean(channel),  # Moyenne
                np.std(channel),   # Écart-type
                np.percentile(channel, 25),  # Premier quartile
                np.percentile(channel, 75),  # Troisième quartile
                np.max(channel) - np.min(channel)  # Range
            ])
        
        # Caractéristiques de texture (GLCM)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calcul des caractéristiques de Haralick
        haralick = np.mean(cv2.resize(gray, (32, 32)), axis=0)
        features.extend(haralick)
        
        # Histogrammes de couleur
        for channel in cv2.split(image):
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # Détection de contours et statistiques
        edges = cv2.Canny(gray, 100, 200)
        features.extend([
            np.mean(edges),
            np.std(edges),
            np.sum(edges > 0) / edges.size  # Ratio de bords
        ])
        
        return np.array(features, dtype=np.float32)

class LeukemiaDetector:
    """Détecteur de leucémie basé sur Random Forest"""
    
    def __init__(self, config: LeukemiaConfig):
        self.config = config
        self.classifier = None
        self.feature_extractor = ImageFeatureExtractor()
        self.results_dir = Path('results') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        log_file = self.results_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_and_process_image(self, image_path: Path) -> np.ndarray:
        """Charge et prétraite une image"""
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, self.config.image_size)
        return image

    def _process_single_file(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Traite un fichier unique et retourne ses caractéristiques"""
        image = self._load_and_process_image(file_path)
        features = self.feature_extractor.extract_features(image)
        label = int(file_path.parent.name == 'positive')
        return features, label

    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Charge et traite le dataset complet en parallèle"""
        all_files = list(self.config.data_dir.rglob('*.[jp][pn][g]'))
        self.logger.info(f"Traitement de {len(all_files)} images...")
        
        features = []
        labels = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            results = list(executor.map(self._process_single_file, all_files))
        
        features, labels = zip(*results)
        return np.array(features), np.array(labels)

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Génère la matrice de confusion"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion')
        plt.ylabel('Vrai label')
        plt.xlabel('Prédiction')
        plt.savefig(self.results_dir / 'confusion_matrix.png')
        plt.close()

    def _plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """Génère la courbe ROC"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title('Courbe ROC')
        plt.legend(loc="lower right")
        plt.savefig(self.results_dir / 'roc_curve.png')
        plt.close()

    def train(self) -> Dict[str, Any]:
        """Entraîne le détecteur de leucémie"""
        self.logger.info("Chargement et traitement des données...")
        X, y = self._load_dataset()
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Entraînement du modèle
        self.logger.info("Entraînement du Random Forest...")
        self.classifier = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            class_weight='balanced',
            n_jobs=-1,
            random_state=self.config.random_state,
            verbose=1
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)
        
        # Métriques et visualisations
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_roc_curve(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.classifier, X_test, y_test,
            cv=self.config.cv_folds, scoring='roc_auc'
        )
        
        # Sauvegarde du modèle
        joblib.dump(self.classifier, self.results_dir / 'leukemia_detector.joblib')
        
        results = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std(),
        }
        
        self.logger.info(f"\nScore CV moyen: {results['cv_scores_mean']:.3f} "
                        f"(±{results['cv_scores_std']:.3f})")
        self.logger.info("\nRapport de classification:")
        self.logger.info(classification_report(y_test, y_pred))
        
        return results

    def predict(self, image_path: Path) -> Tuple[str, float]:
        """Prédit la classe d'une nouvelle image"""
        if self.classifier is None:
            raise RuntimeError("Le modèle n'est pas entraîné")
        
        image = self._load_and_process_image(image_path)
        features = self.feature_extractor.extract_features(image)
        
        prediction = self.classifier.predict([features])[0]
        probability = self.classifier.predict_proba([features]).max()
        
        result = "Leucémie détectée" if prediction == 1 else "Pas de leucémie détectée"
        return result, probability

def main():
    config = LeukemiaConfig()
    detector = LeukemiaDetector(config)
    results = detector.train()
    
    # Exemple de prédiction
    test_image_path = Path("path/to/test/image.jpg")
    if test_image_path.exists():
        result, confidence = detector.predict(test_image_path)
        print(f"\nRésultat: {result}")
        print(f"Confiance: {confidence:.2%}")

if __name__ == "__main__":
    main()
