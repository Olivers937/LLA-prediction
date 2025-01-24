from dataclasses import dataclass
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import time
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import joblib
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

@dataclass
class LeukemiaConfig:
    """Configuration pour la détection de leucémie"""
    data_dir: Path = Path("path/to/training/set")
    batch_size: int = 32  # Réduit pour gérer des images potentiellement plus grandes
    num_workers: int = 8
    image_size: int = 299  # Augmenté pour plus de détails
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200  # Augmenté pour plus de robustesse
    model_name: str = "efficientnet_b4"  # Modèle plus performant pour les détails fins
    use_gpu: bool = True
    cv_folds: int = 5

class LeukemiaDetector:
    """Détecteur de leucémie basé sur l'analyse d'images de cellules sanguines"""
    
    def __init__(self, config: LeukemiaConfig):
        self.config = config
        self._setup_logging()
        self.device = self._setup_device()
        self.feature_extractor = None
        self.classifier = None
        self._setup_transforms()
        self.class_names = None
        self.results_dir = Path('results') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir.mkdir(parents=True, exist_ok=True)

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

    def _setup_transforms(self) -> None:
        """Prépare les transformations d'images adaptées aux cellules sanguines"""
        self.transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentations spécifiques pour les cellules sanguines
        self.transform_aug = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.RandomRotation(180),  # Rotation complète car l'orientation n'est pas importante
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _create_feature_extractor(self) -> None:
        """Crée l'extracteur de caractéristiques optimisé pour les cellules sanguines"""
        model = getattr(torchvision.models, self.config.model_name)(weights='IMAGENET1K_V1')
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Trace et sauvegarde la matrice de confusion"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion')
        plt.ylabel('Vrai label')
        plt.xlabel('Prédiction')
        plt.savefig(self.results_dir / 'confusion_matrix.png')
        plt.close()

    def _plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """Trace et sauvegarde la courbe ROC"""
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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Évalue le modèle avec des métriques appropriées pour le diagnostic médical"""
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)
        
        # Calcul des métriques
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Tracer les visualisations
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_roc_curve(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.classifier, X_test, y_test, 
            cv=self.config.cv_folds, scoring='roc_auc'
        )
        
        return {
            'classification_report': report,
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    def train(self) -> Dict:
        """Entraîne le détecteur de leucémie"""
        self.logger.info("Début de l'entraînement du détecteur de leucémie")
        
        if self.feature_extractor is None:
            self._create_feature_extractor()

        # Chargement et préparation des données
        dataset = torchvision.datasets.ImageFolder(
            root=self.config.data_dir,
            transform=self.transform
        )
        self.class_names = dataset.classes
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        # Extraction des caractéristiques
        features, labels = [], []
        with torch.no_grad():
            for images, batch_labels in dataloader:
                images = images.to(self.device)
                batch_features = self.feature_extractor(images)
                features.append(batch_features.cpu().numpy().reshape(images.shape[0], -1))
                labels.extend(batch_labels.tolist())

        X = np.concatenate(features)
        y = np.array(labels)

        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        # Entraînement du classificateur
        self.classifier = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            class_weight='balanced',
            n_jobs=-1,
            random_state=self.config.random_state,
            verbose=1
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Évaluation
        evaluation_results = self.evaluate(X_test, y_test)
        
        # Sauvegarde du modèle
        joblib.dump(self.classifier, self.results_dir / 'leukemia_detector.joblib')
        
        # Log des résultats
        self.logger.info("\nRésultats de l'évaluation:")
        self.logger.info(f"Score CV moyen: {evaluation_results['cv_scores_mean']:.3f} "
                        f"(±{evaluation_results['cv_scores_std']:.3f})")
        self.logger.info("\nRapport de classification:")
        self.logger.info(evaluation_results['classification_report'])
        
        return evaluation_results

    def predict(self, image_path: Path) -> Tuple[str, float]:
        """Prédit si une image de cellule montre des signes de leucémie"""
        if self.classifier is None:
            raise RuntimeError("Le modèle n'est pas entraîné")
        
        image = torchvision.io.read_image(str(image_path))
        image = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            image = image.to(self.device)
            features = self.feature_extractor(image)
            features = features.cpu().numpy().reshape(1, -1)
        
        prediction = self.classifier.predict(features)[0]
        probability = self.classifier.predict_proba(features).max()
        
        result = "Leucémie détectée" if prediction == 1 else "Pas de leucémie détectée"
        
        return result, probability

def main():
    # Configuration
    config = LeukemiaConfig()
    
    # Création et entraînement du détecteur
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
