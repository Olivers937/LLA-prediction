from dataclasses import dataclass
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import time
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict
import joblib
from contextlib import contextmanager

@dataclass
class MLConfig:
    """Configuration du pipeline ML"""
    data_dir: Path
    batch_size: int = 64
    num_workers: int = 8
    image_size: int = 224
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    rotation_degrees: int = 45
    model_name: str = "resnet18"
    use_gpu: bool = True

class MLPipeline:
    """Pipeline d'apprentissage automatique pour la classification d'images"""
    
    def __init__(self, config: MLConfig):
        """Initialise le pipeline avec la configuration donnée"""
        self.config = config
        self._setup_logging()
        self.device = self._setup_device()
        self.feature_extractor = None
        self.rf_classifier = None
        self.rf_classifier_aug = None
        self._setup_transforms()
        
    def _setup_logging(self) -> None:
        """Configure le système de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_device(self) -> torch.device:
        """Configure le device optimal pour PyTorch"""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            self.logger.info("Utilisation de CUDA")
        else:
            device = torch.device("cpu")
            self.logger.info("Utilisation du CPU")
        return device

    def _setup_transforms(self) -> None:
        """Prépare les transformations d'images"""
        self.transform = self._create_transforms()
        self.transform_aug = self._create_transforms(augment=True)

    def _create_transforms(self, augment: bool = False) -> transforms.Compose:
        """Crée les transformations pour les images"""
        transform_list = [
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
        ]
        
        if augment:
            transform_list.extend([
                transforms.RandomRotation(degrees=self.config.rotation_degrees),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transforms.Compose(transform_list)

    @contextmanager
    def _timer(self, description: str) -> None:
        """Context manager pour mesurer le temps d'exécution"""
        start = time.time()
        yield
        elapsed = time.time() - start
        self.logger.info(f"{description}: {elapsed:.2f} secondes")

    def _create_feature_extractor(self) -> None:
        """Crée et configure l'extracteur de caractéristiques"""
        model = getattr(torchvision.models, self.config.model_name)(
            weights='IMAGENET1K_V1'
        )
        self.feature_extractor = torch.nn.Sequential(
            *list(model.children())[:-1]
        ).to(self.device)
        self.feature_extractor.eval()

    def _create_dataloader(self, transform: transforms.Compose) -> DataLoader:
        """Crée un DataLoader pour les données"""
        dataset = torchvision.datasets.ImageFolder(
            root=self.config.data_dir / 'train',
            transform=transform
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def _extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extrait les caractéristiques des images"""
        features = []
        labels = []
        
        with torch.no_grad(), self._timer("Extraction des caractéristiques"):
            for images, batch_labels in dataloader:
                images = images.to(self.device, non_blocking=True)
                features_batch = self.feature_extractor(images)
                features.append(
                    features_batch.cpu().numpy().reshape(images.shape[0], -1)
                )
                labels.extend(batch_labels.tolist())
        
        return np.concatenate(features), np.array(labels)

    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> RandomForestClassifier:
        """Entraîne le classificateur Random Forest"""
        with self._timer("Entraînement du Random Forest"):
            clf = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                n_jobs=-1,
                random_state=self.config.random_state,
                verbose=1,
                class_weight='balanced'
            )
            clf.fit(X_train, y_train)
        return clf

    def _save_model(self, model: RandomForestClassifier, filename: str) -> None:
        """Sauvegarde le modèle avec joblib"""
        path = Path(filename)
        joblib.dump(model, path)
        self.logger.info(f"Modèle sauvegardé: {path.absolute()}")

    def train(self, save_models: bool = True) -> Dict[str, float]:
        """Exécute le pipeline d'entraînement complet"""
        if self.feature_extractor is None:
            self._create_feature_extractor()

        # Extraction des caractéristiques initiales
        dataloader = self._create_dataloader(self.transform)
        features, labels = self._extract_features(dataloader)
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=labels
        )
        
        # Entraînement initial
        self.rf_classifier = self._train_random_forest(X_train, y_train)
        if save_models:
            self._save_model(self.rf_classifier, 'rf_model.joblib')
        
        # Évaluation initiale
        y_pred = self.rf_classifier.predict(X_test)
        self.logger.info("\nRésultats sans augmentation:")
        self.logger.info("\n" + classification_report(y_test, y_pred))
        
        # Augmentation de données
        dataloader_aug = self._create_dataloader(self.transform_aug)
        features_aug, labels_aug = self._extract_features(dataloader_aug)
        
        # Entraînement avec augmentation
        X_train_aug = np.concatenate((X_train, features_aug))
        y_train_aug = np.concatenate((y_train, labels_aug))
        
        self.rf_classifier_aug = self._train_random_forest(X_train_aug, y_train_aug)
        if save_models:
            self._save_model(self.rf_classifier_aug, 'rf_model_aug.joblib')
        
        # Évaluation finale
        y_pred_aug = self.rf_classifier_aug.predict(X_test)
        self.logger.info("\nRésultats avec augmentation:")
        self.logger.info("\n" + classification_report(y_test, y_pred_aug))
        
        return {
            'accuracy_base': (y_pred == y_test).mean(),
            'accuracy_aug': (y_pred_aug == y_test).mean()
        }

    def predict(
        self,
        image_path: Path,
        use_augmented: bool = True
    ) -> Tuple[int, float]:
        """Prédit la classe d'une image"""
        if self.feature_extractor is None:
            raise RuntimeError("Le modèle n'est pas entraîné")
            
        transform = self.transform_aug if use_augmented else self.transform
        classifier = self.rf_classifier_aug if use_augmented else self.rf_classifier
        
        # Chargement et transformation de l'image
        image = torchvision.io.read_image(str(image_path))
        image = transform(image).unsqueeze(0)
        
        # Extraction des caractéristiques
        with torch.no_grad():
            image = image.to(self.device)
            features = self.feature_extractor(image)
            features = features.cpu().numpy().reshape(1, -1)
        
        # Prédiction
        prediction = classifier.predict(features)[0]
        probability = classifier.predict_proba(features).max()
        
        return prediction, probability

def main():
    config = MLConfig(
        data_dir=Path("chemin/vers/votre/repertoire/dimages"),
        batch_size=64,
        num_workers=8
    )
    
    pipeline = MLPipeline(config)
    results = pipeline.train()
    
    print(f"Accuracy sans augmentation: {results['accuracy_base']:.4f}")
    print(f"Accuracy avec augmentation: {results['accuracy_aug']:.4f}")

if __name__ == "__main__":
    main()
