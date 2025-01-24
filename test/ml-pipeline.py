import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import time
from pathlib import Path
import logging
from typing import Tuple, List
import joblib
from dataclasses import dataclass
from contextlib import contextmanager

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration du modèle et des hyperparamètres"""
    data_dir: Path
    batch_size: int = 64
    num_workers: int = 8
    image_size: int = 224
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    rotation_degrees: int = 45

@contextmanager
def timer(description: str) -> None:
    """Context manager pour mesurer le temps d'exécution"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description}: {elapsed:.2f} secondes")

def get_device() -> torch.device:
    """Détermine et configure le device optimal"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Optimisation CUDA
    else:
        device = torch.device("cpu")
    return device

def create_feature_extractor(device: torch.device) -> torch.nn.Module:
    """Crée et configure l'extracteur de caractéristiques"""
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor.to(device)

def get_transforms(config: ModelConfig, augment: bool = False) -> transforms.Compose:
    """Crée les transformations pour les images"""
    transform_list = [
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
    ]
    
    if augment:
        transform_list.extend([
            transforms.RandomRotation(degrees=config.rotation_degrees),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)

def extract_features(
    dataloader: DataLoader,
    feature_extractor: torch.nn.Module,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrait les caractéristiques des images"""
    features = []
    labels = []
    
    with torch.no_grad(), timer("Extraction des caractéristiques"):
        for images, batch_labels in dataloader:
            images = images.to(device, non_blocking=True)
            features_batch = feature_extractor(images)
            features.append(features_batch.cpu().numpy().reshape(images.shape[0], -1))
            labels.extend(batch_labels.tolist())
    
    return np.concatenate(features), np.array(labels)

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ModelConfig
) -> RandomForestClassifier:
    """Entraîne le classificateur Random Forest"""
    with timer("Entraînement du Random Forest"):
        clf = RandomForestClassifier(
            n_estimators=config.n_estimators,
            n_jobs=-1,
            random_state=config.random_state,
            verbose=1
        )
        clf.fit(X_train, y_train)
    return clf

def save_model(model: RandomForestClassifier, filename: str) -> None:
    """Sauvegarde le modèle avec joblib"""
    joblib.dump(model, filename)
    logger.info(f"Modèle sauvegardé: {filename}")

def main():
    # Configuration
    config = ModelConfig(
        data_dir=Path("chemin/vers/votre/repertoire/dimages"),
        batch_size=64,
        num_workers=8
    )
    
    device = get_device()
    feature_extractor = create_feature_extractor(device)
    
    # Création des datasets et dataloaders
    transform = get_transforms(config)
    transform_aug = get_transforms(config, augment=True)
    
    dataset = torchvision.datasets.ImageFolder(
        root=config.data_dir / 'train',
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Extraction des caractéristiques initiales
    features, labels = extract_features(dataloader, feature_extractor, device)
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels
    )
    
    # Entraînement et évaluation du modèle initial
    rf_classifier = train_random_forest(X_train, y_train, config)
    save_model(rf_classifier, 'rf_model.joblib')
    
    y_pred = rf_classifier.predict(X_test)
    logger.info("\nRésultats sans augmentation:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Augmentation de données
    dataset_aug = torchvision.datasets.ImageFolder(
        root=config.data_dir / 'train',
        transform=transform_aug
    )
    dataloader_aug = DataLoader(
        dataset_aug,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Extraction des caractéristiques augmentées
    features_aug, labels_aug = extract_features(dataloader_aug, feature_extractor, device)
    
    # Combinaison des données et entraînement final
    X_train_aug = np.concatenate((X_train, features_aug))
    y_train_aug = np.concatenate((y_train, labels_aug))
    
    rf_classifier_aug = train_random_forest(X_train_aug, y_train_aug, config)
    save_model(rf_classifier_aug, 'rf_model_aug.joblib')
    
    y_pred_aug = rf_classifier_aug.predict(X_test)
    logger.info("\nRésultats avec augmentation:")
    logger.info("\n" + classification_report(y_test, y_pred_aug))

if __name__ == "__main__":
    main()
