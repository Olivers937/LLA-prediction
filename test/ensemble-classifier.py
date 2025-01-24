from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
import logging
from pathlib import Path
from datetime import datetime
import joblib
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class EnsembleConfig:
    """Configuration pour l'ensemble de classificateurs"""
    n_jobs: int = -1  # Utiliser tous les cores disponibles
    cv_folds: int = 5
    random_state: int = 42
    test_size: float = 0.2
    
    # Paramètres de grille pour chaque modèle
    rf_params: Dict = None
    etc_params: Dict = None
    svm_params: Dict = None
    lr_params: Dict = None
    
    def __post_init__(self):
        # Grilles de paramètres par défaut pour chaque modèle
        self.rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        self.etc_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        self.svm_params = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear']
        }
        
        self.lr_params = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

class BaseModelWrapper:
    """Classe de base pour wrapper les modèles individuels"""
    
    def __init__(self, model: BaseEstimator, param_grid: Dict, name: str):
        self.model = model
        self.param_grid = param_grid
        self.name = name
        self.best_model = None
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entraîne le modèle avec recherche de paramètres"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Recherche des meilleurs paramètres
        grid_search = GridSearchCV(
            self.model,
            self.param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        grid_search.fit(X_scaled, y)
        self.best_model = grid_search.best_estimator_
        logging.info(f"{self.name} - Meilleurs paramètres: {grid_search.best_params_}")
        logging.info(f"{self.name} - Score: {grid_search.best_score_:.3f}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions avec le meilleur modèle"""
        if self.best_model is None:
            raise ValueError(f"{self.name} n'est pas encore entraîné")
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne les probabilités de prédiction"""
        if self.best_model is None:
            raise ValueError(f"{self.name} n'est pas encore entraîné")
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict_proba(X_scaled)

class VotingEnsemble:
    """Ensemble de modèles avec système de vote"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = []
        self.setup_logging()
        self._initialize_models()
        
    def setup_logging(self) -> None:
        """Configure le système de logging"""
        self.results_dir = Path('results') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
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
                "Random Forest"
            ),
            BaseModelWrapper(
                ExtraTreesClassifier(random_state=self.config.random_state),
                self.config.etc_params,
                "Extra Trees"
            ),
            BaseModelWrapper(
                SVC(probability=True, random_state=self.config.random_state),
                self.config.svm_params,
                "SVM"
            ),
            BaseModelWrapper(
                LogisticRegression(random_state=self.config.random_state),
                self.config.lr_params,
                "Logistic Regression"
            )
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entraîne tous les modèles en parallèle"""
        logging.info("Démarrage de l'entraînement des modèles...")
        
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            for model in self.models:
                executor.submit(model.fit, X, y)
                
        logging.info("Entraînement terminé")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions avec vote majoritaire"""
        predictions = np.array([model.predict(X) for model in self.models])
        # Vote majoritaire
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Combine les probabilités de tous les modèles"""
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(probas, axis=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Évalue l'ensemble et génère des visualisations"""
        predictions = self.predict(X)
        probas = self.predict_proba(X)
        
        # Calcul des métriques
        results = {
            'classification_report': classification_report(y, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y, predictions),
            'roc_auc': roc_auc_score(y, probas[:, 1])
        }
        
        # Évaluation individuelle des modèles
        individual_scores = {}
        for model in self.models:
            model_preds = model.predict(X)
            individual_scores[model.name] = {
                'accuracy': classification_report(y, model_preds, output_dict=True)['accuracy'],
                'roc_auc': roc_auc_score(y, model.predict_proba(X)[:, 1])
            }
        
        results['individual_scores'] = individual_scores
        
        # Génération des visualisations
        self._plot_confusion_matrix(results['confusion_matrix'])
        self._plot_model_comparison(individual_scores)
        
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Génère et sauvegarde la matrice de confusion"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion - Ensemble')
        plt.ylabel('Vrai label')
        plt.xlabel('Prédiction')
        plt.savefig(self.results_dir / 'confusion_matrix.png')
        plt.close()
    
    def _plot_model_comparison(self, scores: Dict) -> None:
        """Génère un graphique comparatif des performances"""
        models = list(scores.keys())
        accuracies = [scores[m]['accuracy'] for m in models]
        roc_aucs = [scores[m]['roc_auc'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, accuracies, width, label='Accuracy')
        ax.bar(x + width/2, roc_aucs, width, label='ROC AUC')
        
        ax.set_ylabel('Score')
        ax.set_title('Comparaison des performances des modèles')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison.png')
        plt.close()
    
    def save_model(self, path: str) -> None:
        """Sauvegarde l'ensemble complet"""
        joblib.dump(self, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'VotingEnsemble':
        """Charge un ensemble sauvegardé"""
        return joblib.load(path)

def main():
    # Configuration
    config = EnsembleConfig()
    
    # Création et entraînement de l'ensemble
    ensemble = VotingEnsemble(config)
    
    # Supposons que nous ayons les données de votre détecteur de leucémie
    # X, y = load_data()  # À implémenter selon votre source de données
    
    # Entraînement
    ensemble.fit(X, y)
    
    # Évaluation
    results = ensemble.evaluate(X_test, y_test)
    
    # Sauvegarde du modèle
    ensemble.save_model('ensemble_model.joblib')
    
    logging.info(f"Performance finale (ROC AUC): {results['roc_auc']:.3f}")

if __name__ == "__main__":
    main()
