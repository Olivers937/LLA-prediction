import logging
from typing import Dict

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from EDA.config import Config as config


class BaseModelWrapper:
    """Wrapper pour les modèles de classification binaire"""

    def __init__(self, model, param_grid: Dict, name: str):
        self.model = model
        self.param_grid = param_grid
        self.name = name
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
            cv=config.cv_folds,
            n_jobs=config.n_jobs,
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
        return "BaseModelWrapper : {" + f"model = {self.model}; name = {self.name}; config = {config}; best_model = {self.best_model}; scaler = {self.scaler}" + "}"


if __name__ == '__main__':
    print("bonjour le monde")