from typing import Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

class BaseModelWrapper:
    def __init__(self, model: BaseEstimator, params: Dict[str, Any], name: str):
        """
        Wrapper for machine learning models with grid search capabilities
        
        Args:
            model (BaseEstimator): Sklearn-compatible classifier
            params (Dict[str, Any]): Hyperparameter grid for model
            name (str): Model name for identification
        """
        self.model = model
        self.params = params
        self.name = name
        self.best_model = None
        self.classification_report = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit model with grid search cross-validation
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
        """
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.params,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        return self.best_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.best_model.predict_proba(X)
    
    def to_string(self) -> str:
        """Generate model performance summary"""
        return f"Model: {self.name}"
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate model performance
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
        
        Returns:
            Dict: Classification report
        """
        y_pred = self.predict(X_test)
        self.classification_report = classification_report(
            y_test, y_pred, output_dict=True
        )
        return self.classification_report
