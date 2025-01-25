import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from config import Config
from utils.path_manager import PathManager

class ModelManager:
    """Manages machine learning models for LMA diagnosis"""
    
    def __init__(self, model_choice: int = 4):
        """
        Initialize models based on choice
        
        Args:
            model_choice (int): Number of models to use (0-4)
        """
        self.paths = PathManager()
        self.model_choice = model_choice
        self.models = self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models based on model_choice"""
        model_configs = {
            0: [RandomForestClassifier],
            1: [SVC],
            2: [ExtraTreesClassifier],
            3: [LogisticRegression],
            4: [RandomForestClassifier, SVC, ExtraTreesClassifier, LogisticRegression]
        }
        
        return [
            model(probability=True, random_state=Config.random_state)
            for model in model_configs.get(self.model_choice, model_configs[4])
        ]
    
    def train_models(self, X_train, y_train):
        """Train models on the dataset"""
        for model in self.models:
            model.fit(X_train, y_train)
        
        self._save_models()
    
    def _save_models(self):
        """Save trained models"""
        os.makedirs(self.paths.models_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            path = os.path.join(self.paths.models_dir, f'model_{i}.pkl')
            joblib.dump(model, path)
    
    def load_models(self):
        """Load pre-trained models"""
        for i, model in enumerate(self.models):
            path = os.path.join(self.paths.models_dir, f'model_{i}.pkl')
            if os.path.exists(path):
                self.models[i] = joblib.load(path)
