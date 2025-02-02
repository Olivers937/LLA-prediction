import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from config import Config as config
from backend.features_extractor import extract_features
from models.base_model_wrapper import BaseModelWrapper

class LMADiagnosticApp:
    def __init__(self, model_choice: int = 4):
        """
        Initialize the backend with selected models
        
        Args:
            model_choice (int): Number of models to use (0-4)
        """
        self.models = self._initialize_models(model_choice)
        
        # Paths for saving/loading models
        self.models_dir = Path(__file__).parent / "trained_models"
        self.models_dir.mkdir(exist_ok=True)
    
    def _initialize_models(self, model_choice: int) -> list:
        """Initialize models based on configuration"""
        model_configs = {
            0: [RandomForestClassifier],
            1: [SVC],
            2: [ExtraTreesClassifier],
            3: [LogisticRegression],
            4: [RandomForestClassifier, SVC, ExtraTreesClassifier, LogisticRegression]
        }
        
        models = []
        selected_models = model_configs.get(model_choice, model_configs[4])
        
        model_params = {
            RandomForestClassifier: config.rf_params,
            SVC: config.svm_params,
            ExtraTreesClassifier: config.etc_params,
            LogisticRegression: config.lr_params
        }
        
        for model_class in selected_models:
            #model = model_class(random_state=config.random_state, probability=True)
            model = model_class(random_state=config.random_state)
            model_wrapper = BaseModelWrapper(
                model, 
                model_params[model_class], 
                model_class.__name__
            )
            models.append(model_wrapper)
        
        return models
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess the input image for feature extraction
        
        Args:
            image_path (str): Path to the uploaded image
        
        Returns:
            np.ndarray: Processed image
        """
        # Read image
        image = cv2.imread(image_path)
        
        if not os.path.exists(image_path):
	        raise FileNotFoundError(f"L'image {image_path} n'existe pas")
        
        # Resize image to standard size
        image = cv2.resize(image, config.image_size)
        
        return image
    
    def train_models(self, save: bool = True):
        """
        Train models on the dataset
        
        Args:
            save (bool): Whether to save trained models
        """
        # Load dataset
        from utils.data_loader import load_dataset
        
        malign_dir = [
            config.malignant_pre_b_dir.absolute(), 
            config.malignant_pro_b_dir.absolute(), 
            config.malignant_early_pre_b_dir.absolute()
        ]
        
        X, y = load_dataset([config.benign_dir.absolute()], malign_dir)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y
        )
        
        # Train models
        for model in self.models:
            model.fit(X_train, y_train)
        
        # Save models if requested
        if save:
            self.save_models()
    
    def save_models(self):
        """Save trained models to files"""
        for model in self.models:
            model_path = self.models_dir / f"{model.name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_path)
    
    def load_models(self):
        """Load pre-trained models"""
        for model in self.models:
            model_path = self.models_dir / f"{model.name.lower().replace(' ', '_')}_model.pkl"
            if model_path.exists():
                loaded_model = joblib.load(model_path)
                model.best_model = loaded_model.best_model
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predict LMA diagnosis from image
        
        Args:
            image_path (str): Path to uploaded image
        
        Returns:
            Dict: Prediction results
        """
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Extract features
        features = extract_features(processed_image)
        
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Prediction from each model
        predictions = {}
        probabilities = {}
        for model in self.models:
            pred = model.predict(features)
            proba = model.predict_proba(features)[:, 1]
            predictions[model.name] = pred[0]
            probabilities[model.name] = proba[0]
        
        # Ensemble decision (majority voting)
        unique_preds = list(set(predictions.values()))
        voting_results = [list(predictions.values()).count(pred) for pred in unique_preds]
        final_pred = unique_preds[voting_results.index(max(voting_results))]
        
        # Calculate overall probability
        avg_probability = np.mean(list(probabilities.values()))
        
        return {
            "diagnosis": "POSITIF" if final_pred == 1 else "NÉGATIF",
            "probability": round(avg_probability * 100, 2),
            "models": list(predictions.keys()),
            "individual_predictions": predictions,
            "individual_probabilities": {k: round(v * 100, 2) for k, v in probabilities.items()}
        }

def run_diagnostic(image_path: str, model_choice: int = 4):
    """
    Run complete diagnostic workflow
    
    Args:
        image_path (str): Path to uploaded image
        model_choice (int): Number of models to use
    
    Returns:
        Dict: Diagnostic results
    """
    backend = LMADiagnosticApp(model_choice)
    
    # Try to load pre-trained models, train if not available
    try:
        backend.load_models()
    except:
        backend.train_models()
    
    # Predict
    result = backend.predict(image_path)
    return result

if __name__ == "__main__":
    # Example usage
    result = run_diagnostic("./test.jpeg")
    print(result)
