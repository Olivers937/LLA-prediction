from core.dependency_manager import DependencyManager
from app import LMADiagnosticApp

def create_app(model_choice: int = 4):
    """
    Factory function to create LMA Diagnostic Application instance
    
    Args:
        model_choice (int): Number of models to use (0-4)
    
    Returns:
        LMADiagnosticApp: Configured application instance
    """
    return LMADiagnosticApp(model_choice)

def run_diagnostic(image_path: str, model_choice: int = 4):
    """
    Quick diagnostic function for standalone usage
    
    Args:
        image_path (str): Path to the medical image
        model_choice (int): Number of models to use (0-4)
    
    Returns:
        dict: Diagnostic results
    """
    app = create_app(model_choice)
    
    # Try to load pre-trained models, train if not available
    try:
        app.load_models()
    except:
        app.train_models()
    
    # Predict
    result = app.predict(image_path)
    return result

def main():
    """Main entry point for LMA Diagnostic Application"""
    print("LMA Diagnostic AI Application")
    DependencyManager.install_dependencies()
    print("Use create_app() or run_diagnostic() to start")

if __name__ == "__main__":
    main()
