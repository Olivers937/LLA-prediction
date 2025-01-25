import sys
import subprocess
from typing import List

class DependencyManager:
    @staticmethod
    def get_requirements() -> List[str]:
        return [
            'typer', 'loguru', 'tqdm', 'numpy', 'pandas', 
            'scikit-learn', 'opencv-python', 'scipy', 
            'scikit-image', 'matplotlib', 'flask', 
            'flask-cors', 'joblib', 'werkzeug'
        ]

    @classmethod
    def install_dependencies(cls):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        for package in cls.get_requirements():
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
