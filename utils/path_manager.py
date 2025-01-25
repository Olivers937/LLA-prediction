from pathlib import Path

class PathManager:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.dataset_dir = self.project_root / "dataset"
        self.models_dir = self.project_root / "trained_models"
        self.processed_data_dir = self.dataset_dir / "processed_data"

    def ensure_dirs(self):
        """Create necessary directories if they don't exist"""
        self.models_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True, parents=True)
