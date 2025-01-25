class LMADiagnosticApp {
    constructor() {
        this.imageUploader = new ImageUploader('preview');
        this.modelConfigurator = new ModelConfigurator('advancedMenu', 'modelSelection');
        this.resultDisplay = new ResultDisplay('result');

        this.initPredictButton();
    }

    initPredictButton() {
        const predictButton = document.querySelector('button');
        predictButton.addEventListener('click', this.predictLMA.bind(this));
    }

    predictLMA() {
        const imageFile = this.imageUploader.getImageFile();
        const selectedModels = this.modelConfigurator.getSelectedModels();

        if (!imageFile) {
            this.resultDisplay.showError('Veuillez importer une image');
            return;
        }

        // Simulation de prédiction
        const mockResult = {
            models: selectedModels,
            diagnosis: 'POSITIF',
            probability: 78,
            rocScore: 0.85
        };

        this.resultDisplay.displayResult(mockResult);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new LMADiagnosticApp();
});
