class ResultDisplay {
    constructor(resultElementId) {
        this.resultElement = document.getElementById(resultElementId);
    }

    displayResult(resultData) {
        this.resultElement.innerHTML = `
            <div class="bg-gradient-to-br from-[#3a4b5c] to-[#2c3e50] p-6 rounded-lg shadow-xl">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-2xl font-bold text-pink-400">Résultats Diagnostic</h3>
                    <i class="fas fa-chart-line text-purple-500 text-3xl"></i>
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <p class="text-white">Modèles : ${resultData.models.join(', ')}</p>
                        <p>LMA Probable : <span class="text-green-400 font-bold">${resultData.diagnosis}</span></p>
                    </div>
                    <div>
                        <p>Probabilité : <span class="text-pink-400 text-2xl">${resultData.probability}%</span></p>
                        <p>Score ROC : <span class="text-purple-400 text-xl">${resultData.rocScore}</span></p>
                    </div>
                </div>
            </div>
        `;
    }

    showError(message) {
        this.resultElement.innerHTML = `<p class="text-red-400">${message}</p>`;
    }
}
