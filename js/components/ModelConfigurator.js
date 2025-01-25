class ModelConfigurator {
    constructor(menuElementId, selectionContainerId) {
        this.menuElement = document.getElementById(menuElementId);
        this.selectionContainer = document.getElementById(selectionContainerId);
        this.initEventListeners();
    }

    initEventListeners() {
        this.menuElement.addEventListener('change', this.updateModelSelection.bind(this));
    }

    updateModelSelection() {
        const selectedValue = this.menuElement.value;
        const checkboxes = this.selectionContainer.querySelectorAll('input[name="model"]');

        if (selectedValue === '0') {
            this.selectionContainer.style.display = 'none';
        } else {
            this.selectionContainer.style.display = 'grid';
            checkboxes.forEach((checkbox, index) => {
                checkbox.checked = index < parseInt(selectedValue);
            });
        }
    }

    getSelectedModels() {
        const selectedModels = Array.from(
            this.selectionContainer.querySelectorAll('input[name="model"]:checked')
        ).map(checkbox => checkbox.value);
        return selectedModels;
    }
}
