class ImageUploader {
    constructor(previewElementId) {
        this.inputElement = document.getElementById('imageUpload');
        this.previewElement = document.getElementById(previewElementId);
        this.initEventListeners();
    }

    initEventListeners() {
        this.inputElement.addEventListener('change', this.previewImage.bind(this));
    }

    previewImage(event) {
        const file = event.target.files[0];
        const reader = new FileReader();

        reader.onloadend = () => {
            this.previewElement.src = reader.result;
            this.previewElement.style.display = 'block';
        }

        if (file) {
            reader.readAsDataURL(file);
        }
    }

    getImageFile() {
        return this.inputElement.files[0];
    }
}
