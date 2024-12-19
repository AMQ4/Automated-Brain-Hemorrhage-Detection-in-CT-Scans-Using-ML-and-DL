const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const detectButton = document.getElementById('detectButton');
const statusMessage = document.getElementById('statusMessage');
const uploadForm = document.getElementById('uploadForm');
const detectForm = document.getElementById('detectForm');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');

const allowedTypes = ['image/jpeg', 'image/png', 'application/dicom'];

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    handleFile(file);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFile(file);
});

function handleFile(file) {
    if (file && validateFile(file)) {
        statusMessage.textContent = `Potential Brain Hemorrhage Detected. Immediate Medical Attention Recommended.`;
        statusMessage.style.color = 'orange';
        
        detectForm.style.display = 'block';
        uploadButton.style.display = 'none';

        uploadForm.style.display = 'none';
        detectForm.style.display = 'block';

        if (file.type === 'image/jpeg' || file.type === 'image/png') {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImg.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        } else {
            imagePreview.style.display = 'none';
            previewImg.src = '#';
        }
    } else {
        statusMessage.textContent = 'Invalid file type. Please upload a valid CT scan image (.jpg, .jpeg, .png, .dcm).';
        statusMessage.style.color = 'red';
        imagePreview.style.display = 'none';
        previewImg.src = '#';
    }
}

function validateFile(file) {
    return allowedTypes.includes(file.type);
}