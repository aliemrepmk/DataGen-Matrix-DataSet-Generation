document.addEventListener("DOMContentLoaded", function () {
    const algorithmSelect = document.getElementById('generationAlgorithm');
    const kernelSizeGroup = document.getElementById('kernelSizeGroup');
    const imageResamplingGroup = document.getElementById('imageResamplingGroup');
    const waveletTypeGroup = document.getElementById('waveletTypeGroup');

    algorithmSelect.addEventListener('change', function () {
        const value = this.value;
        kernelSizeGroup.style.display = (value === "Lanczos Resampling") ? 'block' : 'none';
        imageResamplingGroup.style.display = (value === "Image-based Scaling") ? 'block' : 'none';
        waveletTypeGroup.style.display = (value === "Wavelet Transformation") ? 'block' : 'none';
    });
});
document.addEventListener("DOMContentLoaded", () => {
    const btn     = document.getElementById("toggleHeatBtn");
    const section = document.getElementById("heatmapSection");

    if (!btn || !section) return;            // no heat-map yet

    btn.addEventListener("click", () => {
        const shown = section.style.display !== "none";
        section.style.display = shown ? "none" : "block";
        btn.textContent = shown ? "Show Heat-map" : "Hide Heat-map";
    });
});

// Folder selection and .mtx file listing
const folderInput = document.getElementById('matrix_folder');
const dropdownGroup = document.getElementById('matrixFileDropdownGroup');
const fileDropdown = document.getElementById('matrix_file_select');
let folderFiles = [];

folderInput.addEventListener('change', function (e) {
    folderFiles = Array.from(e.target.files);
    // Filter for .mtx files
    const mtxFiles = folderFiles.filter(f => f.name.endsWith('.mtx'));
    fileDropdown.innerHTML = '';
    if (mtxFiles.length > 0) {
        mtxFiles.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f.name;
            opt.textContent = f.name;
            fileDropdown.appendChild(opt);
        });
        dropdownGroup.style.display = 'block';
    } else {
        dropdownGroup.style.display = 'none';
    }
});

// Handle form submission
const matrixForm = document.getElementById('matrixForm');
matrixForm.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent form from refreshing the page
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    loading.style.display = 'block';
    results.innerHTML = '';

    // Get selected file name
    const selectedFileName = fileDropdown.value;
    const selectedFile = folderFiles.find(f => f.name === selectedFileName);
    if (!selectedFile) {
        loading.style.display = 'none';
        // Show error in modal
        const messageModal = new bootstrap.Modal(document.getElementById('messageModal'));
        document.getElementById('messageModalBody').textContent = 'No matrix file selected.';
        messageModal.show();
        return;
    }

    try {
        const formData = new FormData();
        // Append only the selected file as 'matrix_file'
        formData.append('matrix_file', selectedFile, selectedFile.name);
        // Append other form fields
        const formElements = Array.from(matrixForm.elements).filter(el => el.name && el.type !== 'file');
        formElements.forEach(el => {
            if (el.type === 'checkbox') {
                if (el.checked) formData.append(el.name, 'on');
            } else {
                formData.append(el.name, el.value);
            }
        });

        // Send the request
        const response = await fetch('/generate-matrix', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.error) {
            results.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            return;
        }
        if (data.success) {
            // Display summary with save button
            const matrixInfo = data.matrix_info;
            results.innerHTML = `
                <div class="card mb-4">
                    <div class="card-body">
                        <h5 class="card-title">Matrix Generated Successfully</h5>
                        <p>Shape: ${matrixInfo.shape[0]} x ${matrixInfo.shape[1]}</p>
                        <p>Non-zero elements: ${matrixInfo.nnz}</p>
                        <p>Density: ${(matrixInfo.density * 100).toFixed(2)}%</p>
                        <button onclick="saveMatrix()" class="btn btn-success mt-3">
                            Save Matrix to Folder
                        </button>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        results.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    } finally {
        loading.style.display = 'none';
    }
});

async function saveMatrix() {
    try {
        // Show loading state
        const loading = document.getElementById('loading');
        loading.style.display = 'block';
        loading.querySelector('p').textContent = 'Saving matrix...';

        // Send request to save matrix
        const response = await fetch('/save-matrix', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Show success message in a modal
        const messageModal = new bootstrap.Modal(document.getElementById('messageModal'));
        document.getElementById('messageModalBody').textContent = data.message;
        messageModal.show();

    } catch (error) {
        // Show error message in a modal
        const messageModal = new bootstrap.Modal(document.getElementById('messageModal'));
        document.getElementById('messageModalBody').textContent = `Error saving matrix: ${error.message}`;
        messageModal.show();
    } finally {
        // Hide loading spinner and reset text
        const loading = document.getElementById('loading');
        loading.style.display = 'none';
        loading.querySelector('p').textContent = 'Generating matrix...';
    }
}