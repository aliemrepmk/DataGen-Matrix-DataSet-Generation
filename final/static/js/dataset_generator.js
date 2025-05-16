// Clear matrices when page loads
window.addEventListener('load', async () => {
    await fetch('/clear', { method: 'POST' });
});

async function saveMatrices() {
    try {
        // Show loading state with just the spinner and text
        const loading = document.getElementById('loading');
        const progressBar = document.getElementById('progress-bar').parentElement; // Get the progress container
        loading.style.display = 'block';
        loading.querySelector('p').textContent = 'Saving matrices...';
        progressBar.style.display = 'none'; // Hide the entire progress bar container

        // Send request to save matrices
        const response = await fetch('/save_matrices', {
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

        // Fetch and display visualization of one random upscaled and one random downscaled matrix
        const visResponse = await fetch('/visualize-saved-matrices');
        const visData = await visResponse.json();
        if (visData.success && visData.image_path) {
            // Insert the image below the summary card
            const results = document.getElementById('results');
            let visDiv = document.getElementById('matrix-visualization');
            if (!visDiv) {
                visDiv = document.createElement('div');
                visDiv.id = 'matrix-visualization';
                visDiv.className = 'mt-4 text-center';
                results.appendChild(visDiv);
            }
            const timestamp = new Date().getTime();
            const imagePath = `${visData.image_path}?t=${timestamp}`;
            visDiv.innerHTML = `<h5>Random Downscaled & Upscaled Matrix Visualization</h5><img src="${imagePath}" alt="Matrix Visualization" class="img-fluid" style="max-width: 100%; height: auto; border: 1px solid #ccc;" />`;
        }
    } catch (error) {
        // Show error message in a modal
        const messageModal = new bootstrap.Modal(document.getElementById('messageModal'));
        document.getElementById('messageModalBody').textContent = `Error saving matrices: ${error.message}`;
        messageModal.show();
    } finally {
        // Only hide the loading spinner and reset its text
        const loading = document.getElementById('loading');
        loading.style.display = 'none';
        loading.querySelector('p').textContent = 'Generating matrices...';
        // Don't show the progress bar again
    }
}

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
document.getElementById('matrixForm').addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent form from refreshing the page
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const progressBar = document.getElementById('progress-bar');
    const progressContainer = progressBar.parentElement;

    loading.style.display = 'block';
    results.innerHTML = '';
    progressContainer.style.display = 'block'; // Show the progress bar container
    progressBar.style.width = '0%';
    progressBar.textContent = '0%';

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
        const formElements = Array.from(e.target.elements).filter(el => el.name && el.type !== 'file');
        formElements.forEach(el => {
            formData.append(el.name, el.value);
        });

        // Send the request
        const response = await fetch('/generate', {
            method: 'POST',
            body: formData
        });

        // Handle response stream (progress updates)
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const events = chunk.split('\n\n');

            for (const event of events) {
                if (!event.trim()) continue;

                const dataStr = event.split('data: ')[1];
                if (!dataStr) continue;

                const data = JSON.parse(dataStr);

                if (data.error) {
                    results.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    return;
                }

                if (data.progress !== undefined) {
                    const progress = Math.round(data.progress);
                    progressBar.style.width = `${progress}%`;
                    progressBar.textContent = `${progress}%`;
                    progressBar.setAttribute('aria-valuenow', progress);
                }

                if (data.success) {
                    const summary = data.summary;
                    let methodList = '';
                    if (summary.method_counts) {
                        methodList = '<h6>Method Usage:</h6><ul>';
                        for (const [method, count] of Object.entries(summary.method_counts)) {
                            methodList += `<li>${method}: ${count}</li>`;
                        }
                        methodList += '</ul>';
                    }
                    results.innerHTML = `
                        <div class="card mb-4">
                            <div class="card-body">
                                <h5 class="card-title">Summary</h5>
                                <p>Total Matrices: ${summary.total_matrices}</p>
                                <p>Upscale Operations: ${summary.upscale_count}</p>
                                <p>Downscale Operations: ${summary.downscale_count}</p>
                                ${methodList}
                                <button onclick="saveMatrices()" class="btn btn-success mt-3">
                                    Save Matrices to Folder
                                </button>
                            </div>
                        </div>
                    `;
                }
            }
        }
    } catch (error) {
        results.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    } finally {
        loading.style.display = 'none';
    }
});

function updateVisualization(data) {
    const visualizationContainer = document.getElementById('visualization');
    if (data.image_path) {
        const timestamp = new Date().getTime();
        const imagePath = `${data.image_path}?t=${timestamp}`;
        visualizationContainer.innerHTML = `<img src="${imagePath}" class="img-fluid" alt="Matrix Visualization">`;
    }
}