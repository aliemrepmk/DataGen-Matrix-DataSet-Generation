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
    
    try {
        // Clear matrices and saved_matrices folder before generating new ones
        await fetch('/clear', { method: 'POST' });
        
        const formData = new FormData(e.target);
        
        // Send the actual request
        const response = await fetch('/generate', {
            method: 'POST',
            body: formData
        });

        // Create a reader for the response stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            // Decode the chunk and split by double newlines
            const chunk = decoder.decode(value);
            const events = chunk.split('\n\n');
            
            for (const event of events) {
                if (!event.trim()) continue;
                
                // Extract the data part after "data: "
                const dataStr = event.split('data: ')[1];
                if (!dataStr) continue;
                
                try {
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
                        // Display summary
                        const summary = data.summary;
                        results.innerHTML = `
                            <div class="card mb-4">
                                <div class="card-body">
                                    <h5 class="card-title">Summary</h5>
                                    <p>Total Matrices: ${summary.total_matrices}</p>
                                    <p>Upscale Operations: ${summary.upscale_count}</p>
                                    <p>Downscale Operations: ${summary.downscale_count}</p>
                                    <button onclick="saveMatrices()" class="btn btn-success mt-3">
                                        Save Matrices to Folder
                                    </button>
                                </div>
                            </div>
                        `;
                    }
                } catch (e) {
                    console.error('Error parsing event data:', e);
                }
            }
        }
        
    } catch (error) {
        results.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    } finally {
        loading.style.display = 'none';
    }
});