document.addEventListener("DOMContentLoaded", function () {
    const algorithmSelect = document.getElementById('generationAlgorithm');
    const kernelSizeGroup = document.getElementById('kernelSizeGroup');
    const imageResamplingGroup = document.getElementById('imageResamplingGroup');
    const waveletTypeGroup = document.getElementById('waveletTypeGroup');
    const blockSizeGroup = document.getElementById('blockSizeGroup');

    algorithmSelect.addEventListener('change', function () {
        const value = this.value;
        kernelSizeGroup.style.display = (value === "Lanczos Resampling") ? 'block' : 'none';
        imageResamplingGroup.style.display = (value === "Image-based Scaling") ? 'block' : 'none';
        waveletTypeGroup.style.display = (value === "Wavelet Transformation") ? 'block' : 'none';
        blockSizeGroup.style.display = (value === "Wavelet Transformation") ? 'block' : 'none';
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