from flask import Flask, render_template, request, send_file, abort, make_response , session
import os
import secrets
from werkzeug.utils import secure_filename
import scipy.io as sio
from logic.load_matrix import load_matrix
from logic.optimize import optimize_matrix
from logic.wavelet import scale_sparse_matrix_wavelet
from logic.nearest import scale_sparse_matrix_nearest
from logic.bilinear import scale_sparse_matrix_bilinear
from logic.lanczos import scale_sparse_matrix_lanczos
from logic.gaussian import gaussian_pyramid_sparse
from logic.image   import scale_sparse_matrix_image, create_heatmap
from logic.visualize_matrix import visualize_matrices , visualize_heatmaps
from PIL import Image
from scipy.sparse import csr_matrix
from logic.graph import scale_sparse_matrix_graph

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded-matrices'
GENERATED_FOLDER = 'generated-matrices'
OPTIMIZED_FOLDER = 'optimized-matrices'
app.secret_key = secrets.token_hex(16)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
os.makedirs(OPTIMIZED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    # Clear previous visualization if it exists
    visual_path = os.path.join("static", "last_visualization.png")
    if os.path.exists(visual_path):
        os.remove(visual_path)

    return render_template('index.html')

@app.route('/visualization')
def serve_visualization():
    visual_path = "static/last_visualization.png"
    if os.path.exists(visual_path):
        response = make_response(send_file(visual_path, mimetype='image/png'))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    else:
        return '', 404

@app.route('/generate', methods=['POST'])
def generate():
    # 1 ─────────── upload or reuse matrix
    uploaded_file = request.files.get('matrix_upload')
    if uploaded_file and uploaded_file.filename.endswith('.mtx'):
        matrix_filename = secure_filename(uploaded_file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, matrix_filename)
        uploaded_file.save(upload_path)
        session['matrix_file'] = matrix_filename
    elif 'matrix_file' in session:
        matrix_filename = session['matrix_file']
        upload_path = os.path.join(UPLOAD_FOLDER, matrix_filename)
        if not os.path.exists(upload_path):
            session.pop('matrix_file', None)
            return "Previously uploaded matrix file is missing. Please re-upload.", 400
    else:
        return "No valid matrix file uploaded and no previous file found.", 400

    # 2 ─────────── form values
    form_data  = request.form
    algorithm  = form_data.get('algorithm')
    try:
        rows = int(form_data.get('rows', 0))
        cols = int(form_data.get('cols', 0))
    except ValueError:
        return "Invalid row/column size provided.", 400

    match_nnz = 'match_nnz' in form_data
    optimize   = 'optimize' in form_data

    kernel_size      = int(form_data.get('kernel_size', 3) or 3)
    image_resampling = form_data.get('image_resampling_method', 'Image.BOX')
    wavelet_type     = form_data.get('wavelet_type', 'db4')
    block_size       = int(form_data.get('block_size', 2) or 2)
    if block_size % 2:
        block_size += 1  # ensure even

    # 3 ─────────── load matrix
    if not os.path.exists(upload_path):
        return f"Matrix file not found at {upload_path}", 500
    matrix = load_matrix(upload_path)
    if matrix is None:
        return "Failed to load uploaded matrix.", 500

    # 4 ─────────── output filename
    safe_algo   = algorithm.replace(' ', '_')
    output_name = f"{os.path.splitext(matrix_filename)[0]}_{safe_algo}.mtx"
    output_path = os.path.join(OPTIMIZED_FOLDER if optimize else GENERATED_FOLDER, output_name)

    # 5 ─────────── run algorithm
    try:
        if algorithm == "Wavelet Transformation":
            result = scale_sparse_matrix_wavelet(matrix, rows, cols, wavelet_type, block_size)
        elif algorithm == "Nearest-neighbor Interpolation":
            result = scale_sparse_matrix_nearest(matrix, rows, output_path, match_nnz)
        elif algorithm == "Bi-linear Interpolation":
            result = scale_sparse_matrix_bilinear(matrix, rows, output_path, match_nnz)
        elif algorithm == "Lanczos Resampling":
            result = scale_sparse_matrix_lanczos(matrix, rows, output_path, match_nnz, kernel_size)
        elif algorithm == "Gaussian Pyramids":
            result = gaussian_pyramid_sparse(matrix)
        elif algorithm == "Image-based Scaling":
            resize_method = getattr(Image, image_resampling.split('.')[-1])
            result = scale_sparse_matrix_image(matrix, rows, output_path, resize_method)
        elif algorithm == "Graph Coarsening & Refinement":
            result = scale_sparse_matrix_graph(matrix, rows, output_path, match_nnz)
        else:
            return "Unsupported algorithm selected.", 400
    except Exception as e:
        return f"Algorithm processing failed: {e}", 500

    # 6 ─────────── optimise
    if optimize:
        try:
            result = optimize_matrix(result)
        except Exception as e:
            return f"Matrix optimisation failed: {e}", 500

    # 7 ─────────── save matrix
    try:
        sio.mmwrite(output_path, result)
    except Exception as e:
        return f"Failed to save generated matrix: {e}", 500

    # 8-a ───────── visualise matrices side-by-side
    visual_path = os.path.join("static", "last_visualization.png")
    try:
        visualize_matrices(matrix, result, save_path=visual_path)
    except Exception as e:
        return f"Matrix visualisation failed: {e}", 500

    heatmap_pair_path = os.path.join("static", "heatmap_pair.png")
    try:
            visualize_heatmaps(matrix, result, save_path=heatmap_pair_path)
    except Exception as e:
            return f"Heat-map visualisation failed: {e}", 500

    # 9 ─────────── render template
    return render_template(
    "index.html",
    image_path         = visual_path,          # ← your spy image (unchanged)
    heatmap_pair_path  = heatmap_pair_path,    # ← new combined heat-map
    form_data          = form_data,
    selected_algorithm = algorithm,
    uploaded_filename  = session.get('matrix_file')
)


if __name__ == '__main__':
    app.run(debug=True)