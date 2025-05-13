import os
import shutil
import random
import json
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context, redirect, url_for
from werkzeug.utils import secure_filename

from logic.load_matrix import load_matrix
from logic.save_matrix import save_matrix
from logic.bilinear import scale_sparse_matrix_bilinear
from logic.dct import scale_sparse_matrix_dct_blockwise
from logic.fourier import scale_sparse_matrix_fourier
from logic.gaussian import scale_sparse_matrix_gaussian
from logic.graph import scale_sparse_matrix_graph
from logic.image import scale_sparse_matrix_image
from logic.lanczos import scale_sparse_matrix_lanczos
from logic.nearest import scale_sparse_matrix_nearest
from logic.wavelet import scale_sparse_matrix_wavelet
from logic.optimize import optimize_matrix

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key-here'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define available scaling methods
scaling_methods = {
    'bilinear': scale_sparse_matrix_bilinear,
    'dct': scale_sparse_matrix_dct_blockwise,
    'fourier': scale_sparse_matrix_fourier,
    'gaussian': scale_sparse_matrix_gaussian,
    'graph': scale_sparse_matrix_graph,
    'image': scale_sparse_matrix_image,
    'lanczos': scale_sparse_matrix_lanczos,
    'nearest': scale_sparse_matrix_nearest,
    'wavelet': scale_sparse_matrix_wavelet
}

# Define which methods can be used for upscaling
upscale_methods = ['bilinear', 'dct', 'fourier', 'graph', 'image', 'nearest', 'wavelet']

# Global lists to store generated matrices
downscaled_matrices = []
upscaled_matrices = []
generated_matrix = None

def clear_saved_matrices():
    """Clear the saved_matrices directory"""
    save_dir = os.path.join(os.getcwd(), 'saved_matrices')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mtx'}

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/matrix-generator')
def matrix_generator():
    global generated_matrix
    generated_matrix = None
    return render_template('matrix_generator.html')

@app.route('/generate-matrix', methods=['POST'])
def generate_single_matrix():
    try:
        if 'matrix_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['matrix_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only .mtx files are allowed'}), 400
        
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load the input matrix
            input_matrix = load_matrix(filepath)
            if input_matrix is None:
                return jsonify({'error': 'Failed to load input matrix'}), 400
            
            # Get parameters from the form
            algorithm = request.form.get('algorithm')
            rows = int(request.form.get('rows'))
            cols = int(request.form.get('cols'))
            match_nnz = 'match_nnz' in request.form
            optimize = 'optimize' in request.form
            
            # Get algorithm-specific parameters
            wavelet_type = int(request.form.get('wavelet_type', 1)) if algorithm == 'wavelet' else None
            image_resampling_method = request.form.get('image_resampling_method') if algorithm == 'image' else None
            kernel_size = int(request.form.get('kernel_size', 3)) if algorithm == 'lanczos' else None
            
            # Select the appropriate scaling method
            scaling_func = scaling_methods.get(algorithm)
            if not scaling_func:
                return jsonify({'error': 'Invalid algorithm selected'}), 400
            
            # Generate the matrix
            global generated_matrix
            if algorithm == 'wavelet':
                generated_matrix = scaling_func(input_matrix, rows, wavelet_type)
            elif algorithm == 'image':
                generated_matrix = scaling_func(input_matrix, rows, image_resampling_method)
            elif algorithm == 'lanczos':
                generated_matrix = scaling_func(input_matrix, rows, kernel_size)
            else:
                generated_matrix = scaling_func(input_matrix, rows)
            
            # Optimize if requested
            if optimize:
                generated_matrix = optimize_matrix(generated_matrix)
            
            # Clear the saved_matrices folder before saving
            clear_saved_matrices()

            # Save the generated matrix
            save_dir = os.path.join(os.getcwd(), 'saved_matrices')
            os.makedirs(save_dir, exist_ok=True)
            save_matrix(generated_matrix, 'generated_matrix', save_dir)
            
            return jsonify({
                'success': True,
                'message': 'Matrix generated and saved successfully',
                'matrix_info': {
                    'shape': generated_matrix.shape,
                    'nnz': generated_matrix.nnz,
                    'density': generated_matrix.nnz / (generated_matrix.shape[0] * generated_matrix.shape[1])
                }
            })
            
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-dataset')
def dataset_generator():
    # Clear the matrices when the page is loaded
    global downscaled_matrices, upscaled_matrices
    downscaled_matrices = []
    upscaled_matrices = []
    clear_saved_matrices()  # Clear saved_matrices folder
    return render_template('dataset_generator.html')

@app.route('/generate', methods=['POST'])
def generate_matrices():
    def generate():
        try:
            # Initialize progress
            session['progress'] = 0
            session.modified = True
            
            # Get parameters from the form
            num_matrices = int(request.form.get('num_matrices', 1))
            min_dim = int(request.form.get('min_dim', 100))
            max_dim = int(request.form.get('max_dim', 1000))
            
            # Validate inputs
            if num_matrices <= 0 or min_dim <= 0 or max_dim <= 0:
                yield f"data: {json.dumps({'error': 'All parameters must be positive integers'})}\n\n"
                return
            
            if max_dim < min_dim:
                yield f"data: {json.dumps({'error': 'Maximum dimension must be greater than or equal to minimum dimension'})}\n\n"
                return
            
            # Check if file was uploaded
            if 'matrix_file' not in request.files:
                yield f"data: {json.dumps({'error': 'No file uploaded'})}\n\n"
                return
                
            file = request.files['matrix_file']
            if file.filename == '':
                yield f"data: {json.dumps({'error': 'No file selected'})}\n\n"
                return
                
            if not allowed_file(file.filename):
                yield f"data: {json.dumps({'error': 'Invalid file type. Only .mtx files are allowed'})}\n\n"
                return
            
            # Save the uploaded file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Load the input matrix
                input_matrix = load_matrix(filepath)
                if input_matrix is None:
                    yield f"data: {json.dumps({'error': 'Failed to load input matrix'})}\n\n"
                    return
                    
                # Generate matrices
                upscale_count = 0
                downscale_count = 0
                
                for i in range(num_matrices):
                    operation = random.randint(0, 1)
                    
                    if operation == 0:  # downscale
                        # Select a random method from all available methods
                        method_name = random.choice(list(scaling_methods.keys()))
                        new_dim = random.randint(min_dim, input_matrix.shape[0])
                        
                        # Skip wavelet and bilinear for downscaling
                        while method_name in ['wavelet', 'bilinear']:
                            method_name = random.choice(list(scaling_methods.keys()))
                        
                        scaling_func = scaling_methods[method_name]
                        scaled_matrix = scaling_func(input_matrix, new_dim)
                        downscaled_matrices.append(scaled_matrix)
                        downscale_count += 1
                        
                    else:  # upscale
                        # Select a random method from upscale methods
                        method_name = random.choice(upscale_methods)
                        new_dim = random.randint(input_matrix.shape[0], max_dim)
                        
                        # Skip wavelet and bilinear for upscaling
                        while method_name in ['wavelet', 'bilinear']:
                            method_name = random.choice(upscale_methods)
                        
                        scaling_func = scaling_methods[method_name]
                        scaled_matrix = scaling_func(input_matrix, new_dim)
                        upscaled_matrices.append(scaled_matrix)
                        upscale_count += 1
                    
                    # Send progress update
                    progress = (i + 1) / num_matrices * 100
                    yield f"data: {json.dumps({'progress': progress})}\n\n"
                
                # Send final result
                yield f"data: {json.dumps({'success': True, 'summary': {'total_matrices': num_matrices, 'upscale_count': upscale_count, 'downscale_count': downscale_count}})}\n\n"
                
            finally:
                # Clean up the uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/clear', methods=['POST'])
def clear_matrices():
    """Endpoint to clear stored matrices"""
    global downscaled_matrices, upscaled_matrices
    downscaled_matrices = []
    upscaled_matrices = []
    session['progress'] = 0  # Reset progress
    clear_saved_matrices()  # Clear saved_matrices folder
    return jsonify({'success': True, 'message': 'All matrices cleared'})

@app.route('/save_matrices', methods=['POST'])
def save_matrices():
    try:
        # Create saved_matrices directory if it doesn't exist
        save_dir = os.path.join(os.getcwd(), 'saved_matrices')
        os.makedirs(save_dir, exist_ok=True)
            
        # Save downscaled matrices
        for i, matrix in enumerate(downscaled_matrices):
            save_matrix(matrix, f'downscaled_matrix_{i+1}', save_dir)
            
        # Save upscaled matrices
        for i, matrix in enumerate(upscaled_matrices):
            save_matrix(matrix, f'upscaled_matrix_{i+1}', save_dir)
            
        return jsonify({
            'success': True,
            'message': f'Successfully saved {len(downscaled_matrices)} downscaled and {len(upscaled_matrices)} upscaled matrices to {save_dir}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/progress')
def get_progress():
    return jsonify({'progress': session.get('progress', 0)})

@app.route('/save-matrix', methods=['POST'])
def save_single_matrix():
    try:
        global generated_matrix
        if generated_matrix is None:
            return jsonify({'error': 'No matrix to save'}), 400
            
        # Create saved_matrices directory if it doesn't exist
        save_dir = os.path.join(os.getcwd(), 'saved_matrices')
        os.makedirs(save_dir, exist_ok=True)
            
        # Save the generated matrix
        save_matrix(generated_matrix, 'generated_matrix', save_dir)
            
        return jsonify({
            'success': True,
            'message': f'Successfully saved matrix to {save_dir}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    #app.run(host='192.168.142.189', port=5000, debug=True)
    app.run(debug=True)