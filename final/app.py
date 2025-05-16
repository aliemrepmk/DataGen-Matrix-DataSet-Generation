import os
import shutil
import random
import json
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context, redirect, url_for, flash
from werkzeug.utils import secure_filename
import torch
import numpy as np
import scipy.sparse
#from torch_geometric.data import Data, Batch
#from torch_geometric.utils import coalesce
import matplotlib.pyplot as plt
import glob
import time
import tempfile
import subprocess
import argparse
from PIL import Image
import matplotlib.colors as mcolors

# Core matrix operations
from logic.load_matrix import load_matrix
from logic.save_matrix import save_matrix
from logic.optimize import optimize_matrix, cost_function_raw, weights, compute_matrix_properties, compute_scaling_params
from logic.features_matrix import compute_features
from logic.visualize_matrix import visualize_matrices, visualize_heatmaps, visualize_graphs
from logic.graph_visual import visualize_matrix_graph

# Matrix scaling methods
from logic.bilinear import scale_sparse_matrix_bilinear
from logic.dct import scale_sparse_matrix_dct_blockwise
from logic.fourier import scale_sparse_matrix_fourier
from logic.gaussian import scale_sparse_matrix_gaussian
from logic.graph import scale_sparse_matrix_graph
from logic.image import scale_sparse_matrix_image
from logic.lanczos import scale_sparse_matrix_lanczos
from logic.nearest import scale_sparse_matrix_nearest
from logic.wavelet import scale_sparse_matrix_wavelet
from logic.kronecker import scale_sparse_matrix_kronecker

# Problem solvers
from problems.fluid import solve_fluid_symmetric_custom_for_web
from problems.heat import solve_heat_for_web
from problems.solid import solve_solid_for_web
from problems.convection_diffusion import solve_convection_diffusion_problem_for_web
from problems.minimal_surface import solve_minimal_surface_for_web

# Cosine similarity functions
from logic.cosine_similarity import cosine_value, cosine_pattern

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'secret-key'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Matrix scaling methods configuration
SCALING_METHODS = {
    'bilinear': scale_sparse_matrix_bilinear,
    'dct': scale_sparse_matrix_dct_blockwise,
    'fourier': scale_sparse_matrix_fourier,
    'gaussian': scale_sparse_matrix_gaussian,
    'graph': scale_sparse_matrix_graph,
    'image': scale_sparse_matrix_image,
    'lanczos': scale_sparse_matrix_lanczos,
    'nearest': scale_sparse_matrix_nearest,
    'wavelet': scale_sparse_matrix_wavelet,
    'kronecker': scale_sparse_matrix_kronecker
}

# Methods that support upscaling
UPSCALE_METHODS = ['bilinear', 'dct', 'fourier', 'graph', 'image', 'nearest', 'wavelet']

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mtx'}

# Global state for matrix operations
class MatrixState:
    def __init__(self):
        self.downscaled_matrices = []
        self.upscaled_matrices = []
        self.generated_matrix = None

matrix_state = MatrixState()

def clear_saved_matrices():
    """Clear the saved_matrices directory"""
    save_dir = os.path.join(os.getcwd(), 'saved_matrices')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

def clear_graph_files():
    """Clear the graph visualization files"""
    graph_files = [
        os.path.join('static', 'graph_original.png'),
        os.path.join('static', 'graph_generated.png')
    ]
    for file in graph_files:
        if os.path.exists(file):
            os.remove(file)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_file_upload():
    """Common file upload handling logic"""
    if 'matrix_file' not in request.files:
        return None, 'No file uploaded'
        
    file = request.files['matrix_file']
    if file.filename == '':
        return None, 'No file selected'
        
    if not allowed_file(file.filename):
        return None, 'Invalid file type. Only .mtx files are allowed'
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return filepath, None

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/matrix-generator')
def matrix_generator():
    matrix_state.generated_matrix = None
    return render_template('matrix_generator.html')

@app.route('/generate-matrix', methods=['POST'])
def generate_single_matrix():
    try:
        filepath, error = handle_file_upload()
        if error:
            return jsonify({'error': error}), 400
        
        # Save the uploaded matrix for visualization (restore previous behavior)
        shutil.copy(filepath, os.path.join(app.config['UPLOAD_FOLDER'], 'last_uploaded_matrix.mtx'))
        
        try:
            # Clear old visualizations
            clear_saved_matrices()
            clear_graph_files()
            
            # Load and process the matrix
            input_matrix = load_matrix(filepath)
            if input_matrix is None:
                return jsonify({'error': 'Failed to load input matrix'}), 400
            
            # Get parameters
            algorithm = request.form.get('algorithm')
            rows = int(request.form.get('rows'))
            cols = int(request.form.get('cols'))
            optimize = 'optimize' in request.form
            
            # Get algorithm-specific parameters
            params = {}
            if algorithm == 'wavelet':
                wavelet_type_int = int(request.form.get('wavelet_type', 1))
                # Convert integer to wavelet type string (db1, db2, etc.)
                params['wavelet_type'] = f'db{wavelet_type_int}'
            elif algorithm == 'image':
                resize_method = request.form.get('image_resampling_method')
                if resize_method == 'NEAREST':
                    params['resize_method'] = Image.Resampling.NEAREST
                elif resize_method == 'BILINEAR':
                    params['resize_method'] = Image.Resampling.BILINEAR
                elif resize_method == 'BICUBIC':
                    params['resize_method'] = Image.Resampling.BICUBIC
                elif resize_method == 'LANCZOS':
                    params['resize_method'] = Image.Resampling.LANCZOS
                else:  # Default to BOX
                    params['resize_method'] = Image.Resampling.BOX
            elif algorithm == 'lanczos':
                params['a'] = int(request.form.get('kernel_size', 3))
            elif algorithm == 'kronecker':
                params['initiator_size'] = int(request.form.get('initiator_size', 25))
                params['output_path'] = os.path.join(app.config['UPLOAD_FOLDER'], 'last_generated_matrix.mtx')
            
            # Select and apply scaling method
            scaling_func = SCALING_METHODS.get(algorithm)
            if not scaling_func:
                return jsonify({'error': 'Invalid algorithm selected'}), 400
            
            start_time = time.time()
            matrix_state.generated_matrix = scaling_func(input_matrix, rows, **params)
            elapsed_time = time.time() - start_time
            
            # Optimize if requested
            if optimize:
                matrix_state.generated_matrix = optimize_matrix(matrix_state.generated_matrix)
            
            # Calculate property loss (raw differences)
            property_loss = cost_function_raw(input_matrix, matrix_state.generated_matrix, weights)
            
            # Calculate cosine similarities
            cosine_val = cosine_value(input_matrix, matrix_state.generated_matrix)
            cosine_pat = cosine_pattern(input_matrix, matrix_state.generated_matrix)
            
            # Save and visualize
            clear_saved_matrices()
            save_dir = os.path.join(os.getcwd(), 'saved_matrices')
            os.makedirs(save_dir, exist_ok=True)
            save_matrix(matrix_state.generated_matrix, 'generated_matrix', save_dir)
            
            return jsonify({
                'success': True,
                'message': 'Matrix generated and saved successfully',
                'matrix_info': {
                    'shape': matrix_state.generated_matrix.shape,
                    'nnz': matrix_state.generated_matrix.nnz,
                    'density': matrix_state.generated_matrix.nnz / (matrix_state.generated_matrix.shape[0] * matrix_state.generated_matrix.shape[1]),
                    'creation_time_sec': elapsed_time,
                    'property_loss': property_loss,
                    'cosine_value': cosine_val,
                    'cosine_pattern': cosine_pat
                },
                'input_features': compute_features(input_matrix),
                'generated_features': compute_features(matrix_state.generated_matrix)
            })
            
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-dataset')
def dataset_generator():
    matrix_state.downscaled_matrices = []
    matrix_state.upscaled_matrices = []
    clear_saved_matrices()
    return render_template('dataset_generator.html')

@app.route('/generate', methods=['POST'])
def generate_matrices():
    def generate():
        try:
            # Initialize progress
            session['progress'] = 0
            session.modified = True
            
            # Clear saved matrices directory and matrix state before generating new ones
            clear_saved_matrices()
            matrix_state.downscaled_matrices = []
            matrix_state.upscaled_matrices = []
            
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
            
            # Handle file upload
            filepath, error = handle_file_upload()
            if error:
                yield f"data: {json.dumps({'error': error})}\n\n"
                return
            
            try:
                # Load the input matrix
                input_matrix = load_matrix(filepath)
                if input_matrix is None:
                    yield f"data: {json.dumps({'error': 'Failed to load input matrix'})}\n\n"
                    return
                
                # Save the original matrix for dataset generator
                shutil.copy(filepath, os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_original_matrix.mtx'))
                    
                # Generate matrices
                upscale_count = 0
                downscale_count = 0
                method_counts = {name: 0 for name in SCALING_METHODS.keys()}
                
                for i in range(num_matrices):
                    operation = random.randint(0, 1)
                    
                    if operation == 0:  # downscale
                        # Select a random method from all available methods except kronecker
                        available_methods = [m for m in SCALING_METHODS.keys() if m != 'kronecker']
                        method_name = random.choice(available_methods)
                        new_dim = random.randint(min_dim, input_matrix.shape[0])
                        
                        # Skip wavelet and bilinear for downscaling
                        while method_name in ['wavelet', 'bilinear']:
                            method_name = random.choice(available_methods)
                        
                        scaling_func = SCALING_METHODS[method_name]
                        scaled_matrix = scaling_func(input_matrix, new_dim)
                        matrix_state.downscaled_matrices.append((scaled_matrix, method_name))
                        downscale_count += 1
                        method_counts[method_name] += 1
                        
                    else:  # upscale
                        # Select a random method from upscale methods (already excludes kronecker)
                        method_name = random.choice(UPSCALE_METHODS)
                        new_dim = random.randint(input_matrix.shape[0], max_dim)
                        
                        # Skip wavelet and bilinear for upscaling
                        while method_name in ['wavelet', 'bilinear']:
                            method_name = random.choice(UPSCALE_METHODS)
                        
                        scaling_func = SCALING_METHODS[method_name]
                        scaled_matrix = scaling_func(input_matrix, new_dim)
                        matrix_state.upscaled_matrices.append((scaled_matrix, method_name))
                        upscale_count += 1
                        method_counts[method_name] += 1
                    
                    # Send progress update
                    progress = (i + 1) / num_matrices * 100
                    yield f"data: {json.dumps({'progress': progress})}\n\n"
                
                # Send final result
                yield f"data: {json.dumps({'success': True, 'summary': {'total_matrices': num_matrices, 'upscale_count': upscale_count, 'downscale_count': downscale_count, 'method_counts': method_counts}})}\n\n"
                
            finally:
                # Clean up the uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/clear', methods=['POST'])
def clear_matrices():
    matrix_state.downscaled_matrices = []
    matrix_state.upscaled_matrices = []
    session['progress'] = 0
    clear_saved_matrices()
    return jsonify({'success': True, 'message': 'Matrices cleared successfully'})

@app.route('/save_matrices', methods=['POST'])
def save_matrices():
    try:
        # Clear the saved_matrices directory before saving new matrices
        clear_saved_matrices()
        
        save_dir = os.path.join(os.getcwd(), 'saved_matrices')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save all matrices with method names in filenames
        for i, (matrix, method_name) in enumerate(matrix_state.downscaled_matrices):
            save_matrix(matrix, f'downscaled_{method_name}_matrix_{i+1}', save_dir)
        
        for i, (matrix, method_name) in enumerate(matrix_state.upscaled_matrices):
            save_matrix(matrix, f'upscaled_{method_name}_matrix_{i+1}', save_dir)
        
        return jsonify({
            'success': True,
            'message': f'Successfully saved {len(matrix_state.downscaled_matrices)} downscaled and {len(matrix_state.upscaled_matrices)} upscaled matrices to {save_dir}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-matrix', methods=['POST'])
def save_single_matrix():
    try:
        if matrix_state.generated_matrix is None:
            return jsonify({'error': 'No matrix to save'}), 400
        
        save_dir = os.path.join(os.getcwd(), 'saved_matrices')
        os.makedirs(save_dir, exist_ok=True)
        save_matrix(matrix_state.generated_matrix, 'generated_matrix', save_dir)
        
        return jsonify({
            'success': True,
            'message': f'Matrix saved successfully to {save_dir}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/progress')
def get_progress():
    return jsonify({'progress': session.get('progress', 0)})

@app.route('/neural-network')
def neural_network():
    # Check if a pretrained matrix exists
    pretrained_matrix_path = os.path.join(os.getcwd(), 'saved_matrices', 'pretrained_generated_matrix.mtx')
    matrix_exists = os.path.exists(pretrained_matrix_path)
    
    # Get matrix info if it exists
    matrix_info = None
    if matrix_exists:
        try:
            matrix = load_matrix(pretrained_matrix_path)
            if matrix is not None:
                matrix_info = {
                    'shape': matrix.shape,
                    'nnz': matrix.nnz,
                    'density': (matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100  # as percentage
                }
        except Exception as e:
            print(f"Error loading matrix info: {e}")
    
    return render_template('neural_network.html', matrix_exists=matrix_exists, matrix_info=matrix_info)

@app.route('/use-pretrained-model', methods=['POST'])
def use_pretrained_model():
    try:
        # Load the pre-trained model
        model_path = os.path.join('network', 'autoregressive_graph_generator.pth')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Import the model class
        from network.autoregressiveGraphGenerator import AutoregressiveGraphGenerator
        from network.autoregressiveGraphGenerator import GraphStateUpdaterGNN

        NODE_EMBEDDING_DIM = 64 # Dimensionality of the initial learnable node features
        GNN_HIDDEN_DIM = 128    # Number of features in the hidden GCN layers
        GRAPH_OUTPUT_DIM = 256  # Output dimension of GraphStateUpdaterGNN, input to prediction head
        MAX_NODES = 499

        state_updater_gnn = GraphStateUpdaterGNN(
            node_embedding_dim=NODE_EMBEDDING_DIM,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            output_dim=GRAPH_OUTPUT_DIM
        )

        model = AutoregressiveGraphGenerator(
            graph_updater=state_updater_gnn,
            graph_embedding_dim=GRAPH_OUTPUT_DIM,
            max_nodes_to_generate=MAX_NODES
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # Import necessary class
        from network.generate_graph_pretrained import generate_graph, pyg_graph_to_scipy_sparse

        NUM_NODES_TO_GENERATE = 50
        SAMPLING_THRESHOLD = 0.3
        generated_sample_graph = generate_graph(
            model=model,
            num_nodes_to_generate=NUM_NODES_TO_GENERATE,
            device=device,
            sampling_threshold=SAMPLING_THRESHOLD
        )
        generated_sparse_matrix = pyg_graph_to_scipy_sparse(generated_sample_graph)
        
        # Compute some basic properties of the matrix
        matrix_size = generated_sparse_matrix.shape[0]
        matrix_nnz = generated_sparse_matrix.nnz
        matrix_density = matrix_nnz / (matrix_size * matrix_size) * 100  # as percentage
        
        # Ensure the saved_matrices directory exists
        save_dir = os.path.join(os.getcwd(), 'saved_matrices')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the generated matrix
        save_matrix(generated_sparse_matrix, 'pretrained_generated_matrix', save_dir)
        
        # Create a more informative message
        flash(f'Matrix successfully generated! Size: {matrix_size}×{matrix_size}, Non-zeros: {matrix_nnz}, Density: {matrix_density:.2f}%. You can visualize it below.')
        
    except Exception as e:
        flash(f'Error using pre-trained model: {e}')
    
    return redirect(url_for('neural_network'))

@app.route('/generate-graphs', methods=['GET'])
def generate_graphs():
    try:
        if matrix_state.generated_matrix is None:
            return jsonify({'error': 'No matrix available for visualization'}), 400
        
        input_matrix = load_matrix(os.path.join(app.config['UPLOAD_FOLDER'], 'last_uploaded_matrix.mtx'))
        if input_matrix is None:
            return jsonify({'error': 'Failed to load input matrix'}), 400
        
        # Generate graph visualizations
        orig_graph_path = os.path.join('static', 'graph_original.png')
        gen_graph_path = os.path.join('static', 'graph_generated.png')
        
        try:
            visualize_matrix_graph(input_matrix, orig_graph_path)
            visualize_matrix_graph(matrix_state.generated_matrix, gen_graph_path)
            return jsonify({
                'success': True,
                'message': 'Graphs generated successfully'
            })
        except Exception as e:
            return jsonify({'error': f'Graph visualization failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize-matrix', methods=['GET'])
def visualize_matrix():
    try:
        if matrix_state.generated_matrix is None:
            return jsonify({'error': 'No matrix available for visualization'}), 400
        
        input_matrix = load_matrix(os.path.join(app.config['UPLOAD_FOLDER'], 'last_uploaded_matrix.mtx'))
        if input_matrix is None:
            return jsonify({'error': 'Failed to load input matrix'}), 400
        
        visualization_type = request.args.get('type')
        if visualization_type == "spy":
            filepath = visualize_matrices(input_matrix, matrix_state.generated_matrix)
        elif visualization_type == "heatmap":
            filepath = visualize_heatmaps(input_matrix, matrix_state.generated_matrix)
        elif visualization_type == "graph":
            filepath = visualize_graphs(input_matrix, matrix_state.generated_matrix)
        else:
            return jsonify({'error': 'Invalid visualization type'}), 400
        
        # Convert the file path to a static-relative path for the frontend
        static_folder = os.path.abspath('static')
        abs_filepath = os.path.abspath(filepath)
        if abs_filepath.startswith(static_folder):
            image_path = '/static/' + os.path.relpath(abs_filepath, static_folder).replace('\\', '/').replace('\\', '/')
        else:
            image_path = filepath  # fallback
        
        return jsonify({
            'success': True,
            'image_path': image_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulation', methods=['GET'])
def simulation():
    """Render the simulation page."""
    # You can pass default values or options to the template if needed
    problem_types = [
        {"value": "fluid_symmetric_custom", "label": "Fluid (Symmetric Custom Params)"},
        {"value": "heat", "label": "Heat Transfer"},
        {"value": "solid", "label": "Solid Mechanics (Elasticity)"},
        {"value": "convection_diffusion", "label": "Convection-Diffusion"},
        {"value": "minimal_surface", "label": "Minimal Surface"},
    ]
    mesh_options = ["circle", "rectangle", "lshaped", "symmetric", "sqsymmetric", "tensor", "random"] # Updated mesh options

    return render_template('simulation.html', problem_types=problem_types, mesh_options=mesh_options)

@app.route('/solve', methods=['POST'])
def solve_problem():
    """Solve the selected problem and return visualizations."""
    try:
        problem_choice = request.form.get('problem_type')
        plots = []
        solver_details = {} # To store some info about what was run

        # --- Fluid Problems ---
        if problem_choice == 'fluid_symmetric_custom':
            solver_details['name'] = "Fluid (Symmetric Custom)"
            mesh_type = request.form.get('fluid_sym_custom_mesh_type', 'random')
            mesh_res_hint = int(request.form.get('fluid_sym_custom_mesh_res', 20))
            viscosity = float(request.form.get('fluid_sym_custom_viscosity', 10.0))
            reg_param = float(request.form.get('fluid_sym_custom_reg_param', 3e-6))
            _, _, _, _, _, _, _, plots = solve_fluid_symmetric_custom_for_web(
                mesh_type, mesh_res_hint, viscosity, reg_param
            )

        # --- Heat Transfer Problem ---
        elif problem_choice == 'heat':
            solver_details['name'] = "Heat Transfer"
            mesh_type = request.form.get('heat_mesh_type', 'random')
            mesh_res_hint = int(request.form.get('heat_mesh_res', 20))
            kappa = float(request.form.get('heat_kappa', np.random.uniform(0.1, 2.0)))
            t_end = float(request.form.get('heat_t_end', 0.5))
            dt = float(request.form.get('heat_dt', 0.01))
            theta = float(request.form.get('heat_theta', 0.7))
            _, _, _, plots = solve_heat_for_web(mesh_type, mesh_res_hint, kappa, t_end, dt, theta)

        # --- Solid Mechanics Problem ---
        elif problem_choice == 'solid':
            solver_details['name'] = "Solid Mechanics"
            mesh_type = request.form.get('solid_mesh_type', 'random')
            mesh_res_hint = int(request.form.get('solid_mesh_res', 20))
            E_val = float(request.form.get('solid_E', np.random.uniform(1e9, 5e10)))
            nu_val = float(request.form.get('solid_nu', np.random.uniform(0.25, 0.33)))
            traction = float(request.form.get('solid_traction', 1e6))
            deformed_scale = float(request.form.get('solid_deformed_scale', 1e3))
            _, _, _, plots = solve_solid_for_web(mesh_type, mesh_res_hint, E_val, nu_val, traction, deformed_scale)

        # --- Convection-Diffusion Problem ---
        elif problem_choice == 'convection_diffusion':
            solver_details['name'] = "Convection-Diffusion"
            mesh_type = request.form.get('cd_mesh_type', 'random')
            mesh_res_hint = int(request.form.get('cd_mesh_res', 20))
            epsilon = float(request.form.get('cd_epsilon', np.random.uniform(1e-3, 5e-2)))
            b_field_x = float(request.form.get('cd_b_field_x', np.random.uniform(-2.0, 2.0)))
            b_field_y = float(request.form.get('cd_b_field_y', np.random.uniform(-2.0, 2.0)))
            f_value = float(request.form.get('cd_f_value', np.random.uniform(0.1, 2.0)))
            _, _, _, plots = solve_convection_diffusion_problem_for_web(
                mesh_type, mesh_res_hint, epsilon, (b_field_x, b_field_y), f_value
            )

        # --- Minimal Surface Problem ---
        elif problem_choice == 'minimal_surface':
            solver_details['name'] = "Minimal Surface"
            mesh_type = request.form.get('ms_mesh_type', 'random')
            mesh_res_hint = int(request.form.get('ms_mesh_res', 20))
            max_iters = int(request.form.get('ms_max_iters', 30))
            tolerance = float(request.form.get('ms_tolerance', 1e-7))
            relaxation = float(request.form.get('ms_relaxation', 0.7))
            _, _, _, plots = solve_minimal_surface_for_web(
                mesh_type, mesh_res_hint, max_iters, tolerance, relaxation
            )

        else:
            # Handle unknown problem choice or redirect
            return redirect(url_for('simulation'))

        return render_template('simulation_results.html', plots=plots, solver_details=solver_details)

    except Exception as e:
        # Basic error handling: print to console and show an error page
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return render_template('simulation_error.html', error_message=str(e))

@app.route('/visualize-saved-matrices', methods=['GET'])
def visualize_saved_matrices():
    try:
        # Find one downscaled and one upscaled matrix
        saved_dir = os.path.join(os.getcwd(), 'saved_matrices')
        down_files = sorted(glob.glob(os.path.join(saved_dir, 'downscaled_*.mtx')))
        up_files = sorted(glob.glob(os.path.join(saved_dir, 'upscaled_*.mtx')))
        
        if not down_files or not up_files:
            return jsonify({'error': 'No downscaled or upscaled matrices found.'}), 400
            
        # Get the method names from the filenames
        down_method = os.path.basename(down_files[0]).split('_')[1]
        up_method = os.path.basename(up_files[0]).split('_')[1]
        
        # Load the matrices
        down_matrix = load_matrix(down_files[0])
        up_matrix = load_matrix(up_files[0])
        
        # Get the original matrix from the dataset generator's upload
        orig_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_original_matrix.mtx')
        if not os.path.exists(orig_path):
            return jsonify({'error': 'Original matrix not found.'}), 400
            
        orig_matrix = load_matrix(orig_path)
        
        if down_matrix is None or up_matrix is None or orig_matrix is None:
            return jsonify({'error': 'Failed to load one of the matrices.'}), 400
            
        # Visualize all three side by side
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].spy(orig_matrix, markersize=1)
        axes[0].set_title("Original Matrix", fontsize=14)
        
        axes[1].spy(down_matrix, markersize=1)
        axes[1].set_title(f"Downscaled Matrix ({down_method})", fontsize=14)
        
        axes[2].spy(up_matrix, markersize=1)
        axes[2].set_title(f"Upscaled Matrix ({up_method})", fontsize=14)
        
        plt.tight_layout()
        
        # Save the plot as an image
        from logic.visualize_matrix import TEMP_VISUALIZATION_FOLDER
        filepath = os.path.join(TEMP_VISUALIZATION_FOLDER, 'spy_plot_three.png')
        plt.savefig(filepath)
        plt.close(fig)
        
        # Convert to static-relative path
        static_folder = os.path.abspath('static')
        abs_filepath = os.path.abspath(filepath)
        if abs_filepath.startswith(static_folder):
            image_path = '/static/' + os.path.relpath(abs_filepath, static_folder).replace('\\', '/')
        else:
            image_path = filepath
            
        return jsonify({'success': True, 'image_path': image_path})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/train-network', methods=['GET', 'POST'])
def train_network():
    if request.method == 'POST':
        # Handle uploaded files and parameters
        epochs = int(request.form.get('epochs', 20))
        batch_size = int(request.form.get('batch_size', 32))
        learning_rate = float(request.form.get('learning_rate', 0.001))
        files = request.files.getlist('dataset_folder')
        if not files:
            flash('No dataset files uploaded.', 'danger')
            return redirect(url_for('train_network'))
        # Save uploaded files to a temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            for f in files:
                if f.filename.endswith('.mtx'):
                    base = os.path.basename(f.filename)
                    f.save(os.path.join(temp_dir, base))
            print('Files in temp_dir:', os.listdir(temp_dir))
            # Call neural.py as a subprocess
            # (Assume neural.py can take args: --data_dir, --epochs, --batch_size, --learning_rate)
            cmd = [
                'python3', 'network/neural.py',
                '--data_dir', temp_dir,
                '--epochs', str(epochs),
                '--batch_size', str(batch_size),
                '--learning_rate', str(learning_rate)
            ]
            try:
                subprocess.run(cmd, check=True)
                flash('Training completed and model saved successfully!', 'success')
            except subprocess.CalledProcessError as e:
                flash(f'Training failed: {e}', 'danger')
        return redirect(url_for('neural_network'))
    return render_template('train_network.html')

@app.route('/visualize-neural-matrix', methods=['GET'])
def visualize_neural_matrix():
    try:
        # Check if pretrained generated matrix file exists
        pretrained_matrix_path = os.path.join(os.getcwd(), 'saved_matrices', 'pretrained_generated_matrix.mtx')
        if not os.path.exists(pretrained_matrix_path):
            return jsonify({'error': 'No generated matrix available. Please generate a matrix first.'}), 400
        
        # Load the pretrained matrix
        pretrained_matrix = load_matrix(pretrained_matrix_path)
        if pretrained_matrix is None:
            return jsonify({'error': 'Failed to load generated matrix.'}), 400
        
        visualization_type = request.args.get('type')
        
        if visualization_type == "spy":
            # For spy plot, we just need to visualize the generated matrix (not comparing)
            # Create a custom spy plot for a single matrix
            import matplotlib.pyplot as plt
            from logic.visualize_matrix import TEMP_VISUALIZATION_FOLDER
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.spy(pretrained_matrix, markersize=1.5, color='blue')
            ax.set_title(f"Generated Matrix (Pretrained Model)", fontsize=14)
            
            filepath = os.path.join(TEMP_VISUALIZATION_FOLDER, 'neural_spy_plot.png')
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close(fig)
            
        elif visualization_type == "heatmap":
            # For heatmap, use similar approach but with heatmap visualization
            import matplotlib.pyplot as plt
            import numpy as np
            from logic.visualize_matrix import TEMP_VISUALIZATION_FOLDER
            
            # Get a dense version, but limit size to avoid memory issues
            if pretrained_matrix.shape[0] > 200:
                # For large matrices, sample a submatrix for visualization
                max_size = 200
                submatrix = pretrained_matrix[:max_size, :max_size].toarray()
            else:
                submatrix = pretrained_matrix.toarray()
            
            # Use the same approach as in visualize_heatmaps
            def limits(mat):
                nz = mat[mat != 0]
                return (nz.min(), nz.max()) if nz.size else (0, 1)
            
            vmin, vmax = limits(submatrix)
            submatrix_masked = np.ma.masked_where(submatrix == 0, submatrix)
            
            # Use the same colormap as in visualize_heatmaps
            cmap = plt.cm.viridis.copy()
            cmap.set_bad(color="white")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(submatrix_masked, cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax), interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Generated Matrix Heatmap (Pretrained Model)", fontsize=14)
            
            filepath = os.path.join(TEMP_VISUALIZATION_FOLDER, 'neural_heatmap_plot.png')
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close(fig)
            
        elif visualization_type == "graph":
            # Use the graph visualization function for the matrix
            filepath = os.path.join('static', 'neural_graph.png')
            visualize_matrix_graph(pretrained_matrix, filepath)
            
        else:
            return jsonify({'error': 'Invalid visualization type'}), 400
        
        # Convert to static-relative path
        static_folder = os.path.abspath('static')
        abs_filepath = os.path.abspath(filepath)
        if abs_filepath.startswith(static_folder):
            image_path = '/static/' + os.path.relpath(abs_filepath, static_folder).replace('\\', '/').replace('\\', '/')
        else:
            image_path = filepath
        
        return jsonify({
            'success': True,
            'image_path': image_path
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)