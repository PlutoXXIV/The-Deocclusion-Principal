from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os
import networkx as nx
import shutil
from scripts.path_planner import get_node_id_from_label, find_path_vector
from scripts.deeplab_model_on import predict_mask
from scripts.graph_builder import build_graph_from_mask   # MST‑enabled version

# Tell Flask to look for index.html in the project root (not in templates/)
app = Flask(__name__, template_folder='.')
CORS(app)

# Ensure required folders exist
os.makedirs('io', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('view_steps', exist_ok=True)

# ── Load graph data from io/ ──────────────────────────────
JSON_PATH = os.path.join('io', 'graph_data.json')
print(f"Looking for graph_data.json at: {JSON_PATH}")

try:
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    print("✅ graph_data.json loaded successfully.")
    G = nx.Graph()
    for u, v, w, wm in data['edges']:
        G.add_edge(u, v, weight=w, weight_m=wm)
    for nid, attrs in data['nodes'].items():
        nid = int(nid)
        G.nodes[nid].update(attrs)
    bc = {int(k): v for k, v in data['bc'].items()}
    resilience = {int(k): v for k, v in data['resilience'].items()}
    image_shape = data.get('image_shape', [1024, 1024])
    edge_bc = nx.edge_betweenness_centrality(G, weight='weight_m')
    edge_bc_str = {f"{u}-{v}": val for (u, v), val in edge_bc.items()}
    graph_loaded = True
except FileNotFoundError:
    G = None
    bc = resilience = edge_bc_str = None
    image_shape = [1024, 1024]
    graph_loaded = False
    print("No graph data found. Upload an image to generate.")
except Exception as e:
    print(f"Error loading graph: {e}")
    graph_loaded = False

# ── Routes ───────────────────────────────────────────────
@app.route('/')
def index():
    # Now index.html is in the root folder (template_folder='.')
    return render_template('index.html')

@app.route('/api/graph', methods=['GET'])
def get_graph():
    if not graph_loaded:
        return jsonify({'error': 'No graph data available'}), 404
    nodes_data = {str(nid): {'r': G.nodes[nid]['r'],
                             'c': G.nodes[nid]['c'],
                             'label': G.nodes[nid]['label'],
                             'type': G.nodes[nid]['type']}
                  for nid in G.nodes()}
    edges_data = [[u, v, d['weight_m']] for u, v, d in G.edges(data=True)]
    return jsonify({
        'nodes': nodes_data,
        'edges': edges_data,
        'bc': {str(k): v for k, v in bc.items()},
        'resilience': {str(k): v for k, v in resilience.items()},
        'edge_bc': edge_bc_str,
        'image_shape': image_shape,
        'node_labels': {str(nid): G.nodes[nid]['label'] for nid in G.nodes()}
    })

@app.route('/api/route', methods=['POST'])
def compute_route():
    if not graph_loaded:
        return jsonify({'error': 'Graph not loaded'}), 500
    req = request.get_json()
    start_label = req.get('start')
    end_label = req.get('end')
    closed_labels = req.get('closed', [])
    try:
        start_id = get_node_id_from_label(G, start_label)
        end_id = get_node_id_from_label(G, end_label)
        closed_ids = [get_node_id_from_label(G, lbl) for lbl in closed_labels]
        path, length, default_length = find_path_vector(
            G, start_id, end_id, closed_nodes=closed_ids, weight='weight_m'
        )
        if path is None:
            return jsonify({'error': 'No path exists with these closures'}), 404
        return jsonify({
            'path': path,
            'length': length,
            'default_length': default_length
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

# ── Upload endpoint ──
@app.route('/upload', methods=['POST'])
def upload_file():
    global G, bc, resilience, edge_bc_str, image_shape, graph_loaded

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 1. Save uploaded satellite image to io/
    sat_io_path = 'io/uploaded_sat.jpg'
    file.save(sat_io_path)
    print(f"Uploaded satellite saved to {sat_io_path}")

    # 2. Run ML model to generate mask
    mask_io_path = 'io/mask.png'
    try:
        predict_mask(sat_io_path, mask_io_path)
        print(f"Mask generated: {mask_io_path}")
    except Exception as e:
        return jsonify({'error': f'Model inference failed: {str(e)}'}), 500

    # 3. Build graph from the mask using the MST‑enabled builder
    try:
        G_new, bc_new, res_new, nodes_new = build_graph_from_mask(
            mask_io_path,
            output_dir='io',                 # JSON goes to io/
            image_dir='view_steps',          # images and Excel go to view_steps/
            metres_per_pixel=10.0,
            spur_min_length=10,
            cluster_radius=8,
            generate_images=True,
            generate_excel=True,
            enable_closing=True,
            closing_radius=8,
            enable_mst_healing=True,
            mst_max_gap_px=20,
            mst_angle_threshold_deg=45,
            mst_tangent_lookback=10
        )
        print("Graph built successfully with MST healing.")
    except Exception as e:
        return jsonify({'error': f'Graph building failed: {str(e)}'}), 500

    # 4. Copy the latest satellite and mask to static/ for serving
    shutil.copy(sat_io_path, 'static/sat.jpg')
    shutil.copy(mask_io_path, 'static/mask.png')
    print("Copied latest images to static/")

    # 5. Update global graph variables directly
    try:
        edge_bc_new = nx.edge_betweenness_centrality(G_new, weight='weight_m')
        edge_bc_str_new = {f"{u}-{v}": val for (u, v), val in edge_bc_new.items()}
        G = G_new
        bc = bc_new
        resilience = res_new
        edge_bc_str = edge_bc_str_new
        image_shape = [1024, 1024]   # you could read from mask if needed
        graph_loaded = True
        print("Global graph variables updated successfully.")
    except Exception as e:
        return jsonify({'error': f'Assigning graph failed: {str(e)}'}), 500

    return jsonify({'success': True, 'message': 'Processing complete. Reloading page...'})

if __name__ == '__main__':
    app.run(debug=True)