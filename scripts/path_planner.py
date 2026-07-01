# path_planner.py
import networkx as nx

def get_node_id_from_label(G, label):
    for nid, data in G.nodes(data=True):
        if data.get('label') == label:
            return nid
    raise ValueError(f"Node label '{label}' not found.")

def find_path_vector(G, start_id, end_id, closed_nodes=None, weight='weight_m'):
    """
    Returns (path, length, default_length) where:
      - path: list of node IDs in order, or None if no path exists
      - length: length of path with closures (metres), or None
      - default_length: length without closures (metres), or None
    """
    if closed_nodes is None:
        closed_nodes = []

    # validate
    for nid in [start_id, end_id] + list(closed_nodes):
        if nid not in G:
            raise ValueError(f"Node {nid} not in graph.")
    if start_id in closed_nodes or end_id in closed_nodes:
        return None, None, None

    # default path (no closures)
    default_length = None
    if nx.has_path(G, start_id, end_id):
        default_path = nx.shortest_path(G, start_id, end_id, weight=weight)
        default_length = nx.path_weight(G, default_path, weight=weight)

    # path with closures
    available = [n for n in G.nodes() if n not in closed_nodes]
    H = G.subgraph(available).copy()
    if not nx.has_path(H, start_id, end_id):
        return None, None, default_length

    path = nx.shortest_path(H, start_id, end_id, weight=weight)
    length = nx.path_weight(H, path, weight=weight)
    return path, length, default_length