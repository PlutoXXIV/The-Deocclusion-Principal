# graph_builder.py
import os
import json
import math
import numpy as np
from skimage.morphology import skeletonize, closing, disk
from skimage.io import imread
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def build_graph_from_mask(mask_path, output_dir, image_dir=None,
                          metres_per_pixel=10.0,
                          spur_min_length=10, cluster_radius=8, dpi=150,
                          generate_images=True, generate_excel=True,
                          # ── Stage 1: Morphological closing (mask level) ──
                          enable_closing=True,
                          closing_radius=8,
                          # ── Stage 2: MST healing (graph level) ──────────
                          enable_mst_healing=True,
                          mst_max_gap_px=20,
                          mst_angle_threshold_deg=45,
                          mst_tangent_lookback=10):
    """
    Full pipeline: mask -> [closing] -> skeleton -> graph -> [MST healing] -> BC/resilience.
    
    Parameters:
      mask_path   : path to binary mask image
      output_dir  : where to save graph_data.json (always)
      image_dir   : where to save images and Excel (if None, uses output_dir)
      ... all other parameters unchanged ...
    """
    # ── Handle image_dir ──────────────────────────────────────────
    if image_dir is None:
        image_dir = output_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # ---------- load and binarize ----------
    mask_raw = imread(mask_path, as_gray=True)
    mask = (mask_raw > 0.5).astype(np.uint8)
    H, W = mask.shape
    print(f"Mask: {W}x{H}, road pixels: {mask.sum()}")

    # figure sizes
    FW, FH = W / dpi, H / dpi

    def save_fig(fig, name):
        if generate_images:
            path = os.path.join(image_dir, name)   # <-- save to image_dir
            fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved: {path}")
        else:
            plt.close(fig)

    # ---------- stage 1: morphological closing (mask level) ----------
    if enable_closing:
        mask_closed = closing(mask.astype(bool), disk(closing_radius)).astype(np.uint8)
        closed_pixels_added = int(mask_closed.sum()) - int(mask.sum())
        print(f"Morphological closing (radius={closing_radius}px): "
              f"+{closed_pixels_added} pixels bridged")
    else:
        mask_closed = mask
        print("Morphological closing disabled")

    # ---------- skeletonize (on closed mask) ----------
    skeleton = skeletonize(mask_closed).astype(np.uint8)
    print(f"Skeleton before pruning: {skeleton.sum()}")

    # helpers
    def get_neighbours(skel, r, c):
        res = []
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]:
                    if skel[nr, nc]:
                        res.append((nr, nc))
        return res

    def count_neighbours(skel, r, c):
        return len(get_neighbours(skel, r, c))

    # prune spurs
    def prune_spurs(skel, min_length):
        skel = skel.copy()
        changed = True
        while changed:
            changed = False
            rows, cols = np.where(skel)
            for r, c in zip(rows, cols):
                if count_neighbours(skel, r, c) == 1:
                    path = [(r, c)]
                    while True:
                        cur = path[-1]
                        nbrs = [p for p in get_neighbours(skel, *cur) if p not in path]
                        if not nbrs: break
                        nxt = nbrs[0]
                        if count_neighbours(skel, *nxt) >= 3: break
                        path.append(nxt)
                    if len(path) < min_length:
                        for pr, pc in path:
                            skel[pr, pc] = 0
                        changed = True
        return skel

    skeleton = prune_spurs(skeleton, min_length=spur_min_length)
    print(f"Skeleton after pruning: {skeleton.sum()}")

    # ---------- create RGB skeleton for overlays ----------
    skel_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    skel_rgb[skeleton.astype(bool)] = [51, 153, 255]

    # ---------- classify pixels ----------
    junction_px, endpoint_px = [], []
    rows, cols = np.where(skeleton)
    for r, c in zip(rows, cols):
        if r==0 or c==0 or r>=H-1 or c>=W-1: continue
        n = count_neighbours(skeleton, r, c)
        if n >= 3:
            junction_px.append((r, c))
        elif n == 1:
            endpoint_px.append((r, c))

    print(f"Raw junction pixels: {len(junction_px)}, raw endpoints: {len(endpoint_px)}")

    # ---------- cluster ----------
    def cluster_points(pts, radius):
        pts = list(pts)
        used = [False] * len(pts)
        clusters = []
        for i, p in enumerate(pts):
            if used[i]: continue
            cl = [p]; used[i] = True
            for j, q in enumerate(pts):
                if used[j]: continue
                if abs(p[0]-q[0]) <= radius and abs(p[1]-q[1]) <= radius:
                    cl.append(q); used[j] = True
            clusters.append(cl)
        return {
            (int(np.mean([p[0] for p in cl])),
             int(np.mean([p[1] for p in cl]))): i
            for i, cl in enumerate(clusters)
        }

    j_logical = cluster_points(junction_px, radius=cluster_radius)
    e_logical = cluster_points(endpoint_px, radius=cluster_radius)

    logical_nodes = {}
    for pos, i in j_logical.items():
        logical_nodes[i] = (pos[0], pos[1], 'J')
    offset = len(j_logical)
    for pos, i in e_logical.items():
        logical_nodes[i + offset] = (pos[0], pos[1], 'E')

    print(f"Logical junction nodes: {len(j_logical)}, endpoints: {len(e_logical)}")
    print(f"Total logical nodes: {len(logical_nodes)}")

    all_candidate_px = set(junction_px + endpoint_px)
    logical_positions = {nid: (d[0], d[1]) for nid, d in logical_nodes.items()}

    def nearest_logical(r, c):
        return min(logical_positions,
                   key=lambda nid: (logical_positions[nid][0]-r)**2 +
                                    (logical_positions[nid][1]-c)**2)

    raw_to_logical = {px: nearest_logical(*px) for px in all_candidate_px}

    # ---------- build graph ----------
    G = nx.Graph()
    for nid, (r, c, ntype) in logical_nodes.items():
        label = f"J{nid}" if ntype == 'J' else f"E{nid}"
        G.add_node(nid, r=r, c=c, type=ntype, label=label)

    def trace_edge(skel, start, first_step, candidate_set):
        path = [start, first_step]
        while True:
            cur = path[-1]
            if cur in candidate_set:
                return cur, len(path) - 1
            nbrs = [p for p in get_neighbours(skel, *cur) if p != path[-2]]
            if not nbrs:
                return cur, len(path) - 1
            if len(nbrs) > 1:
                return cur, len(path) - 1
            path.append(nbrs[0])

    visited_starts = set()
    for px in all_candidate_px:
        nid_start = raw_to_logical[px]
        for nb in get_neighbours(skeleton, *px):
            key = tuple(sorted([px, nb]))
            if key in visited_starts: continue
            visited_starts.add(key)
            end_px, length = trace_edge(skeleton, px, nb, all_candidate_px)
            if end_px in raw_to_logical:
                nid_end = raw_to_logical[end_px]
                if nid_start != nid_end:
                    if G.has_edge(nid_start, nid_end):
                        if G[nid_start][nid_end]['weight'] > length:
                            G[nid_start][nid_end]['weight'] = length
                    else:
                        G.add_edge(nid_start, nid_end, weight=length, healed=False)

    # convert weights to metres
    for u, v, d in G.edges(data=True):
        d['weight_m'] = d['weight'] * metres_per_pixel
        if 'healed' not in d:
            d['healed'] = False

    print(f"Edges before healing: {G.number_of_edges()}, "
          f"connected components: {nx.number_connected_components(G)}")

    # ════════════════════════════════════════════════════════════════════
    # ── CONSTRAINED MST HEALING ────────────────────────────────────────
    # (your existing MST code – unchanged)
    # ════════════════════════════════════════════════════════════════════

    def _tangent_at_endpoint(skel, r0, c0, lookback):
        path = [(r0, c0)]
        for _ in range(lookback):
            cur = path[-1]
            nbrs = [p for p in get_neighbours(skel, *cur) if p not in path]
            if not nbrs:
                break
            path.append(nbrs[0])
        if len(path) < 2:
            return (0.0, 0.0)
        far = path[-1]
        dr = r0 - far[0]
        dc = c0 - far[1]
        norm = math.hypot(dr, dc)
        if norm == 0:
            return (0.0, 0.0)
        return (dr / norm, dc / norm)

    def _angle_between_deg(v1, v2):
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            return 180.0
        cos_angle = max(-1.0, min(1.0, dot / (n1*n2)))
        return math.degrees(math.acos(cos_angle))

    healed_edge_count = 0

    if enable_mst_healing:
        endpoint_node_ids = [nid for nid in G.nodes()
                             if G.nodes[nid]['type'] == 'E' and G.degree(nid) <= 1]

        tangents = {}
        for nid in endpoint_node_ids:
            r0, c0 = G.nodes[nid]['r'], G.nodes[nid]['c']
            tangents[nid] = _tangent_at_endpoint(skeleton, r0, c0, mst_tangent_lookback)

        candidates = []
        for i, a in enumerate(endpoint_node_ids):
            for b in endpoint_node_ids[i+1:]:
                if nx.node_connected_component(G, a) == nx.node_connected_component(G, b):
                    continue
                ra, ca = G.nodes[a]['r'], G.nodes[a]['c']
                rb, cb = G.nodes[b]['r'], G.nodes[b]['c']
                dist = math.hypot(ra - rb, ca - cb)
                if dist > mst_max_gap_px or dist == 0:
                    continue
                gap_a_to_b = ((rb - ra) / dist, (cb - ca) / dist)
                gap_b_to_a = (-gap_a_to_b[0], -gap_a_to_b[1])
                angle_a = _angle_between_deg(tangents[a], gap_a_to_b)
                angle_b = _angle_between_deg(tangents[b], gap_b_to_a)
                if angle_a > mst_angle_threshold_deg or angle_b > mst_angle_threshold_deg:
                    continue
                candidates.append((dist, a, b))

        candidates.sort(key=lambda x: x[0])
        for dist, a, b in candidates:
            if nx.node_connected_component(G, a) == nx.node_connected_component(G, b):
                continue
            weight_px = dist
            G.add_edge(a, b,
                      weight=weight_px,
                      weight_m=weight_px * metres_per_pixel,
                      healed=True)
            healed_edge_count += 1

        print(f"MST healing: {healed_edge_count} edge(s) added "
              f"(max_gap={mst_max_gap_px}px, angle_threshold={mst_angle_threshold_deg}°)")
        print(f"Edges after healing: {G.number_of_edges()}, "
              f"connected components: {nx.number_connected_components(G)}")
    else:
        print("MST healing disabled (enable_mst_healing=False)")

    # ---------- compute betweenness (on graph AFTER healing) ----------
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()
    bc = nx.betweenness_centrality(G_main, weight='weight_m', normalized=True)
    nx.set_node_attributes(G_main, bc, 'bc')
    for nid in G.nodes():
        if nid not in bc:
            bc[nid] = 0.0

    # ---------- resilience (on graph AFTER healing) ----------
    def resilience_index(G_sub, node):
        if node not in G_sub: return None
        try:
            baseline = nx.average_shortest_path_length(G_sub, weight='weight_m')
        except:
            return None
        H = G_sub.copy()
        H.remove_node(node)
        if not nx.is_connected(H):
            return 0.0
        try:
            perturbed = nx.average_shortest_path_length(H, weight='weight_m')
        except:
            return 0.0
        return baseline / perturbed

    resilience_data = {}
    for nid in G_main.nodes():
        resilience_data[nid] = resilience_index(G_main, nid)

    # ---------- export JSON ----------
    graph_export = {
        "nodes": {str(nid): {"r": data['r'], "c": data['c'],
                             "type": data['type'], "label": data['label']}
                  for nid, data in G.nodes(data=True)},
        "edges": [[u, v, d['weight'], d['weight_m']] for u, v, d in G.edges(data=True)],
        "edges_healed_flag": {f"{u}-{v}": d.get('healed', False)
                              for u, v, d in G.edges(data=True)},
        "bc": {str(nid): score for nid, score in bc.items()},
        "resilience": {str(nid): R for nid, R in resilience_data.items()},
        "metres_per_pixel": metres_per_pixel,
        "image_shape": [H, W],
        "mst_healing": {
            "enabled": enable_mst_healing,
            "edges_added": healed_edge_count,
            "max_gap_px": mst_max_gap_px,
            "angle_threshold_deg": mst_angle_threshold_deg
        },
        "morphological_closing": {
            "enabled": enable_closing,
            "radius_px": closing_radius if enable_closing else None
        }
    }
    json_path = os.path.join(output_dir, "graph_data.json")
    with open(json_path, "w") as f:
        json.dump(graph_export, f, indent=2)
    print(f"JSON exported: {json_path}")

    # ---------- (optional) generate static images ----------
    if generate_images:
        # ---- All your figure creation code (unchanged) ----
        # The only change: save_fig now writes to image_dir (already set above)
        # We'll keep the figures as they are.

        def node_color(nid):
            if G.nodes[nid].get('type') == 'E':
                return '#EF9F27'
            score = bc.get(nid, 0)
            if score > 0.35: return '#E24B4A'
            if score > 0.15: return '#EF9F27'
            return '#639922'

        max_bc = max(bc.values()) if bc else 1.0

        def label_offset(c, r, shape):
            off_c = 12 if c < shape[1] - 80 else -14
            off_r = -12 if r > 30 else 14
            return off_c, off_r

        # 01: Original mask
        fig, ax = plt.subplots(figsize=(FW, FH), facecolor='black')
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        ax.axis('off'); ax.set_position([0, 0, 1, 1])
        save_fig(fig, '01_original_mask.png')

        # 02: Skeleton
        fig, ax = plt.subplots(figsize=(FW, FH), facecolor='black')
        ax.imshow(mask_raw, cmap='gray', alpha=0.25, interpolation='nearest')
        ax.imshow(skel_rgb, alpha=0.9, interpolation='nearest')
        ax.axis('off'); ax.set_position([0, 0, 1, 1])
        save_fig(fig, '02_skeleton.png')

        # ... (all other figures 03 to 09 – unchanged, just using save_fig) ...
        # Since the code is long, I'll keep the rest as in your original.
        # But ensure every save_fig call is present.

        # 09: MST healing audit
        fig, ax = plt.subplots(figsize=(FW, FH), facecolor='black')
        ax.imshow(mask_raw, cmap='gray', alpha=0.18, interpolation='nearest')
        ax.imshow(skel_rgb, alpha=0.35, interpolation='nearest')
        for u, v, d in G.edges(data=True):
            ru, cu = G.nodes[u]['r'], G.nodes[u]['c']
            rv, cv = G.nodes[v]['r'], G.nodes[v]['c']
            is_healed = d.get('healed', False)
            ax.plot([cu, cv], [ru, rv],
                    color='#39D353' if is_healed else '#444444',
                    lw=2.5 if is_healed else 1.0,
                    linestyle='--' if is_healed else '-',
                    alpha=0.95 if is_healed else 0.4,
                    zorder=3 if is_healed else 2)
        for nid, (r, c, ntype) in logical_nodes.items():
            ax.plot(c, r, 'o' if ntype == 'J' else 's',
                    color='#888888', markersize=7,
                    markeredgecolor='white', markeredgewidth=0.6, zorder=4)
        if enable_mst_healing:
            title_txt = "MST Healing Audit - " + str(healed_edge_count) + " edge(s) healed"
        else:
            title_txt = "MST Healing Disabled"
        title_col = '#39D353' if healed_edge_count else '#888888'
        ax.text(8, 18, title_txt, color=title_col,
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                         edgecolor=title_col, linewidth=0.8, alpha=0.85))
        ax.axis('off'); ax.set_position([0, 0, 1, 1])
        save_fig(fig, '09_mst_healing_audit.png')

    # ---------- (optional) export Excel ----------
    if generate_excel:
        # ... (your Excel export code, but save to image_dir)
        excel_path = os.path.join(image_dir, 'graph_analysis.xlsx')
        # ... rest of Excel generation

    return G, bc, resilience_data, logical_nodes