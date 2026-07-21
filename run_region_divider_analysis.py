import os
import json
import numpy as np
import pandas as pd
import scipy.ndimage

def find_motifs_dir():
    # Dynamic parent-directory traversal to support Google Colab and local executions
    try:
        from google.colab import drive
        colab_path = '/content/drive/MyDrive/motifs/'
        if os.path.exists(colab_path):
            return colab_path
    except ImportError:
        pass

    current = os.path.abspath(os.getcwd())
    while True:
        candidate = os.path.join(current, 'motifs')
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return './motifs/'

def detect_type1_dividers(grid):
    H, W = grid.shape
    dividers = []

    # Check horizontal lines
    for r in range(H):
        row_cells = grid[r, :]
        unique_colors = np.unique(row_cells)
        if len(unique_colors) == 1 and unique_colors[0] > 0:
            color = unique_colors[0]
            # Contrast: adjacent rows (excluding divider color)
            adj_cells = []
            if r > 0: adj_cells.extend(grid[r-1, :])
            if r < H - 1: adj_cells.extend(grid[r+1, :])
            contrast = np.mean([1 if x != color else 0 for x in adj_cells]) if adj_cells else 1.0

            # Symmetry: middle row
            is_position_symmetric = (r == (H - 1) / 2)

            dividers.append({
                'type': 1,
                'orientation': 'horizontal',
                'index': r,
                'color': int(color),
                'colors': [int(color)],
                'is_edge': r == 0 or r == H - 1,
                'cells_count': W,
                'contrast': float(contrast),
                'is_symmetric': bool(is_position_symmetric),
                'edge_cells_ratio': 2.0 / W if W > 0 else 0.0 # ends of the line are on left/right edges
            })

    # Check vertical lines
    for c in range(W):
        col_cells = grid[:, c]
        unique_colors = np.unique(col_cells)
        if len(unique_colors) == 1 and unique_colors[0] > 0:
            color = unique_colors[0]
            adj_cells = []
            if c > 0: adj_cells.extend(grid[:, c-1])
            if c < W - 1: adj_cells.extend(grid[:, c+1])
            contrast = np.mean([1 if x != color else 0 for x in adj_cells]) if adj_cells else 1.0

            is_position_symmetric = (c == (W - 1) / 2)

            dividers.append({
                'type': 1,
                'orientation': 'vertical',
                'index': c,
                'color': int(color),
                'colors': [int(color)],
                'is_edge': c == 0 or c == W - 1,
                'cells_count': H,
                'contrast': float(contrast),
                'is_symmetric': bool(is_position_symmetric),
                'edge_cells_ratio': 2.0 / H if H > 0 else 0.0 # ends are on top/bottom edges
            })

    return dividers

def detect_type2_dividers(grid):
    H, W = grid.shape
    dividers = []

    # Check horizontal lines
    for r in range(H):
        row_cells = grid[r, :]
        unique_non_zero = np.unique(row_cells[row_cells > 0])
        if len(unique_non_zero) >= 2:
            surrounded = True
            if r > 0 and not np.all(grid[r-1, :] == 0):
                surrounded = False
            if r < H - 1 and not np.all(grid[r+1, :] == 0):
                surrounded = False

            if surrounded:
                is_position_symmetric = (r == (H - 1) / 2)
                # Check line color-sequence symmetry (palindrome)
                is_seq_symmetric = np.all(row_cells == row_cells[::-1])

                dividers.append({
                    'type': 2,
                    'orientation': 'horizontal',
                    'index': r,
                    'color': -1, # Multi-color
                    'colors': [int(x) for x in unique_non_zero],
                    'is_edge': r == 0 or r == H - 1,
                    'cells_count': W,
                    'contrast': 1.0, # Surrounded by 0, high contrast
                    'is_symmetric': bool(is_position_symmetric or is_seq_symmetric),
                    'edge_cells_ratio': 2.0 / W if W > 0 else 0.0
                })

    # Check vertical lines
    for c in range(W):
        col_cells = grid[:, c]
        unique_non_zero = np.unique(col_cells[col_cells > 0])
        if len(unique_non_zero) >= 2:
            surrounded = True
            if c > 0 and not np.all(grid[:, c-1] == 0):
                surrounded = False
            if c < W - 1 and not np.all(grid[:, c+1] == 0):
                surrounded = False

            if surrounded:
                is_position_symmetric = (c == (W - 1) / 2)
                is_seq_symmetric = np.all(col_cells == col_cells[::-1])

                dividers.append({
                    'type': 2,
                    'orientation': 'vertical',
                    'index': c,
                    'color': -1,
                    'colors': [int(x) for x in unique_non_zero],
                    'is_edge': c == 0 or c == W - 1,
                    'cells_count': H,
                    'contrast': 1.0,
                    'is_symmetric': bool(is_position_symmetric or is_seq_symmetric),
                    'edge_cells_ratio': 2.0 / H if H > 0 else 0.0
                })

    return dividers

def detect_type3_dividers(grid):
    H, W = grid.shape
    dividers = []

    unique_colors = np.unique(grid[grid > 0])

    for color in unique_colors:
        mask = (grid == color)
        labeled_comp, num_features = scipy.ndimage.label(mask, structure=np.ones((3, 3)))

        for comp_idx in range(1, num_features + 1):
            comp_mask = (labeled_comp == comp_idx)
            rows, cols = np.where(comp_mask)

            # Check if straight
            unique_rows = np.unique(rows)
            unique_cols = np.unique(cols)
            if len(unique_rows) <= 1 or len(unique_cols) <= 1:
                continue

            # Check if touches at least two boundary edges
            touches_top = np.any(comp_mask[0, :])
            touches_bottom = np.any(comp_mask[H-1, :])
            touches_left = np.any(comp_mask[:, 0])
            touches_right = np.any(comp_mask[:, W-1])

            boundaries = []
            if touches_top: boundaries.append('top')
            if touches_bottom: boundaries.append('bottom')
            if touches_left: boundaries.append('left')
            if touches_right: boundaries.append('right')

            if len(boundaries) >= 2:
                # Check if removing this component divides the remaining grid space (0s)
                bg_mask = (grid == 0) & (~comp_mask)
                labeled_bg, num_bg = scipy.ndimage.label(bg_mask, structure=[[0,1,0],[1,1,1],[0,1,0]])
                bg_sizes = [np.sum(labeled_bg == i) for i in range(1, num_bg + 1)]
                significant_regions = sum(1 for sz in bg_sizes if sz >= 1)

                if significant_regions >= 2:
                    # Calculate contrast: adjacent cells of different colors
                    # Dilate component to find neighbors
                    dilated = scipy.ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3)))
                    neighbor_mask = dilated & (~comp_mask)
                    neighbor_colors = grid[neighbor_mask]
                    contrast = np.mean([1 if x != color else 0 for x in neighbor_colors]) if len(neighbor_colors) > 0 else 1.0

                    # Calculate symmetry of the binary mask
                    flipped_h = comp_mask[:, ::-1]
                    flipped_v = comp_mask[::-1, :]
                    is_sym_h = np.all(comp_mask == flipped_h)
                    is_sym_v = np.all(comp_mask == flipped_v)

                    # Count how many of its cells lie on the boundary edges
                    edge_cells_count = 0
                    edge_cells_count += np.sum(comp_mask[0, :])
                    edge_cells_count += np.sum(comp_mask[H-1, :])
                    edge_cells_count += np.sum(comp_mask[1:H-1, 0])
                    edge_cells_count += np.sum(comp_mask[1:H-1, W-1])
                    total_cells = np.sum(comp_mask)
                    edge_ratio = edge_cells_count / total_cells if total_cells > 0 else 0.0

                    dividers.append({
                        'type': 3,
                        'orientation': 'non-straight',
                        'index': -1,
                        'color': int(color),
                        'colors': [int(color)],
                        'is_edge': False, # Type 3 cannot lie completely on one edge since it's non-straight and divides
                        'cells_count': int(total_cells),
                        'contrast': float(contrast),
                        'is_symmetric': bool(is_sym_h or is_sym_v),
                        'edge_cells_ratio': float(edge_ratio)
                    })

    return dividers

def analyze_grid(grid):
    t1 = detect_type1_dividers(grid)
    t2 = detect_type2_dividers(grid)
    t3 = detect_type3_dividers(grid)
    return t1 + t2 + t3

def run_analysis():
    motifs_dir = find_motifs_dir()
    os.makedirs(motifs_dir, exist_ok=True)

    print("Loading datasets from", motifs_dir)
    with open(os.path.join(motifs_dir, "arc_training_consolidated.json"), "r") as f:
        train_tasks = json.load(f)
    with open(os.path.join(motifs_dir, "arc_evaluation_consolidated.json"), "r") as f:
        eval_tasks = json.load(f)

    all_data = []
    puzzle_data = []

    subsets = {"Training": train_tasks, "Evaluation": eval_tasks}

    # We will collect matrix level details
    for subset_name, tasks in subsets.items():
        for task_id, task in tasks.items():
            puzzle_matrices = []

            # Process train pairs
            for idx, pair in enumerate(task.get("train", [])):
                puzzle_matrices.append(("train_input", idx, np.array(pair["input"])))
                puzzle_matrices.append(("train_output", idx, np.array(pair["output"])))

            # Process test pairs
            for idx, pair in enumerate(task.get("test", [])):
                puzzle_matrices.append(("test_input", idx, np.array(pair["input"])))
                if "output" in pair:
                    puzzle_matrices.append(("test_output", idx, np.array(pair["output"])))

            # Analyze each matrix
            puzzle_has_region = []
            matrix_records = []

            for role, idx, grid in puzzle_matrices:
                divs = analyze_grid(grid)
                has_any = len(divs) > 0
                puzzle_has_region.append(has_any)

                # Height, width, unique color count
                H, W = grid.shape
                unique_colors_count = len(np.unique(grid))

                matrix_record = {
                    "task_id": task_id,
                    "subset": subset_name,
                    "role": role,
                    "index": idx,
                    "height": H,
                    "width": W,
                    "total_cells": H * W,
                    "unique_colors": unique_colors_count,
                    "has_any_divider": has_any,
                    "num_dividers": len(divs),
                    "dividers": divs
                }
                matrix_records.append(matrix_record)

                # Save flat records for each divider
                if not divs:
                    # Save a row indicating no dividers
                    all_data.append({
                        "task_id": task_id,
                        "subset": subset_name,
                        "role": role,
                        "index": idx,
                        "height": H,
                        "width": W,
                        "total_cells": H * W,
                        "unique_colors": unique_colors_count,
                        "has_divider": False,
                        "divider_type": 0,
                        "orientation": "none",
                        "color": -1,
                        "contrast": 0.0,
                        "occupancy": 0.0,
                        "is_symmetric": False,
                        "edge_ratio": 0.0
                    })
                else:
                    for div in divs:
                        all_data.append({
                            "task_id": task_id,
                            "subset": subset_name,
                            "role": role,
                            "index": idx,
                            "height": H,
                            "width": W,
                            "total_cells": H * W,
                            "unique_colors": unique_colors_count,
                            "has_divider": True,
                            "divider_type": div["type"],
                            "orientation": div["orientation"],
                            "color": div["color"],
                            "colors": div["colors"],
                            "contrast": div["contrast"],
                            "occupancy": (div["cells_count"] / (H * W)) * 100.0,
                            "is_symmetric": div["is_symmetric"],
                            "edge_ratio": div["edge_cells_ratio"]
                        })

            # Puzzle level summarization
            # "likelihood that other matrices in the same puzzle also have one region. What is the likelihood that the output pair also has a region."
            # Let's map matrices in this puzzle
            num_matrices = len(matrix_records)
            num_dividers = sum(r["num_dividers"] for r in matrix_records)
            has_t1 = any(any(d["type"] == 1 for d in r["dividers"]) for r in matrix_records)
            has_t2 = any(any(d["type"] == 2 for d in r["dividers"]) for r in matrix_records)
            has_t3 = any(any(d["type"] == 3 for d in r["dividers"]) for r in matrix_records)
            has_any_puz = any(r["has_any_divider"] for r in matrix_records)

            # We want to compute likelihoods *per input matrix with a divider* in this puzzle,
            # then aggregate them globally. Or compute puzzle-wide stats.
            # Let's compute them at puzzle level first and save them!
            puzzle_data.append({
                "task_id": task_id,
                "subset": subset_name,
                "num_matrices": num_matrices,
                "num_dividers": num_dividers,
                "has_type1": has_t1,
                "has_type2": has_t2,
                "has_type3": has_t3,
                "has_any": has_any_puz
            })

    df_matrices = pd.DataFrame(all_data)
    df_puzzles = pd.DataFrame(puzzle_data)

    # Save the matrices and puzzles detailed sheets
    df_matrices.to_csv(os.path.join(motifs_dir, "8-A.divider_matrices.csv"), index=False)
    df_puzzles.to_csv(os.path.join(motifs_dir, "8-B.divider_puzzles.csv"), index=False)

    print(f"Exported {len(df_matrices)} matrix rows to 8-A.divider_matrices.csv")
    print(f"Exported {len(df_puzzles)} puzzle rows to 8-B.divider_puzzles.csv")

    # Let's compute Likelihood probabilities globally and print them!
    # For each input matrix with at least one divider, what is:
    # 1. Likelihood that other matrices in the same puzzle have at least one divider
    # 2. Likelihood that its output pair has at least one divider

    # To do this, let's group df_matrices by task_id and subset
    input_roles = ["train_input", "test_input"]
    output_roles = ["train_output", "test_output"]

    input_dividers_count = 0
    other_matrices_div_sum = 0.0
    other_matrices_total_count = 0
    output_pair_div_count = 0

    # We will also compute this by divider type
    type_stats = {1: {"inputs": 0, "others_sum": 0, "others_tot": 0, "outputs": 0},
                  2: {"inputs": 0, "others_sum": 0, "others_tot": 0, "outputs": 0},
                  3: {"inputs": 0, "others_sum": 0, "others_tot": 0, "outputs": 0}}

    grouped = df_matrices.groupby(["task_id", "subset"])
    for (task_id, subset), group in grouped:
        # Map matrix index to its characteristics
        # Let's build a dictionary of matrices in this puzzle
        puzzle_mats = {}
        for idx, r in group.iterrows():
            key = (r["role"], r["index"])
            puzzle_mats[key] = {
                "has_divider": r["has_divider"],
                "divider_type": r["divider_type"],
                "role": r["role"],
                "index": r["index"]
            }

        for key, mat in puzzle_mats.items():
            role, index = key
            if role in input_roles and mat["has_divider"]:
                input_dividers_count += 1

                # Check other matrices in this puzzle
                others = [m for k, m in puzzle_mats.items() if k != key]
                others_with_div = sum(1 for m in others if m["has_divider"])
                other_matrices_div_sum += others_with_div
                other_matrices_total_count += len(others)

                # Check its corresponding output matrix
                out_role = "train_output" if role == "train_input" else "test_output"
                out_key = (out_role, index)
                if out_key in puzzle_mats:
                    if puzzle_mats[out_key]["has_divider"]:
                        output_pair_div_count += 1

                # Check by specific divider type
                dtype = mat["divider_type"]
                if dtype in type_stats:
                    type_stats[dtype]["inputs"] += 1
                    type_stats[dtype]["others_sum"] += others_with_div
                    type_stats[dtype]["others_tot"] += len(others)
                    if out_key in puzzle_mats and puzzle_mats[out_key]["has_divider"]:
                        type_stats[dtype]["outputs"] += 1

    global_other_likelihood = other_matrices_div_sum / other_matrices_total_count if other_matrices_total_count > 0 else 0.0
    global_output_likelihood = output_pair_div_count / input_dividers_count if input_dividers_count > 0 else 0.0

    print("\n--- GLOBAL LIKELIHOOD METRICS ---")
    print(f"Total input matrices with divider: {input_dividers_count}")
    print(f"Likelihood that other matrices in the same puzzle have a region: {global_other_likelihood:.4f}")
    print(f"Likelihood that its corresponding output matrix has a region: {global_output_likelihood:.4f}")

    # Save these metrics to a small CSV
    likelihood_records = [{
        "type": "Aggregated",
        "inputs_count": input_dividers_count,
        "other_likelihood": global_other_likelihood,
        "output_likelihood": global_output_likelihood
    }]
    for dtype, stats_dict in type_stats.items():
        sub_other_lik = stats_dict["others_sum"] / stats_dict["others_tot"] if stats_dict["others_tot"] > 0 else 0.0
        sub_out_lik = stats_dict["outputs"] / stats_dict["inputs"] if stats_dict["inputs"] > 0 else 0.0
        likelihood_records.append({
            "type": f"Type {dtype}",
            "inputs_count": stats_dict["inputs"],
            "other_likelihood": sub_other_lik,
            "output_likelihood": sub_out_lik
        })
    df_likelihood = pd.DataFrame(likelihood_records)
    df_likelihood.to_csv(os.path.join(motifs_dir, "8-C.likelihood_metrics.csv"), index=False)
    print("Saved likelihood metrics to 8-C.likelihood_metrics.csv")

if __name__ == "__main__":
    run_analysis()
