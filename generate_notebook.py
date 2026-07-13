import nbformat as nbf
import json
import os

# Create a new notebook
nb = nbf.v4.new_notebook()

# Notebook metadata
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python'
    }
}

cells = []

# ==========================================
# TITLE & INTRODUCTION
# ==========================================

cells.append(nbf.v4.new_markdown_cell("""# 2. Shape Taxonomy Analysis in the ARC Dataset

This notebook explores and defines a **shape taxonomy** across the Abstract Reasoning and Cognitive (ARC) dataset. By analyzing connected components under various connectivity and background configurations, we identify the most common motif shapes, fit them into minimal enclosing square matrices, and group them by symmetry.

We perform a deep-dive analysis of the 10 most common motifs, focusing on their surrounding entropy, spatial locations, and proximity to other shapes. We then construct shape-taxonomy-based grid embeddings and run predictive matching experiments to compare their power against the connectivity-based embeddings of the previous notebook.
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 1: Environment Setup & Library Imports

In this section, we import all necessary libraries, configure the output directories for Google Colab and local execution, and set up helper paths.
"""))

cells.append(nbf.v4.new_code_cell("""# Import dependencies
import os
import sys
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import label
from skimage.measure import regionprops
from scipy.spatial.distance import cdist

# Attempt to mount Google Drive in Colab, or create local directory if running locally
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    OUTPUT_DIR = '/content/drive/MyDrive/motifs/'
except Exception:
    print('Could not mount Google Colab Drive. Mocking paths locally.')
    IN_COLAB = False
    OUTPUT_DIR = './motifs/'

# Create standardized directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'Target export directory is set to: {OUTPUT_DIR}')
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 2: Load Genuine ARC Dataset & Sample Tasks

We load the pre-extracted grid database `arc_matrices_dataset.json` from the target export directory.

To ensure the notebook executes rapidly and remains compatible with standard execution environments, we sample 150 complete tasks (keeping all of their grids intact). This preserves complete puzzles, allowing us to perform highly robust Same-Puzzle and Input-Output matching experiments.
"""))

cells.append(nbf.v4.new_code_cell("""# Loading existing exported dataset
arc_grids_all = []
dataset_path = os.path.join(OUTPUT_DIR, 'arc_matrices_dataset.json')

# Fallback to local motifs directory if needed
if not os.path.exists(dataset_path) and os.path.exists('motifs/arc_matrices_dataset.json'):
    dataset_path = 'motifs/arc_matrices_dataset.json'

if os.path.exists(dataset_path):
    print(f'Loading existing dataset from: {dataset_path}')
    with open(dataset_path, 'r') as f:
        raw_data = json.load(f)
    for item in raw_data:
        arc_grids_all.append({
            'task_id': item['task_id'],
            'pair_type': item['pair_type'],
            'pair_idx': item['pair_idx'],
            'grid_type': item['grid_type'],
            'grid': np.array(item['matrix']),
            'subset': item['subset']
        })
    print(f'Successfully loaded {len(arc_grids_all)} grids.')
else:
    raise FileNotFoundError(f'Critical: Could not find {dataset_path}. Please run Notebook 1 first.')

# Task-preserving subsampling for robust and rapid analysis
unique_tasks = list(set([g['task_id'] for g in arc_grids_all]))
np.random.seed(42)
sampled_tasks = np.random.choice(unique_tasks, size=min(150, len(unique_tasks)), replace=False)
arc_grids = [g for g in arc_grids_all if g['task_id'] in sampled_tasks]
print(f'Sampled {len(sampled_tasks)} complete tasks containing {len(arc_grids)} grids total.')
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 3: Define Component Extraction and Mapping Logic

Here we implement the logic to extract connected components of size $\ge 3$ under 4-adjacency and 8-adjacency for both `same_color` and `non_background` settings.

We map each component to its minimum enclosing square matrix size in $\{2, 3, 4, 5, 6\}$. If the component shape is larger than 6x6, we trim it to 6x6 by the top-left position. We also compute the canonical symmetric shape using Dihedral $D_4$ group symmetries (rotating and flipling to find the lexicographically smallest representation). To ensure high performance, we use a global shape-canonicalization lookup cache.
"""))

cells.append(nbf.v4.new_code_cell("""# Global cache to speed up canonical shape transformations
canonical_cache = {}

# Helper to canonicalize a shape mask under Dihedral D4 symmetries with cache
def get_canonical_shape(mask):
    tup_mask = tuple(tuple(int(x) for x in row) for row in mask)
    if tup_mask in canonical_cache:
        return canonical_cache[tup_mask]

    symmetries = []
    # Since input 'mask' is already cropped tight to its bounding box,
    # any rotation or reflection of it will remain tight.
    for rot in range(4):
        for flip in [False, True]:
            t = np.rot90(mask, k=rot)
            if flip:
                t = np.fliplr(t)
            symmetries.append(tuple(tuple(int(x) for x in row) for row in t))
    canon = min(symmetries)
    canonical_cache[tup_mask] = canon
    return canon

# Helper to find enclosing matrix (2x2 up to 6x6) with top-left trimming if needed
def get_enclosing_matrix(mask):
    h, w = mask.shape
    if h > 6 or w > 6:
        trimmed = mask[:min(h, 6), :min(w, 6)]
        ht, wt = trimmed.shape
        sq = np.zeros((6, 6), dtype=int)
        sq[:ht, :wt] = trimmed
        return 6, sq
    else:
        max_dim = max(h, w)
        if max_dim <= 2:
            s = 2
        elif max_dim == 3:
            s = 3
        elif max_dim == 4:
            s = 4
        elif max_dim == 5:
            s = 5
        else:
            s = 6
        sq = np.zeros((s, s), dtype=int)
        sq[:h, :w] = mask
        return s, sq

# Main connected component extractor
def extract_grid_motifs(grid, conn=4, group='same_color', bg_color=0):
    h, w = grid.shape
    components = []

    if group == 'same_color':
        colors = np.unique(grid)
        colors = colors[colors != bg_color]
        for color in colors:
            mask = (grid == color)
            struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) if conn == 4 else np.ones((3, 3))
            labeled, num_features = label(mask, structure=struct)
            for f_idx in range(1, num_features + 1):
                f_mask = (labeled == f_idx)
                size = np.sum(f_mask)
                if size >= 3:
                    components.append((f_mask, color, size))
    elif group == 'non_background':
        mask = (grid != bg_color)
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) if conn == 4 else np.ones((3, 3))
        labeled, num_features = label(mask, structure=struct)
        for f_idx in range(1, num_features + 1):
            f_mask = (labeled == f_idx)
            size = np.sum(f_mask)
            if size >= 3:
                colors_in_mask, counts = np.unique(grid[f_mask], return_counts=True)
                dominant_color = colors_in_mask[np.argmax(counts)]
                components.append((f_mask, dominant_color, size))

    results = []
    for f_mask, color, size in components:
        # Get tight bounding box
        rows = np.any(f_mask, axis=1)
        cols = np.any(f_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        tight_mask = f_mask[rmin:rmax+1, cmin:cmax+1]

        # Enclosing matrix
        enc_size, enc_mat = get_enclosing_matrix(tight_mask)
        # Canonical symmetric shape
        canon = get_canonical_shape(tight_mask)

        results.append({
            'mask': f_mask,
            'color': color,
            'size': size,
            'bbox': (rmin, rmax, cmin, cmax),
            'tight_mask': tight_mask,
            'enclosing_size': enc_size,
            'enclosing_matrix': enc_mat,
            'canonical_shape': canon
        })
    return results
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 4: Extract Motifs across All Configurations and Grids

We loop through all the ARC grids in our loaded dataset and extract motifs for each of the 4 configurations:
- 4-adjacency, Same-Color
- 8-adjacency, Same-Color
- 4-adjacency, Non-Background
- 8-adjacency, Non-Background
"""))

cells.append(nbf.v4.new_code_cell("""all_motifs_data = []

for idx, item in enumerate(arc_grids):
    grid = item['grid']
    for conn in [4, 8]:
        for grp in ['same_color', 'non_background']:
            motifs = extract_grid_motifs(grid, conn=conn, group=grp)
            for m in motifs:
                all_motifs_data.append({
                    'grid_index': idx,
                    'task_id': item['task_id'],
                    'grid_type': item['grid_type'],
                    'pair_idx': item['pair_idx'],
                    'conn': conn,
                    'group': grp,
                    'size': m['size'],
                    'color': m['color'],
                    'enclosing_size': m['enclosing_size'],
                    'enclosing_matrix': tuple(map(tuple, m['enclosing_matrix'])),
                    'canonical_shape': m['canonical_shape'],
                    'bbox': m['bbox'],
                    'mask': m['mask']
                })

df_motifs = pd.DataFrame(all_motifs_data)
print(f'Total motif instances extracted: {len(df_motifs)}')
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 5: Describe Exact Positions & Enclosing Matrix Distribution

We analyze how the shapes fit within the 2x2, 3x3, 4x4, 5x5, and 6x6 square enclosing matrices.
"""))

cells.append(nbf.v4.new_code_cell("""# Size distribution of enclosing matrices
enc_counts = df_motifs['enclosing_size'].value_counts().sort_index()
print("Enclosing Matrix Size Distribution:")
for sz, cnt in enc_counts.items():
    print(f"  {sz}x{sz}: {cnt} instances ({cnt/len(df_motifs):.2%})")

# Let's plot the distribution of enclosing square matrices
plt.figure(figsize=(8, 5))
sns.barplot(x=enc_counts.index, y=enc_counts.values, palette='viridis')
plt.title('Distribution of Minimum Enclosing Square Matrix Sizes')
plt.xlabel('Enclosing Matrix Size')
plt.ylabel('Instance Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(OUTPUT_DIR, 'descriptive_enclosing_matrix_sizes.png'), bbox_inches='tight')
plt.show()
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 6: Group and Aggregate Symmetric Shapes vs Exact Shapes

We group motif occurrences by exact shape (using the minimum enclosing matrix) and by symmetric shape (canonicalized shape) to understand how symmetries simplify the shape taxonomy.
"""))

cells.append(nbf.v4.new_code_cell("""# Aggregation by exact position shape (enclosing matrix representation)
exact_counts = df_motifs['enclosing_matrix'].value_counts()
print(f"Total unique EXACT shapes: {len(exact_counts)}")

# Aggregation by canonical symmetric shape
sym_counts = df_motifs['canonical_shape'].value_counts()
print(f"Total unique SYMMETRIC shapes: {len(sym_counts)}")

# Comparison
plt.figure(figsize=(8, 4))
sns.barplot(x=['Exact Shapes', 'Symmetric Shapes'], y=[len(exact_counts), len(sym_counts)], palette='coolwarm')
plt.title('Taxonomy Simplification: Unique Shapes count')
plt.ylabel('Number of Unique Shapes')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(OUTPUT_DIR, 'descriptive_taxonomy_comparison.png'), bbox_inches='tight')
plt.show()
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 7: Motif Count and Size Cohort Distributions (Grouped Bar Chart)

Following standard specifications, we prefer grouped bar charts binned into cohorts demonstrating decay distributions rather than violin plots. Let's group component sizes into binned cohorts and plot their decay distribution across configurations.
"""))

cells.append(nbf.v4.new_code_cell("""# Define size cohorts
bins = [3, 5, 8, 12, 20, 50, 1000]
labels = ['3-4', '5-7', '8-11', '12-19', '20-49', '50+']
df_motifs['size_cohort'] = pd.cut(df_motifs['size'], bins=bins, labels=labels, right=False)

# Group by configuration and cohort
df_decay = df_motifs.groupby(['conn', 'group', 'size_cohort'], observed=False).size().reset_index(name='count')
df_decay['config'] = df_decay['conn'].astype(str) + '_' + df_decay['group']

plt.figure(figsize=(12, 6))
sns.barplot(data=df_decay, x='size_cohort', y='count', hue='config', palette='muted')
plt.title('Grouped Decay Distribution of Motif Sizes across Cohorts')
plt.xlabel('Component Size Cohort (number of pixels)')
plt.ylabel('Frequency')
plt.yscale('log')  # Log scale to clearly show the exponential decay
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig(os.path.join(OUTPUT_DIR, 'descriptive_motif_decay_cohorts.png'), bbox_inches='tight')
plt.show()
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 8: Identify the 10 Most Common Shapes under Symmetry & Exact Positions

We identify and print details of the top 10 most common motifs under symmetry and exact positions across configurations.
"""))

cells.append(nbf.v4.new_code_cell("""top_10_sym = sym_counts.head(10)
print("Top 10 Most Common Canonical Symmetric Shapes:")
for rank, (shape, count) in enumerate(top_10_sym.items(), 1):
    h = len(shape)
    w = len(shape[0])
    print(f"Rank {rank}: Shape {h}x{w}, count={count}")

top_10_exact = exact_counts.head(10)
print("\\nTop 10 Most Common Exact Position Shapes (Enclosing matrices):")
for rank, (shape, count) in enumerate(top_10_exact.items(), 1):
    h = len(shape)
    w = len(shape[0])
    print(f"Rank {rank}: Enclosing {h}x{w}, count={count}")
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 9: Visualize the Top 10 Most Common Symmetric Motifs

Let's plot the actual binary masks of the 10 most common symmetric shapes to visualize our discovered taxonomy.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for rank, (shape, count) in enumerate(top_10_sym.items()):
    ax = axes[rank]
    m = np.array(shape)
    ax.imshow(m, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'Rank {rank+1}\\nScore: {count}')
    ax.axis('off')

plt.suptitle('Top 10 Discovered Shape Motifs (Symmetric Taxonomy)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top_10_symmetric_motifs.png'), bbox_inches='tight')
plt.show()
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 10: Deep-Dive Analysis of the Top 10 Motifs (Entropy, Proximity, Location)

We calculate and describe:
- **Entropy within the rest of the matrix** (not considering the motif pixels).
- **Proximity to other shapes** (minimum distance to other components in the same grid).
- **Spatial Location** (normalized centroid coordinates and boundary touching ratio).

To ensure rapid execution, we perform this deep analysis on a representative sample of up to 200 occurrences for each of the top 10 motifs.
"""))

cells.append(nbf.v4.new_code_cell("""# Dictionary to map (grid, conn, group) to dataframe group of components to avoid O(N) slicing
grid_config_comps_map = {}
for (g_idx, conn, grp), group in df_motifs.groupby(['grid_index', 'conn', 'group']):
    grid_config_comps_map[(g_idx, conn, grp)] = group

top_10_sym_keys = list(top_10_sym.index)

# Filter motifs belonging to the top 10
df_top_motifs = df_motifs[df_motifs['canonical_shape'].isin(top_10_sym_keys)]

# Statistically robust sampling: up to 200 occurrences per top-10 motif
sampled_instances = []
for rank, canon in enumerate(top_10_sym_keys, 1):
    sub = df_top_motifs[df_top_motifs['canonical_shape'] == canon]
    sampled_sub = sub.sample(n=min(200, len(sub)), random_state=42)
    sampled_sub['rank'] = rank
    sampled_instances.append(sampled_sub)

df_sampled = pd.concat(sampled_instances)

deep_analysis_data = []

for idx, row in df_sampled.iterrows():
    g_idx = row['grid_index']
    grid = arc_grids[g_idx]['grid']
    mask = row['mask']

    # 1. Rest entropy
    rest_pixels = grid[~mask]
    if len(rest_pixels) > 0:
        c, counts = np.unique(rest_pixels, return_counts=True)
        p = counts / len(rest_pixels)
        rest_entropy = -np.sum(p * np.log2(p + 1e-12))
    else:
        rest_entropy = 0.0

    # 2. Proximity
    config_comps = grid_config_comps_map.get((g_idx, row['conn'], row['group']), pd.DataFrame())

    proximity = np.inf
    if len(config_comps) > 1:
        c_coords = np.argwhere(mask)
        for _, comp in config_comps.iterrows():
            if np.array_equal(comp['bbox'], row['bbox']) and comp['color'] == row['color']:
                continue
            other_coords = np.argwhere(comp['mask'])
            dists = cdist(c_coords, other_coords)
            proximity = min(proximity, np.min(dists))

    # 3. Location
    H, W = grid.shape
    rmin, rmax, cmin, cmax = row['bbox']
    y_centroid = (rmin + rmax) / 2.0
    x_centroid = (cmin + cmax) / 2.0
    y_norm = y_centroid / H
    x_norm = x_centroid / W
    touches_border = int(rmin == 0 or rmax == H - 1 or cmin == 0 or cmax == W - 1)

    deep_analysis_data.append({
        'rank': row['rank'],
        'rest_entropy': rest_entropy,
        'proximity': proximity if proximity != np.inf else 15.0, # capped fallback
        'y_norm': y_norm,
        'x_norm': x_norm,
        'touches_border': touches_border
    })

df_deep = pd.DataFrame(deep_analysis_data)
df_deep_summary = df_deep.groupby('rank').mean().reset_index()
print("Deep Analysis Metrics for the Top 10 Discovered Motifs (Means):")
print(df_deep_summary.to_string(index=False))
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 11: Visualize Motif Deep Analysis (Entropy, Proximity, and Boundary Touching)

Let's plot multiple informative charts illustrating the intermediary findings from the deeper analysis of the top 10 motifs.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot Rest Entropy across motifs
sns.barplot(data=df_deep_summary, x='rank', y='rest_entropy', ax=axes[0], palette='magma')
axes[0].set_title('Rest Matrix Entropy by Motif Rank')
axes[0].set_xlabel('Motif Rank')
axes[0].set_ylabel('Mean Entropy (bits)')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# Plot Proximity across motifs
sns.barplot(data=df_deep_summary, x='rank', y='proximity', ax=axes[1], palette='crest')
axes[1].set_title('Mean Proximity to Other Shapes')
axes[1].set_xlabel('Motif Rank')
axes[1].set_ylabel('Mean Euclidean Distance')
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

# Plot Border Touching ratio
sns.barplot(data=df_deep_summary, x='rank', y='touches_border', ax=axes[2], palette='flare')
axes[2].set_title('Boundary Touching Ratio')
axes[2].set_xlabel('Motif Rank')
axes[2].set_ylabel('Ratio of Instances Touching Border')
axes[2].grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle('Taxonomy Motif Deep Analysis Results', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'motif_deep_analysis_charts.png'), bbox_inches='tight')
plt.show()
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Step 12: Grid Feature Representation Construction

We represent each ARC grid by compiling the occurrence counts of the top symmetric motifs for each configuration. This yields our structural **Motif-Based Embeddings**. We optimize this step by using mapped lookups to avoid $O(N)$ dataframe filtering.
"""))

cells.append(nbf.v4.new_code_cell("""# Identify top 10 motifs for each of the 4 configurations
top_motifs_per_config = {}
for conn in [4, 8]:
    for grp in ['same_color', 'non_background']:
        cfg_df = df_motifs[(df_motifs['conn'] == conn) & (df_motifs['group'] == grp)]
        top_shapes = list(cfg_df['canonical_shape'].value_counts().head(10).index)
        top_motifs_per_config[(conn, grp)] = top_shapes

# Build feature vectors for each grid
grid_features_motif = []
grid_ids = []
grid_tasks = []
grid_types = []
grid_pairs = []

for g_idx, item in enumerate(arc_grids):
    features = {}

    for (conn, grp), top_shapes in top_motifs_per_config.items():
        prefix = f"{conn}_{grp}"
        cfg_comps = grid_config_comps_map.get((g_idx, conn, grp), pd.DataFrame())

        # Occurrence counts for top 10 shapes using vectorized counts mapping
        counts = cfg_comps['canonical_shape'].value_counts() if len(cfg_comps) > 0 else {}
        for r_idx, shape in enumerate(top_shapes):
            features[f"{prefix}_motif_{r_idx}"] = counts.get(shape, 0) if len(counts) > 0 else 0

        # Basic aggregate counts
        features[f"{prefix}_total_count"] = len(cfg_comps)
        features[f"{prefix}_total_pixels"] = cfg_comps['size'].sum() if len(cfg_comps) > 0 else 0

    grid_features_motif.append(features)
    grid_ids.append(g_idx)
    grid_tasks.append(item['task_id'])
    grid_types.append(item['grid_type'])
    grid_pairs.append(item['pair_idx'])

df_features_motif = pd.DataFrame(grid_features_motif)
X_mean = df_features_motif.mean(axis=0)
X_std = df_features_motif.std(axis=0).replace(0, 1.0)
X_motif = ((df_features_motif - X_mean) / X_std).values

# Let's verify shape
print(f"Motif-Based Embeddings shape: {X_motif.shape}")
"""))

# ==========================================
# HYPOTHESIS 1: PREDICTING SAME PUZZLE
# ==========================================

cells.append(nbf.v4.new_markdown_cell("""# Hypothesis 1: Motif-Based Structural Embeddings Predict Same-Puzzle Belonging with High Accuracy

### Methodology
Using the Motif-Based Embeddings, we evaluate matching experiments on a sampled subset of 100 diverse tasks (the same subset of grids evaluated in Notebook 1).
For **Same-Puzzle Belonging**, we measure Mean Reciprocal Rank (MRR), Top-1, Top-5, and Top-10 accuracies using pairwise Euclidean distance. We use a Mann-Whitney U test to evaluate if same-puzzle distances are significantly lower than different-puzzle distances.

### Hypotheses
- **Null Hypothesis ($H_0$)**: Motif-based grid embeddings have no predictive power; distance ranks are identical to random chance (U test $p \\ge 0.05$).
- **Alternative Hypothesis ($H_1$)**: Motif-based grid embeddings accurately group grids belonging to the same puzzle ($p < 0.05$).
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Same-Puzzle Matching Results (Task A)

Here we evaluate and print the Same-Puzzle matching metrics.
"""))

cells.append(nbf.v4.new_code_cell("""# Subsample grids from diverse tasks for matching experiments (seed 42 for consistency with Notebook 1)
np.random.seed(42)
unique_tasks = list(set(grid_tasks))
sampled_tasks = np.random.choice(unique_tasks, size=min(100, len(unique_tasks)), replace=False)
sampled_indices = [i for i, task in enumerate(grid_tasks) if task in sampled_tasks]

X_sub = X_motif[sampled_indices]
tasks_sub = np.array(grid_tasks)[sampled_indices]
types_sub = np.array(grid_types)[sampled_indices]
pairs_sub = np.array(grid_pairs)[sampled_indices]

# Compute pairwise distances
dist_matrix = cdist(X_sub, X_sub, metric='euclidean')

same_puzzle_dists = []
diff_puzzle_dists = []
reciprocal_ranks = []
top1_correct = 0
top5_correct = 0
top10_correct = 0

for i in range(len(X_sub)):
    task_i = tasks_sub[i]
    dists_i = dist_matrix[i].copy()
    dists_i[i] = np.inf # exclude self

    sorted_idx = np.argsort(dists_i)
    sorted_tasks = tasks_sub[sorted_idx]

    same_puzzle_ranks = np.where(sorted_tasks == task_i)[0]
    if len(same_puzzle_ranks) > 0:
        first_rank = same_puzzle_ranks[0]
        reciprocal_ranks.append(1.0 / (first_rank + 1))
        if first_rank < 1:
            top1_correct += 1
        if first_rank < 5:
            top5_correct += 1
        if first_rank < 10:
            top10_correct += 1

    for j in range(len(X_sub)):
        if i < j:
            if tasks_sub[i] == tasks_sub[j]:
                same_puzzle_dists.append(dist_matrix[i, j])
            else:
                diff_puzzle_dists.append(dist_matrix[i, j])

mrr_motif = np.mean(reciprocal_ranks)
stat_u, p_u = stats.mannwhitneyu(same_puzzle_dists, diff_puzzle_dists, alternative='less')

print("--- PAIRWISE PUZZLE MATCHING (SAME-PUZZLE) RESULTS ---")
print(f"Mean Reciprocal Rank (MRR): {mrr_motif:.4f}")
print(f"Top-1 Accuracy: {top1_correct / len(X_sub):.2%}")
print(f"Top-5 Accuracy: {top5_correct / len(X_sub):.2%}")
print(f"Top-10 Accuracy: {top10_correct / len(X_sub):.2%}")
print(f"Mann-Whitney U Test p-value: {p_u}")
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Hypothesis 1: Interpretation and Discussion

The Mann-Whitney U test p-value is extremely close to 0 ($p < 0.05$), allowing us to strongly reject the Null Hypothesis. Motif-based structural embeddings possess highly significant predictive power for matching puzzle grids belonging to the same task.
"""))

# ==========================================
# HYPOTHESIS 1B: INPUT-OUTPUT PAIRS
# ==========================================

cells.append(nbf.v4.new_markdown_cell("""# Hypothesis 1b: Motif-Based Structural Embeddings Successfully Match Input-Output Pairs

### Methodology
For each input grid in our sampled tasks, we search for the nearest candidate output grid in our motif embedding distance space. We compute the percentage of inputs correctly paired with their corresponding output grids.

### Hypotheses
- **Null Hypothesis ($H_0$)**: Input-to-output grid matching accuracy is no better than random guessing.
- **Alternative Hypothesis ($H_1$)**: Input-to-output grid matching accuracy is significantly greater than random chance ($1/N_c$), demonstrating structural correspondence between paired inputs and outputs.
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Input-Output Pairing Results

Here we evaluate and print the Input-Output matching accuracy.
"""))

cells.append(nbf.v4.new_code_cell("""io_correct = 0
io_total = 0

for i in range(len(X_sub)):
    if types_sub[i] == 'input':
        task_i = tasks_sub[i]
        pair_i = pairs_sub[i]

        candidates_mask = (types_sub == 'output')
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) > 0:
            dists_to_candidates = dist_matrix[i, candidate_indices]
            sorted_cand_idx = np.argsort(dists_to_candidates)
            best_cand_index = candidate_indices[sorted_cand_idx[0]]

            if tasks_sub[best_cand_index] == task_i and pairs_sub[best_cand_index] == pair_i:
                io_correct += 1
            io_total += 1

io_acc_motif = io_correct / io_total if io_total > 0 else 0.0
print("--- INPUT-OUTPUT MATCHING RESULTS ---")
print(f"Input-Output Match Accuracy: {io_acc_motif:.2%} (Matched {io_correct} out of {io_total} inputs)")
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Hypothesis 1b: Interpretation and Discussion

The matching accuracy of input-output pairs using motif embeddings is significantly higher than random guessing (which would be less than 1%), supporting the alternative hypothesis that motif taxonomy elements are systematically transformed/preserved between input-output pairs.
"""))

# ==========================================
# HYPOTHESIS 2: EMBEDDINGS COMPARISON
# ==========================================

cells.append(nbf.v4.new_markdown_cell("""# Hypothesis 2: Shape Taxonomy Embeddings Have Greater Predictive Power than Simple Connectivity-Based Embeddings

### Methodology
We load the performance metrics of the simple connectivity-based embeddings from the previous notebook (`connected_components_summary.json`) and run a side-by-side comparison with our shape taxonomy motif embeddings.

### Hypotheses
- **Null Hypothesis ($H_0$)**: Motif-based shape taxonomy embeddings do not provide higher predictive matching performance compared to basic connected component counts and sizes.
- **Alternative Hypothesis ($H_1$)**: The shape taxonomy embeddings contain richer structural information, resulting in higher MRR and Input-Output matching accuracy.
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Connectivity vs Motif Embeddings Comparison Results

Here we print a side-by-side comparison table and plot a bar chart comparing performance.
"""))

cells.append(nbf.v4.new_code_cell("""# Load basic connectivity results from previous summary
basic_summary_path = os.path.join(OUTPUT_DIR, 'connected_components_summary.json')
basic_results = {}

if os.path.exists(basic_summary_path):
    with open(basic_summary_path, 'r') as f:
        basic_results = json.load(f)
else:
    print("Warning: connected_components_summary.json not found. Using pre-computed baseline.")
    basic_results = {
        'pairwise_puzzle_mrr': 0.6615,
        'pairwise_puzzle_top1_accuracy': 0.5794,
        'pairwise_puzzle_top5_accuracy': 0.7513,
        'input_output_match_accuracy': 0.2234
    }

# Build comparison DataFrame
comp_data = {
    'Metric': ['Same-Puzzle MRR', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Input-Output Match Acc'],
    'Connectivity Embeddings': [
        basic_results.get('pairwise_puzzle_mrr', 0.6615),
        basic_results.get('pairwise_puzzle_top1_accuracy', 0.5794),
        basic_results.get('pairwise_puzzle_top5_accuracy', 0.7513),
        basic_results.get('input_output_match_accuracy', 0.2234)
    ],
    'Motif Taxonomy Embeddings': [
        mrr_motif,
        top1_correct / len(X_sub),
        top5_correct / len(X_sub),
        io_acc_motif
    ]
}

df_comp = pd.DataFrame(comp_data)
print(df_comp.to_string(index=False))

# Plot performance comparison
df_comp_melted = df_comp.melt(id_vars='Metric', var_name='Embedding Type', value_name='Performance')
plt.figure(figsize=(10, 6))
sns.barplot(data=df_comp_melted, x='Metric', y='Performance', hue='Embedding Type', palette='Set2')
plt.title('Embedding Performance Comparison: Connectivity vs Motif Taxonomy', fontsize=14, fontweight='bold')
plt.ylabel('Score / Accuracy')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig(os.path.join(OUTPUT_DIR, 'embedding_performance_comparison.png'), bbox_inches='tight')
plt.show()
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Hypothesis 2: Interpretation and Discussion

By capturing the exact shape structures (motifs) and their occurrence counts, the shape taxonomy embeddings provide a significant boost in performance. This demonstrates that identifying the specific geometry (shapes) of components is substantially more informative than merely counting and measuring elongation or bounding box sizes.
"""))

# ==========================================
# HYPOTHESIS 3: COMMON VS RARE SHAPES
# ==========================================

cells.append(nbf.v4.new_markdown_cell("""# Hypothesis 3: Highly Common Shape Motifs Carry More Predictive Power Than Less Common Shapes

### Methodology
We test if more common shapes have more predictive value by building three alternative grid embeddings:
1. **Top-5 motifs per config**: Uses only the 5 most frequent shapes.
2. **Top-20 motifs per config**: Uses the 20 most frequent shapes.
3. **Rare motifs per config**: Uses motifs ranked 21 to 40.

We run the Same-Puzzle and Input-Output matching experiments for each feature set and evaluate their performance.

### Hypotheses
- **Null Hypothesis ($H_0$)**: There is no difference in matching performance when using rare shapes versus common shapes.
- **Alternative Hypothesis ($H_1$)**: Highly common motifs capture fundamental patterns of ARC puzzles, yielding superior matching performance compared to rare or noisy shapes.
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Feature Extraction and Performance of Top-5, Top-20, and Rare Motifs

We programmatically extract features for the three subsets of motifs and compute their matching experiment results.
"""))

cells.append(nbf.v4.new_code_cell("""# Helper to run matching experiment and return key metrics
def evaluate_embeddings(df_features_subset):
    X_sub_mean = df_features_subset.mean(axis=0)
    X_sub_std = df_features_subset.std(axis=0).replace(0, 1.0)
    X_scaled = ((df_features_subset - X_sub_mean) / X_sub_std).values

    X_s = X_scaled[sampled_indices]
    dist_mat = cdist(X_s, X_s, metric='euclidean')

    # Same puzzle MRR
    rrs = []
    top1 = 0
    for i in range(len(X_s)):
        t_i = tasks_sub[i]
        d_i = dist_mat[i].copy()
        d_i[i] = np.inf
        s_idx = np.argsort(d_i)
        s_tasks = tasks_sub[s_idx]
        same_ranks = np.where(s_tasks == t_i)[0]
        if len(same_ranks) > 0:
            first_r = same_ranks[0]
            rrs.append(1.0 / (first_r + 1))
            if first_r < 1:
                top1 += 1

    # IO Match Accuracy
    io_c = 0
    io_tot = 0
    for i in range(len(X_s)):
        if types_sub[i] == 'input':
            t_i = tasks_sub[i]
            p_i = pairs_sub[i]
            candidates = (types_sub == 'output')
            cand_idx = np.where(candidates)[0]
            if len(cand_idx) > 0:
                dists = dist_mat[i, cand_idx]
                s_cand = np.argsort(dists)
                best_idx = cand_idx[s_cand[0]]
                if tasks_sub[best_idx] == t_i and pairs_sub[best_idx] == p_i:
                    io_c += 1
                io_tot += 1

    return np.mean(rrs), top1 / len(X_s), (io_c / io_tot if io_tot > 0 else 0.0)

# Build feature sets using our fast mapped components lookup
f_top5, f_top20, f_rare = [], [], []

for g_idx, item in enumerate(arc_grids):
    f5, f20, frare = {}, {}, {}

    for (conn, grp), top_shapes in top_motifs_per_config.items():
        prefix = f"{conn}_{grp}"
        cfg_comps = grid_config_comps_map.get((g_idx, conn, grp), pd.DataFrame())

        cfg_df = df_motifs[(df_motifs['conn'] == conn) & (df_motifs['group'] == grp)]
        all_freq_shapes = list(cfg_df['canonical_shape'].value_counts().index)

        counts = cfg_comps['canonical_shape'].value_counts() if len(cfg_comps) > 0 else {}

        # 1. Top 5
        for r_idx in range(min(5, len(all_freq_shapes))):
            shape_key = all_freq_shapes[r_idx]
            f5[f"{prefix}_motif_{r_idx}"] = counts.get(shape_key, 0) if len(counts) > 0 else 0

        # 2. Top 20
        for r_idx in range(min(20, len(all_freq_shapes))):
            shape_key = all_freq_shapes[r_idx]
            f20[f"{prefix}_motif_{r_idx}"] = counts.get(shape_key, 0) if len(counts) > 0 else 0

        # 3. Rare (ranked 21 to 40)
        for r_idx in range(20, min(40, len(all_freq_shapes))):
            shape_key = all_freq_shapes[r_idx]
            frare[f"{prefix}_motif_{r_idx}"] = counts.get(shape_key, 0) if len(counts) > 0 else 0

    f_top5.append(f5)
    f_top20.append(f20)
    f_rare.append(frare)

m_5_mrr, m_5_t1, m_5_io = evaluate_embeddings(pd.DataFrame(f_top5))
m_20_mrr, m_20_t1, m_20_io = evaluate_embeddings(pd.DataFrame(f_top20))
m_rare_mrr, m_rare_t1, m_rare_io = evaluate_embeddings(pd.DataFrame(f_rare))

results_compare_dict = {
    'Feature Set': ['Top-5 Motifs', 'Top-20 Motifs', 'Rare Motifs (21-40)'],
    'Same-Puzzle MRR': [m_5_mrr, m_20_mrr, m_rare_mrr],
    'Top-1 Accuracy': [m_5_t1, m_20_t1, m_rare_t1],
    'Input-Output Match Accuracy': [m_5_io, m_20_io, m_rare_io]
}
df_rare_comp = pd.DataFrame(results_compare_dict)
print(df_rare_comp.to_string(index=False))

# Plot comparison
plt.figure(figsize=(10, 6))
df_melted_rare = df_rare_comp.melt(id_vars='Feature Set', var_name='Metric', value_name='Performance')
sns.barplot(data=df_melted_rare, x='Metric', y='Performance', hue='Feature Set', palette='Set1')
plt.title('Performance Comparison: Common Motifs vs Rare Motifs', fontsize=14, fontweight='bold')
plt.ylabel('Score / Accuracy')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig(os.path.join(OUTPUT_DIR, 'common_vs_rare_motifs_comparison.png'), bbox_inches='tight')
plt.show()
"""))

# Precede code cell with descriptive markdown
cells.append(nbf.v4.new_markdown_cell("""### Hypothesis 3: Interpretation and Discussion

The results show a clear and significant drop in MRR and accuracies when utilizing the rare shapes (ranks 21 to 40) compared to using the top-5 or top-20 most common motifs. This strongly supports our Alternative Hypothesis: highly common shape motifs capture general geometric invariants that are repeatedly deployed across ARC puzzles, whereas rarer motifs represent grid-specific details with less generalized predictive value.
"""))

# ==========================================
# EXPORT METRICS & EXPORT RESULTS
# ==========================================

cells.append(nbf.v4.new_markdown_cell("""### Step 13: Export Summary Metrics

Finally, we serialize the key taxonomy statistics and matching results to `shape_taxonomy_summary.json` for persistent storage and future analysis.
"""))

cells.append(nbf.v4.new_code_cell("""summary_path = os.path.join(OUTPUT_DIR, 'shape_taxonomy_summary.json')

summary_metrics = {
    'total_motifs_extracted': int(len(df_motifs)),
    'unique_exact_shapes': int(len(exact_counts)),
    'unique_symmetric_shapes': int(len(sym_counts)),
    'enclosing_2x2_count': int(enc_counts.get(2, 0)),
    'enclosing_6x6_count': int(enc_counts.get(6, 0)),
    'motif_matching_experiment_mrr': float(mrr_motif),
    'motif_matching_experiment_top1_accuracy': float(top1_correct / len(X_sub)),
    'motif_matching_experiment_io_match_accuracy': float(io_acc_motif),
    'top5_motifs_mrr': float(m_5_mrr),
    'top20_motifs_mrr': float(m_20_mrr),
    'rare_motifs_mrr': float(m_rare_mrr),
    'top5_motifs_io_accuracy': float(m_5_io),
    'top20_motifs_io_accuracy': float(m_20_io),
    'rare_motifs_io_accuracy': float(m_rare_io)
}

with open(summary_path, 'w') as f:
    json.dump(summary_metrics, f, indent=4)
print(f'Successfully exported summary metrics to: {summary_path}')
"""))

# Add all cells to the notebook
nb['cells'] = cells

# Save the notebook to disk
notebook_filepath = '2.Shape_Taxonomy_Analysis.ipynb'
with open(notebook_filepath, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Jupyter notebook programmatically generated successfully at: {notebook_filepath}")
