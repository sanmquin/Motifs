# 7. I/O Pair Representation & Same-Puzzle Classification

This directory contains research and implementations for representing **Input-Output (I/O) pairs** as multi-dimensional joint embeddings, capturing rule-based transformations in the Abstraction and Reasoning Corpus 2 (ARC-AGI-2).

Rather than encoding isolated grids, our framework constructs joint features modeling how spatial dimensions, connected component populations, pixel colors, spatial distributions, and component geometries transform from the input grid to the output grid.

## Research Notebook

- **7.IO_Pair_Representation.ipynb**: Tests the hypothesis that joint I/O pair embeddings facilitate high-accuracy same-puzzle classification and outperform individual grid representations. It constructs a multi-dimensional transformation feature vector for every training pair, evaluates cosine similarity distributions, and tests same-puzzle pairing classification.

---

## Implemented Feature Taxonomy

### 1. Spatial & Grid Geometry Transformations
- **Matrix Resize Features**: Extracts height and width ratio ($\frac{H_{out}}{H_{in}}, \frac{W_{out}}{W_{in}}$), area scaling factors ($\frac{\text{Area}_{out}}{\text{Area}_{in}}$), and change in aspect ratio/elongation.

### 2. Connected Component Cardinality Shifts
- **Component Counting**: Computes component counts under 4-adjacency (same-color) and 8-adjacency (non-background) on input and output grids, and calculates the delta ($\Delta = \text{Count}_{out} - \text{Count}_{in}$).

### 3. Change in Color of Components
- **Color Mapping Frequencies**: Builds a $10 \times 10$ normalized bipartite color transition matrix by measuring overlapping input-output pixels. This enables O(1) identification of full same-color shifts or partial color-mapping transitions.

### 4. Color Density & Non-Background Pixels
- **Visual Order and Complexity**: Extracts Shannon entropy differences ($\Delta \text{Entropy} = \text{Entropy}_{out} - \text{Entropy}_{in}$) and active/non-background pixel ratio differences to model whether output grids represent consolidated, simplified visual states.

### 5. Change in Location of Components
- **Centroid Translation Offsets**: Measures the coordinate shift (dx, dy) of the global center-of-mass of active pixels between input and output grids to capture translation-invariance or systematic directional movements.

### 6. Change in Colors & Background
- **Additions and Removals**: Tracks the count of unique colors added to the output or removed from the input, along with specific checks for whether the background color (approximate from corner pixels) underwent alteration.

### 7. Component Size Distributions
- **Scale and Shape Summary**: Extracts the mean, max, and standard deviation of connected component sizes under multiple connectivity configurations to summarize spatial fragmentation or consolidation.

### 8. Boundary Distributions and Symmetries
- **Boundary Distributions**: Tracks active pixel ratios along grid borders and corners to model whether components tend to reside on margins.
- **Global Symmetries**: Measures horizontal, vertical, and rotational symmetry scores (for 90°, 180°, and 270° rotations) across input, output, and transformation states.

---

## Advanced Feature Support

### 1. Nested Component Detection
- **Hierarchical Containment**: Programmatically detects nested structures by evaluating whether non-background component bounding boxes are entirely contained within larger non-background component bounding boxes.

### 2. Component Type Taxonomy
- **Lines**: Orthogonal 1D structures of length >= 3.
- **Squares**: Solid same-color squares of size >= 2x2.
- **Rectangles**: Solid same-color rectangles (non-square, size >= 2x2).
- **Corners, T-Shapes, and Crosses**: Junction-based orthogonal structures.
- **Diagonals**: Same-color components structured strictly diagonally.

---

## Future Improvements & Missing Steps (TODOs)

- [ ] **TODO: Incorporate Bipartite Object Tracking**: Map individual connected components between input and output using maximum-bipartite matching or IoU-overlap thresholds to extract exact per-object translation vectors (dx, dy) rather than global centroid shifts.
- [ ] **TODO: Expand Nested Color Hierarchies**: Model the color relationship of nested components (e.g., "color A inside color B") to handle specific containment rules.
- [ ] **TODO: Integrate Dihedral Symmetric Normalization**: Standardize and canonicalize grid orientations using Dihedral $D_4$ symmetries before feature extraction to ensure absolute rotation and reflection invariance.
- [ ] **TODO: Train Metric Learning Models**: Replace cosine similarity of standardized features with a learned metric distance (e.g., training a triplet Siamese Network) to specialize representations for task/puzzle retrieval.
