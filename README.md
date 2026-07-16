# Motifs

Embedding library for finding, evaluating, and representing motifs in the ARC (Abstraction and Reasoning Corpus) dataset.

## Consolidated ARC-AGI-2 Dataset Exports

To facilitate seamless importing and integration into downstream applications (e.g., neural pipelines and interactive web tools), the dispersed raw ARC-AGI-2 dataset has been consolidated into single-file JSON aggregates:

-   **Consolidated Training Set**: `motifs/arc_training_consolidated.json`
    -   Contains exactly 1,000 tasks.
-   **Consolidated Evaluation Set**: `motifs/arc_evaluation_consolidated.json`
    -   Contains exactly 120 tasks.

Each file is structured as a dictionary where keys are unique Task IDs (e.g., `"007bbfb7"`) and values are dictionaries containing lists of input-output grid matrices for train and test pairs:
```json
{
  "007bbfb7": {
    "train": [
      { "input": [[...]], "output": [[...]] }
    ],
    "test": [
      { "input": [[...]], "output": [[...]] }
    ]
  }
}
```

## Notebook Directory

-   **0-A.Data_Consolidation_and_Export.ipynb**: Downloads the official ARC-AGI-2 dataset repository and compiles training and evaluation tasks into single-file JSON exports.
-   **0.ARC_and_Random_Matrix_Comparison.ipynb**: Establishes baseline comparisons between true ARC grids and randomized control matrices.
-   **1.Connected_Components_Analysis.ipynb**: Explores connected component statistics under 4-adjacency and 8-adjacency for same-color and non-background configurations, establishing baseline connectivity embeddings.
-   **2.Shape_Taxonomy_Analysis.ipynb**: Discovers a geometric shape taxonomy, canonicalizes shape motifs under Dihedral $D_4$ symmetries, maps them to square enclosing matrices (2x2 to 6x6), and analyzes motif embeddings for puzzle matching.
-   **3.Combined_Feature_Clustering.ipynb**: Performs balanced, size-constrained clustering of the complete training corpus (all 1,000 puzzles, 8,616 matrices) on 22-dimensional connectivity embeddings. Ordered by exact Euclidean relevance distance.
-   **4.Input_Output_Structural_Analysis.ipynb**: Performs systematic, comparative quantitative analysis of structural, geometric, and topological differences between input and output matrices across the ARC-AGI-2 corpus.
-   **5.Feature_Dimensionality_Analysis.ipynb**: Systematically sweeps approximately 50 distinct dimensional sizes to locate optimal representational bottlenecks and analyze how over-parameterization/curse of dimensionality degrades performance.

## Future Plans & TODOs

-   [x] **DONE**: Focus the next notebooks on understanding the structural differences between input and output matrices.
-   [x] **DONE**: Split and expand the clustering/feature dimensionality analysis on the complete training set.
-   [ ] **TODO**: Conduct future analysis of statistical shapes based on the most common motifs discovered in the shape taxonomy.
