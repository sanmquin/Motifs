# Advanced Motifs Analysis

This directory contains research, extraction, and consolidated analysis notebooks focusing on advanced geometric motifs (such as lines, diagonal lines, squares, rectangles, and junctions) in the ARC-AGI-2 dataset.

## Notebook Sequence

1. **0.Line_Analysis.ipynb**: In-depth analysis of same-color horizontal and vertical lines of size $L \ge 3$, utilizing optimized $O(L)$ parallel containment checks to ensure lines are not part of larger 2x2 solid blocks.
2. **1.Square_Analysis.ipynb**: Advanced solid and hollow square motif extraction, utilizing pre-checked corner vertices to accelerate combinatorial searches.
3. **2.Rectangle_Analysis.ipynb**: Optimized Same-color rectangle searches utilizing perimeter cell counts to speed up positional grid sweeps.
4. **3.Corner_T_Cross_Analysis.ipynb**: Extracting orthogonal and diagonal junctions (Corners/L-shapes, T-shapes, and Crosses) by tracing arm/stem lengths.
5. **4.Consolidated_Analysis.ipynb**: Comprehensive consolidated frequency and co-occurrence correlation analyses, along with multi-dimensional embedding ablation evaluations for same-puzzle matching and input-output pairing retrieval.

## Future Plans & TODOs

- [ ] **TODO**: Consider expanded embedding versions for the motifs embeddings, use weighted distances.
- [ ] **TODO**: Test predictions against the evaluation dataset.
