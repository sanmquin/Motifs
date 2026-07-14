# Information Density and Structural Conservation in Abstract Visual Reasoning: A Quantitative Analysis of Input-Output Matrix Mappings in the Abstraction and Reasoning Corpus

**Author:** Jules, Senior Research Engineer
**Date:** July 2024
**Affiliation:** Motifs AI Research Laboratory

---

### Abstract

The Abstraction and Reasoning Corpus (ARC) provides a foundational benchmark for measuring human-like general intelligence by presenting abstract visual grids that require inductive reasoning from very few examples. Although extensive research has focused on generating symbolic domain-specific languages and learning relational graph representations, there remains a critical gap in understanding the raw statistical, topological, and informational transformations that map input matrices to output matrices. In this study, we conduct a systematic, quantitative comparative analysis of 3,500 input-output grid pairs extracted across the training and evaluation splits of the ARC benchmark. We extract multi-dimensional structural features, including grid area, shape elongation, Shannon information entropy, same-color connected component topography under eight-connectivity, and edge transition density.

We formulate and test three core hypotheses regarding how space, objects, and colors are transformed. Under Hypothesis 1, we establish that output grids experience a highly significant systematic reduction in size on average, with the mean grid area collapsing from 186.67 to 152.07 pixels ($p = 1.77 \times 10^{-69}$), driven by target crop operations. Under Hypothesis 2, we show that same-color connected component cardinality significantly decreases in the output space ($p = 0.0045$), reflecting a visual prior of object consolidation and background noise elimination. Under Hypothesis 3, we uncover a counter-intuitive phenomenon: despite a reduction in active color diversity, output grids exhibit a highly significant *increase* in Shannon information entropy ($p = 3.01 \times 10^{-61}$) and spatial transition density ($p = 1.48 \times 10^{-42}$). This increase in entropy is caused by the tight cropping of output matrices, which removes large uniform background regions and dramatically elevates local visual complexity and information density. Our findings provide researchers with a rigorous mathematical foundation to guide search-space reduction and crop-priority heuristics in automated ARC solvers.

---

## 1. Introduction

Evaluating artificial general intelligence has historically been complicated by the conflation of broad generalized reasoning with narrow task-specific memorization. The Abstraction and Reasoning Corpus (ARC-AGI), introduced by François Chollet, sidesteps this pitfall by presenting visual puzzles that cannot be solved via traditional deep learning architectures trained on billions of parameters. Instead, ARC puzzles present three to five training examples of input-output grid pairs, and the reasoning agent must infer the underlying rules to solve an unseen test input. These puzzles are designed around core knowledge priors of human developmental psychology: object cohesion, basic geometry, symmetry, topological boundaries, and color motifs.

While the ultimate goal of an ARC solver is to synthesize a symbolic or programmatic function that transforms the input matrix to the correct output matrix, most search engines and DSL solvers suffer from massive combinatorial explosion. They generate hundreds of thousands of candidate programs without any guidance regarding the likely shape, size, or complexity of the target output grid. To prune this search space, it is vital to mathematically map out the physical and statistical characteristics of input-output pairs. When a researcher can explain exactly how output matrices differ from input matrices, they can engineer precise inductive biases and heuristic filters.

In this work, we present a rigorous, large-scale quantitative analysis of the structural, geometric, and topological transformations that map inputs to outputs in ARC. We analyze over 3,500 input-output matrix pairs across 800 tasks spanning both the training and evaluation splits. Rather than limiting our study to simple dimension checks, we construct a comprehensive set of descriptive features including grid scale, shape aspect ratio, Shannon information entropy, same-color connected component topology under eight-connectivity, and transition density.

Through these descriptive features, we test three formal hypotheses. First, we investigate whether ARC output grids preserve or alter the spatial boundaries of the input grid. Second, we examine whether output grids exhibit higher object cohesion—manifested as a reduction in connected component cardinality—indicating that abstract reasoning consolidates fragmented inputs into structured figures. Third, we measure changes in information theory metrics, evaluating whether the mapping process reduces or enhances visual complexity and entropy.

By analyzing these metrics, we reveal a critical and counter-intuitive topological shift: output matrices represent a highly concentrated, condensed state of information. The output grids are physically smaller on average, yet they possess significantly higher Shannon information entropy and edge density. This finding reframes the task of ARC solving not as a generative expansion, but as an informational compression process where the background noise is stripped away to highlight the core geometric motifs of the task.

---

## 2. Theoretical Framework and Analytical Methodology

To conduct a quantitative comparison of inputs and outputs, we establish a theoretical framework grounded in digital image topology and information theory. We treat each ARC grid as a two-dimensional matrix $G \in \{0, 1, \dots, 9\}^{H \times W}$, where $H$ is the height, $W$ is the width, and the integers represent distinct color values. The value $0$ represents the black background, and values $1$ through $9$ represent active foreground colors.

Our dataset is constructed by ingesting the consolidated training and evaluation sets, giving us a complete census of all 800 tasks. For each task, we extract both the training pairs and the test pairs (relying on the test outputs provided within the benchmark dataset), giving us a total pool of 3,500 unique pairs. This represents a substantial increase in statistical power compared to studies that examine only small subsets of tasks.

For each grid, we extract a vector of descriptive characteristics across four thematic domains comprising spatial dimensions, color diversity, connected component topography, and visual complexity.

### 2.1 Spatial Dimensions and Scale

The spatial scale of a grid is defined by its height $H$ and width $W$. The total grid area $A$ is the product of its dimensions:
$$A = H \times W$$
The elongation ratio $R$ represents the aspect ratio of the grid, capturing whether the matrix is square or highly elongated:
$$R = \frac{\max(H, W)}{\min(H, W)}$$
For grids that are perfectly square, $R$ is exactly 1.0. If either dimension is zero, $R$ defaults to 1.0.

### 2.2 Color Diversity and Information Entropy

We measure color complexity by calculating two metrics: color diversity and Shannon information entropy. Color diversity $C$ is defined as the number of unique active non-background colors present in the grid. Shannon information entropy $S$ measures the average uncertainty or information content in the distribution of grid pixel values. Let $p_c$ be the empirical probability of a color $c \in \{0, 1, \dots, 9\}$ in the grid:
$$S = -\sum_{c \in \text{unique}(G)} p_c \log_2(p_c)$$
When a grid is dominated by a single background color (e.g., a large black grid with a few colored pixels), the probability of the background color is close to 1.0, and the Shannon entropy approaches zero. Conversely, if multiple colors are distributed more equitably across the grid, the entropy increases.

### 2.3 Connected Component Topography

To capture object-centric structure, we segment each grid into cohesive same-color components. We use eight-connectivity ($N_8$), which permits diagonal connections between adjacent pixels of the same color, aligning with human visual grouping principles.

For each grid, we identify all same-color components of any size (greater than or equal to 1). Let $N$ be the total count of same-color components in the grid. For each component $C_k$, we compute its size $S(C_k)$ as the total number of pixels it contains. We also compute its bounding box elongation $E(C_k)$ to capture the aspect ratio of individual objects:
$$E(C_k) = \frac{\max(h_k, w_k)}{\min(h_k, w_k)}$$
where $h_k$ and $w_k$ are the height and width of the bounding box enclosing component $C_k$. We average these individual component sizes and elongations to obtain the mean component size and mean component elongation for the grid, and we also track the maximum size and maximum elongation.

### 2.4 Visual Complexity and Transition Density

To measure visual fragmentation and high-frequency boundaries, we define a transition density metric $T$. Transition density represents the proportion of adjacent cell pairs (horizontally and vertically) that contain different color values. This captures the frequency of edge transitions in the matrix, indicating whether the grid is composed of solid blocks of color or is highly fragmented and detailed.

Let $N_H$ be the number of horizontally adjacent pixel pairs and $N_V$ be the number of vertically adjacent pairs. We define:
$$T = \frac{\sum_{i=1}^{H} \sum_{j=1}^{W-1} \mathbb{I}[G(i, j) \neq G(i, j+1)] + \sum_{i=1}^{H-1} \sum_{j=1}^{W} \mathbb{I}[G(i, j) \neq G(i+1, j)]}{H(W-1) + (H-1)W}$$
where $\mathbb{I}$ is the indicator function. A transition density of 0.0 indicates a completely uniform grid, while higher values indicate complex, highly detailed grids with numerous boundaries.

---

## 3. Descriptive Profiling and Exploratory Statistics

Before conducting formal hypothesis testing, we analyze the descriptive statistics of our features across the entire sample of 3,500 grid pairs. This exploratory profiling allows us to observe general trends and locate overall structural differences between inputs and outputs. Table 1 outlines the means, standard deviations, and mean differences for the selected features.

### Table 1: Summary Statistics of Input and Output Grid Characteristics

| Metric / Characteristic | Input Mean | Input SD | Output Mean | Output SD | Mean Difference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Grid Height ($H$) | 11.84 | 6.83 | 10.32 | 6.74 | -1.52 |
| Grid Width ($W$) | 12.30 | 6.84 | 10.73 | 6.85 | -1.57 |
| Grid Area ($A$) | 186.67 | 202.35 | 152.07 | 189.68 | -34.60 |
| Grid Elongation ($R$) | 1.295 | 0.845 | 1.277 | 0.722 | -0.018 |
| Shannon Information Entropy ($S$) | 1.087 | 0.646 | 1.223 | 0.582 | +0.136 |
| Active Color Diversity ($C$) | 2.96 | 2.02 | 2.85 | 1.71 | -0.11 |
| Connected Component Count ($N$) | 13.85 | 36.22 | 12.26 | 34.09 | -1.59 |
| Mean Component Size | 10.76 | 23.40 | 10.39 | 17.44 | -0.37 |
| Mean Component Elongation | 1.547 | 1.085 | 1.645 | 1.166 | +0.098 |
| Transition Density ($T$) | 0.324 | 0.226 | 0.377 | 0.227 | +0.053 |

The exploratory profiling reveals distinct structural trends. First, we observe a general contraction in grid dimensions. The average height decreases from 11.84 to 10.32, and the average width decreases from 12.30 to 10.73. This results in a massive average area reduction of 34.60 pixels per grid. Despite this physical shrinkage, the elongation ratio of the grid container remains remarkably stable, only decreasing by 0.018, which suggests that the overall aspect ratio of the grid is generally conserved even when its size is reduced.

Second, we observe a fascinating increase in Shannon information entropy and transition density. The average Shannon entropy rises from 1.087 bits in the input to 1.223 bits in the output, an increase of 0.136 bits. Similarly, the transition density rises from 0.324 to 0.377, an increase of 0.053. This indicates that output grids contain a higher density of color boundaries and are more informationally dense. This is especially interesting because the active color diversity actually decreases slightly by an average of 0.11 colors, falling from 2.96 to 2.85.

Third, the connected component properties reveal a reduction in overall component count from 13.85 to 12.26 (a drop of 1.59 components per grid), while the mean size of components remains relatively stable, slightly declining from 10.76 to 10.39 pixels. Meanwhile, the mean elongation of individual components increases from 1.547 to 1.645, indicating that components in output grids are slightly more stretched or directional than in input grids.

These exploratory findings point toward a consistent theme: abstract reasoning in ARC involves compressing spatial matrices, removing unneeded black background space, and concentrating the active geometric components, which increases the local informational entropy and transition density of the output grids.

---

## 4. Hypothesis 1: The Output Grid Dimension Hypothesis

We now formulate and test our first hypothesis to establish whether the observed contraction in grid area is statistically significant.

### 4.1 Statement of Hypotheses

We investigate whether output matrices experience systematic dimensional alterations or if they preserve spatial boundaries. We define the change in grid area as the difference between output area and input area. The null hypothesis ($H_0^1$) states that the median difference in grid area between input and output matrices is equal to zero, indicating that there is no systematic scaling or resizing of grid containers. The alternative hypothesis ($H_a^1$) states that the median difference in grid area between input and output matrices is significantly different from zero, which would suggest a directional bias in grid scaling across the corpus.

### 4.2 Statistical Results and Visualizations

To test this hypothesis, we perform a two-sided Wilcoxon signed-rank test on the paired input and output area measurements. The Wilcoxon signed-rank test is selected because grid area differences are highly non-normal and contain discrete, repeating values. The statistical test yields a Wilcoxon test statistic of $170,705.0$ and an extremely small p-value of $1.7726 \times 10^{-69}$. This allows us to reject the null hypothesis with absolute confidence and accept the alternative hypothesis. The output grid area is statistically and significantly smaller than the input grid area.

To visualize this trend, we bin the grid areas into five distinct cohorts representing different scale ranges, including Micro (1-10 pixels), Small (11-50 pixels), Medium (51-150 pixels), Large (151-400 pixels), and Gigantic (401+ pixels). We construct a grouped bar chart showing the frequency distribution of inputs and outputs across these cohorts, which is exported to the file `area_cohort_distribution.png`.

The grouped bar chart demonstrates a clear decay distribution. The large majority of grids, both input and output, reside in the Medium and Large cohorts, which correspond to typical ARC grid sizes like 10x10, 15x15, or 20x20. However, we observe a significant shift in the tails of the distribution. The output grids have a higher frequency in the Micro and Small cohorts, and a lower frequency in the Large and Gigantic cohorts compared to inputs. This confirms a systematic shift toward smaller grid areas.

### 4.3 Interpretation and Cognitive Implications

The rejection of the null hypothesis confirms that output grids are systematically smaller than input grids. This size reduction has profound implications for researchers. In ARC, many tasks require the solver to isolate a specific object, crop out a target pattern, or extract a small subgrid containing a geometric key.

When a task requires cropping, the output matrix is typically a small fraction of the input size. Because cropping operations are far more common in ARC than scaling-up operations, the overall population exhibits a strong directional bias toward smaller output matrices.

For automated solvers, this means that the search space for output dimensions is highly constrained. If an input grid is large, a solver's crop heuristics should be heavily biased toward finding smaller subgrids. The spatial container of the output is not a random variable; it is systematically bound and scaled-down.

---

## 5. Hypothesis 2: The Connected Component Conservation Hypothesis

Next, we evaluate whether abstract reasoning tasks alter the topography of same-color connected components, testing whether the output grids exhibit higher object cohesion and a reduction in component cardinality.

### 5.1 Statement of Hypotheses

Let the change in component count be defined as the difference between the output component count and the input component count. The null hypothesis ($H_0^2$) states that the median count of same-color connected components is identical between input and output grids. The alternative hypothesis ($H_a^2$) states that the median count of same-color connected components differs significantly between input and output grids, suggesting that visual reasoning systematically consolidates or disperses active visual elements.

### 5.2 Statistical Results and Visualizations

We perform a paired Wilcoxon signed-rank test on the connected component counts of input and output grids. The test yields a test statistic of $1,446,568.5$ and a p-value of $0.0045$. Because the p-value is well below our significance threshold ($\alpha = 0.05$), we reject the null hypothesis in favor of the alternative hypothesis. The number of same-color connected components is significantly different between input and output matrices, with a clear directional decline.

We group the component counts into five cohorts, including None (0 components), Sparse (1-2 components), Low (3-5 components), Moderate (6-10 components), and Dense (11+ components). We construct a grouped bar chart illustrating these counts, which is exported to the file `component_count_distribution.png`.

The visualization shows a decay distribution where the majority of ARC grids possess fewer than 10 connected components, reflecting the human developmental prior of sparse, cohesive objects. When comparing inputs and outputs, the output grids exhibit a higher frequency in the Sparse and Low cohorts, while the input grids exhibit a higher frequency in the Dense cohort. This provides visual validation of the reduction in component cardinality.

### 5.3 Interpretation and Cognitive Implications

The significant decrease in connected component counts represents a fundamental cognitive prior of ARC: object consolidation. Input grids are often presented with scattered elements, background clutter, or structural scaffolding. The process of solving an ARC task frequently involves removing these temporary scaffolding lines, aligning scattered blocks into a single cohesive structure, or filtering out high-frequency noise.

This results in an output grid that contains fewer, more consolidated objects. For instance, in connectivity tasks, multiple isolated input components are merged into a single continuous path or shape, collapsing the total component count.

For model architects, this finding validates the use of cohesion priors. When generating candidate output grids, configurations that exhibit high fragmentation should be heavily penalized or filtered out. The abstract reasoning process is a journey from high-entropy fragmentation to low-entropy, cohesive geometric structures.

---

## 6. Hypothesis 3: The Color Complexity and Information Entropy Reduction Hypothesis

Our third hypothesis examines the information-theoretic properties of ARC grids, focusing on how Shannon entropy and transition density are modified.

### 6.1 Statement of Hypotheses

Let the change in entropy be defined as the difference between output Shannon entropy and input Shannon entropy. We investigate whether output matrices exhibit higher order, formulating the hypotheses where the null hypothesis ($H_0^3$) states that the median Shannon information entropy of output grids is identical to or greater than that of input grids. The alternative hypothesis ($H_a^3$) states that the median Shannon information entropy of output grids is significantly lower than that of input grids, representing a global increase in spatial uniformity and order.

### 6.2 Statistical Results and Visualizations

We execute a one-tailed Wilcoxon signed-rank test to determine if output grids have significantly lower entropy than input grids. The test yields a p-value of exactly 1.0, indicating that we fail to reject the null hypothesis in this direction. Because of this result, we perform a two-sided Wilcoxon signed-rank test to evaluate if there is any significant difference in either direction. The two-sided test yields a test statistic of $1,500,779.0$ and a p-value of $3.0116 \times 10^{-61}$. This is an extraordinary statistical result: the null hypothesis is rejected with extreme significance, but the directional shift is the exact opposite of our initial hypothesis. Output grids possess significantly higher Shannon information entropy than input grids on average.

To confirm this, we analyze the transition density metric. A two-sided Wilcoxon signed-rank test on transition density yields a p-value of $1.4824 \times 10^{-42}$, confirming that output grids also have significantly higher edge transition densities than input grids. We group the Shannon entropy values into four cohorts, including Very Low (0.0-0.5 bits), Low (0.5-1.2 bits), Medium (1.2-2.0 bits), and High (2.0+ bits). We plot the grouped bar chart, which is saved to `entropy_distribution.png`.

The chart reveals a striking distribution: output grids have a much lower frequency in the Very Low cohort and a significantly higher frequency in the Medium and High cohorts compared to inputs. This visually demonstrates the upward shift in information entropy.

### 6.3 Interpretation and Cognitive Implications

The discovery that Shannon entropy and transition density significantly increase in output grids is one of the most critical findings of this research. It is initially counter-intuitive: if output grids represent clean, resolved states where noise is eliminated, why do they exhibit higher entropy and boundary complexity?

The explanation lies in the mathematics of Shannon entropy and the spatial cropping prior of ARC. Most ARC input grids are large matrices where the vast majority of pixels are black background. Because a single class dominates 90% or more of the grid, the probability distribution of pixel values is highly skewed, resulting in very low Shannon entropy.

When a task requires cropping to a small target region, the black background is stripped away. The resulting small output grid contains a highly diverse mix of colors with very few or no background pixels. This makes the probability distribution of pixel values much more uniform, which mathematically causes a massive increase in Shannon entropy.

Furthermore, the higher transition density indicates that the output grid has a much higher density of active color boundaries per unit area. Instead of scattered objects floating in a vast, empty black space, the output grid is a highly concentrated, dense packet of structured information.

This reveals that the abstract reasoning process in ARC does not reduce information complexity globally; rather, it condenses information. The output grid is a high-density semantic representation. For automated solvers, this suggests that maximizing information density in the active subgrid is a powerful heuristic. If a model must choose between a sparse, empty grid and a tightly bound, complex grid, the empirical data strongly favors the latter.

---

## 7. Discussion and Collaborative Interpretations

To effectively explain these structural differences to research colleagues and collaborative teams, we must synthesize these spatial, topological, and informational metrics into a coherent narrative. The structural shift from input to output in ARC can be described as a transition from sparse, low-density matrices to highly concentrated, high-density motifs. Our findings provide a clear roadmap for designing high-performance heuristic filters in ARC solvers, establishing several key structural priors.

First, we recommend prioritizing Crop-First Search Heuristics. Automated solvers that use program synthesis often struggle with the size of the search space. Because output grids have a significantly smaller mean area, solvers should prioritize cropping operations. When an input grid contains large uniform regions, search algorithms should immediately generate candidate outputs by cropping around dense, multi-colored same-color connected components.

Second, we suggest implementing Entropy-Maximization as a Selection Prior. When a solver generates multiple valid programs that produce different output sizes, the programs that output grids with higher Shannon entropy and higher transition density should be prioritized. A high Shannon entropy is highly diagnostic of a valid output grid, as it indicates the background has been successfully cropped out and the active objects are well-represented.

Third, we propose employing Connected Component Cohesion Filters. Solvers should implement a cohesion filter that penalizes candidate outputs containing excessive numbers of small, disconnected components. Since output grids exhibit a significant decline in component cardinality, valid transformations tend to merge, align, or eliminate floating noise pixels.

By incorporating these principles, collaborative research teams can dramatically reduce the search space of domain-specific language and neuro-symbolic ARC solvers, accelerating the discovery of correct programs.

---

## 8. Conclusion

In this paper, we have presented a comprehensive, quantitative comparative analysis of the structural and informational differences between input and output matrices in the Abstraction and Reasoning Corpus. By extracting multi-dimensional features across 3,500 grid pairs and testing three formal hypotheses, we have mapped out the statistical priors of the ARC benchmark.

Our results demonstrate that output grids undergo a highly significant spatial contraction, resulting in a 34.60 pixel reduction in average grid area. At the same time, they experience a significant reduction in same-color connected component cardinality, reflecting a cognitive prior of object consolidation and noise removal. Crucially, we discovered that output grids exhibit a highly significant increase in Shannon information entropy and transition density, driven by the elimination of uniform background pixels during cropping operations.

These findings reshape our understanding of the ARC benchmark, showing that abstract reasoning is mathematically characterized by informational compression and concentration. By leveraging these insights, researchers and system developers can build highly focused, object-centric solvers that align more closely with human cognitive priors, moving closer to solving the general abstract reasoning challenges of the ARC benchmark.
