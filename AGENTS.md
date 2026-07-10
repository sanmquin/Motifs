# Agent Guidelines

This repository follows a strict set of guidelines for research, development, and documentation to ensure consistency and scientific rigor.

## Notebook Standards

All Jupyter Notebooks (`.ipynb`) must adhere to the following structure:
1.  **Hypothesis as Title**: Each major analysis section must be titled with the hypothesis it is testing.
2.  **Methodology**: A dedicated markdown cell explaining the approach, data processing, and metrics used.
3.  **Hypotheses**: Explicit statement of the null and alternative hypotheses.
4.  **Results**: Visualization (charts, plots) and quantitative metrics (R2, MAE, p-values, correlations).
5.  **Interpretation**: Guidance on how to read the results and what they mean for the hypothesis.

### Technical Requirements
-   **Colab Compatibility**: Include `google.colab.drive.mount` and handle environment-specific library installations (e.g., specific versions to avoid NumPy conflicts).
-   **Programmatic Generation**: To avoid JSON syntax errors, notebooks should be generated or modified using Python scripts rather than manual text editing.
-   **Data Paths**: Use standardized paths within `/content/drive/MyDrive/numeric_inference_outputs/` for exports.

## Documentation and Research Papers

-   **Style**: Research papers must follow a formal narrative academic style. Avoid bullet points in the main body.
-   **Length**: Target 3,000 to 3,500 words for full research papers.
-   **Naming**: Follow the established numbering system for notebooks (e.g., `0.Data_cleaning.ipynb`, `1.Pipeline.ipynb`) to indicate sequence.

## Coding Practices

-   **Efficiency**: Batch API requests (e.g., YouTube API `videos.list` in groups of 50).
-   **Security**: Always use `google.colab.userdata.get()` for API keys and credentials.
-   **Caching**: Implement persistent caching for expensive operations (embeddings, LLM responses) using JSON files in Google Drive.
-   **Dependencies**: Explicitly install necessary libraries like `sentence-transformers` or `statsmodels` at the start of the notebook. Avoid `-U` flags on core libraries like `numpy` to maintain Colab compatibility.
