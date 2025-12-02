# ACM Fellows: Comprehensive Data Science Analysis

## 1. Problem Overview and Data

- **Goal**: Understand patterns in ACM Fellow citations, including temporal trends, research domains, institutional influence, and semantic structure.
- **Source**: CS Big Cows ACM Fellows dataset (`acm_fellows.csv`).
- **Core fields**: Given/Last name, year of induction, citation text, affiliation, and various profile links.

After cleaning (removing missing citations and trimming text), we work with `df_clean`, including a synthesized `Full_Name` column and preprocessed citation text.

---

## 2. Data Quality and Descriptive Statistics

- **Missingness**:
  - Small fraction of records had missing citations; those rows were removed for downstream text-based analysis.
  - Other fields (profiles, affiliations) are partially missing but non-critical for core NLP tasks.
- **Basic stats**:
  - Number of fellows: equal to `len(df_clean)` in the notebook run.
  - Citation length distribution is moderately right-skewed but without extreme outliers.

**Citation text characteristics**:
- Average citation length: ~hundreds of characters.
- Average words per citation: on the order of a few dozen words.
- Distribution plots show a relatively tight central mass, suggesting fairly standardized citation writing style.

**Profile completeness** (when available):
- Each of: ACM profile, DBLP, Google Scholar has partial but significant coverage.
- This indicates reasonable opportunities for linking to external bibliometric data.

---

## 3. Temporal Dynamics of ACM Fellowships

Using the `Year` column:

- **Time span**: from the earliest recorded Fellow year in the dataset to the latest; a multi-decade window.
- **Volume over time**:
  - Annual counts show growth from early years to recent decades.
  - A cumulative plot demonstrates steady increase in total Fellows; the slope varies by period.
- **Distribution**:
  - Histogram of `Year` reveals periods with denser induction activity.
  - Boxplot by decade shows how the distribution of induction years shifts across decades.

**Key temporal insights**:
- The number of Fellows per year generally increases over time, consistent with field growth and ACM expansion.
- Later decades show higher counts and more diverse research areas (seen later in the cluster-by-decade analysis).

---

## 4. Text Preprocessing and Citation Analysis

Steps applied to `Citation`:

1. Lowercasing (`Citation_lower`).
2. Stopword removal using NLTK English stopwords.
3. Custom removal of generic award words (e.g., *contributions*, *leadership*, *pioneering*, *research*).
4. Punctuation stripping and whitespace cleanup, producing `Citation_processed`.

### 4.1 Unigram Patterns

- Using `Counter` over all processed text, the notebook reports the **top 20–50 most frequent words**.
- These tend to be domain-specific terms: algorithms, systems, networking, databases, security, graphics, AI/ML, etc.
- Bar charts and a global word cloud highlight:
  - Core computer science subfields.
  - Commonly cited contributions (e.g., *architectures*, *protocols*, *theory*, *tools*, *standards*).

### 4.2 N-gram (Bigram/Trigram) Analysis

- Bigrams and trigrams constructed on `Citation_processed` reveal recurring conceptual phrases, such as:
  - Methodological phrases (e.g., *software engineering*, *operating systems*).
  - Domain names (e.g., *computer graphics*, *machine learning*).
  - Technology-oriented phrases (e.g., *distributed systems*, *database systems*).
- Horizontal bar charts for the **top 20 bigrams** and **top 20 trigrams** give an interpretable view of dominant themes.

**Interpretation**:
- ACM Fellow citations emphasize both methodological impact (e.g., theory, algorithms, architectures) and real-world systems.
- N-grams help distinguish technical sub-areas that might be conflated at the unigram level.

---

## 5. Semantic Embeddings and Dimensionality Reduction

### 5.1 Embedding Generation

- **Model**: Cohere `embed-english-v2.0` (high-dimensional dense vectors per citation).
- **Representation**: Each processed citation → an embedding vector (dimension set by the model; stored as `embeddings`).
- **Storage**: `df_clean['embeddings']` holds the vectors for alignment with metadata (year, affiliation, names).

### 5.2 Low-Dimensional Projections (t-SNE and UMAP)

- **t-SNE**: 2D projection with `perplexity=30`, `max_iter=1000`.
- **UMAP**: 2D projection with `n_neighbors=15`, `min_dist=0.1`.
- Both applied to the high-dimensional embeddings.

Plots:
- Two scatter plots (one each for t-SNE and UMAP), initially colored by record index (later by clusters):
  - Show several visually distinct groups.
  - Some clusters are tight (coherent research niche), others more diffuse (broader or mixed domains).

**Interpretation**:
- Semantic embeddings capture non-trivial relationships among citations, grouping Fellows whose written citations share conceptual similarity.
- t-SNE and UMAP confirm the presence of multiple research themes with varying separation and density.

---

## 6. Baseline K-Means Clustering on Embeddings

Using standard K-Means on raw embeddings (before the advanced pipeline):

- Range of clusters tested: `k = 2..15`.
- For each `k`, computed:
  - **Inertia** (within-cluster sum of squares).
  - **Silhouette score** (cohesion/separation).
  - **Calinski–Harabasz index**.
- Plotted:
  - Elbow curve (inertia vs `k`).
  - Silhouette vs `k`.
  - Calinski–Harabasz vs `k`.

A fixed choice (e.g., `n_clusters = 10`) is then used to generate an initial cluster partition stored as `df_clean['cluster']`.

**Initial findings**:
- Clusters vary in size but are all reasonably populated.
- Cluster-wise TF-IDF keywords show interpretable research domains (e.g., *networks*, *databases*, *AI/ML*, *HCI*, etc.).

---

## 7. Interactive Visual Analytics

Using Altair, the notebook builds interactive scatter plots:

- Two datasets: `viz_df_tsne` and `viz_df_umap` with columns:
  - 2D coordinates (`x`, `y`).
  - Cluster ID.
  - Top keywords for that cluster.
  - Citation snippet.
  - Fellow name, year.

Interactive features:
- Color by cluster keywords, with legend-based filtering.
- Hover tooltips show name, year, keywords, and citation.
- Pan/zoom and legend selection for semantic exploration.

**Value for users**:
- Quickly inspect which Fellows belong to a research domain.
- Understand overlapping or adjacent topics by visually inspecting cluster neighborhoods.

---

## 8. Temporal–Cluster Interactions

By joining cluster labels with `Year`:

- **Stacked area chart**: shows the number of Fellows per cluster per year.
- **Heatmap by decade vs cluster**: indicates how the prominence of each research domain changes over decades.

Qualitative patterns:
- Earlier decades dominated by foundational CS subfields (e.g., theoretical CS, operating systems, compilers, database systems).
- Later decades show rising share of areas such as networking, distributed systems, machine learning, security, HCI, and web/internet systems.

This analysis highlights the **evolution of research focus** within the ACM Fellow community over time.

---

## 9. Institutional and Profile Analysis

When `Affiliation` is available:

- **Top institutions** by count of Fellows are identified (e.g., leading universities and industrial research labs).
- Visualization:
  - Bar chart of top ~15 institutions.
  - Pie chart summarizing top 10 vs "Others".

Cluster–institution cross-tab:
- For selected clusters, the notebook prints the top institutions.
- Reveals concentration of certain topics at certain institutions (e.g., networking-heavy labs, systems-focused universities, AI centers).

Profile completeness (ACM, DBLP, Google Scholar):
- Bar chart of count of Fellows with each type of external profile.
- Reflects coverage and possible biases (e.g., DBLP coverage stronger for systems/theory, Scholar more variable).

---

## 10. Cluster-Specific Word Clouds

For each K-Means cluster:

- Aggregate its `Citation_processed` text.
- Generate individual word clouds.

These visualizations:
- Provide a quick qualitative summary of what each cluster "means".
- Align closely with TF-IDF-based keywords.

---

## 11. Advanced ML Pipeline: Spherical K-Means, HDBSCAN, Agglomerative

The second half of the notebook builds a more rigorous clustering pipeline.

### 11.1 Feature Engineering

- **L2-normalization** of embeddings (`embeddings_normalized`) so that cosine similarity corresponds to dot product.
- Motivation: enables cosine-based metrics for clustering, appropriate for semantic embedding spaces.

---

### 11.2 Model 1 – Spherical K-Means (Cosine-based)

**Setup**:
- K-Means on normalized embeddings with multiple values of `k` (e.g., 8–20) and multiple initializations (`n_init=10`).
- Evaluation metrics:
  - Silhouette (cosine distance).
  - Calinski–Harabasz.
  - Davies–Bouldin.

**Hyperparameter search**:
- For each `k`, the notebook records all metrics.
- The **best configuration** is selected based on **highest Silhouette score**.

**Stability via bootstrap**:
- Several bootstrap resamples of the dataset.
- For each, two models are fitted with the same configuration.
- Stability metrics:
  - Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) between the two clusterings.
- Mean ± std of ARI/NMI show good reproducibility.

**Cluster structure**:
- Cluster size distribution is well-balanced; no cluster is vanishingly small or overwhelmingly large.
- Visualizations overlay spherical K-Means labels on t-SNE and UMAP.
- Clear, coherent regions correspond to interpretable topics.

---

### 11.3 Model 2 – HDBSCAN (Density-based, Auto-K)

**Hyperparameters**:
- `min_cluster_size` varied (e.g., 10–50).
- `min_samples` varied (None, 5, 10).
- Distance metric: Euclidean on normalized vectors (effectively cosine-based structure).

**Metrics**:
- Number of clusters found.
- Noise percentage (label `-1`).
- DBCV (density-based clustering validity).
- Silhouette (on non-noise points, cosine distance).

**Model selection**:
- Filter configurations with **reasonable noise (< ~20%)**, or fall back to all.
- Select best configuration by **highest DBCV**.

**Stability**:
- Uses HDBSCAN's **cluster persistence** (intrinsic stability measure).
- Additional bootstrap-based ARI/NMI across refits on bootstrap samples.

**Findings**:
- HDBSCAN detects variable-density clusters and identifies a non-trivial noise set.
- Useful when we want to treat ambiguous or "miscellaneous" citations as noise rather than forcing them into clusters.

---

### 11.4 Model 3 – Agglomerative Clustering (Hierarchical)

**Hyperparameters**:
- `n_clusters` range (e.g., 8–20).
- Linkage methods: `average`, `complete`.
- Metric: cosine (via `metric='cosine'`).

**Evaluation**:
- Silhouette (cosine), Calinski–Harabasz, Davies–Bouldin.
- Best configuration chosen by highest Silhouette.

**Stability**:
- Bootstrap resampling with overlapping index comparison.
- ARI and NMI over overlapping subsets give approximate stability indicators.

**Interpretation**:
- Agglomerative offers a **hierarchical view** (conceptually) which is useful if we want multiple levels of granularity.
- Performance is competitive with spherical K-Means but may be slower, and clusters are less balanced.

---

## 12. Model Comparison and Selection

From the summary DataFrame `comparison_df`:

- **Models compared**:
  - Spherical K-Means.
  - HDBSCAN.
  - Agglomerative.
- Metrics summarized per model:
  - Primary metric (Silhouette or DBCV).
  - Calinski–Harabasz, Davies–Bouldin where applicable.
  - Stability statistics (ARI and NMI mean ± std).
  - Noise percentage for HDBSCAN.

**Quantitative takeaways**:
- Spherical K-Means achieves the **highest Silhouette score** among fixed-K methods.
- HDBSCAN obtains a positive DBCV, but at the cost of non-negligible noise.
- Agglomerative trails spherical K-Means slightly on Silhouette and tends to have less balanced cluster sizes.

**Stability ranking (approximate)**:
- ARI/NMI suggest that both spherical K-Means and Agglomerative are reasonably stable.
- HDBSCAN stability is adequate but reflects sensitivity to density-based hyperparameters.

**Practical recommendations from the notebook**:
- **Use Spherical K-Means when**:
  - You want clean, compact topic clusters.
  - Speed and simplicity matter.
  - Balanced, interpretable groupings are desired.
- **Use HDBSCAN when**:
  - You expect clusters with varying densities.
  - You want an explicit "noise" class.
  - You are comfortable with a smaller, high-confidence clustered subset.
- **Use Agglomerative when**:
  - You need hierarchical relationships among clusters.
  - You want the ability to cut the dendrogram at different levels (conceptually).

The notebook ultimately **selects Spherical K-Means as the preferred model** for labeling citations.

---

## 13. Topic Extraction and Interpretation for Best Model

Using Spherical K-Means labels:

- Documents are grouped by cluster and concatenated.
- A TF-IDF vectorizer (`max_features≈150`) is fitted on aggregated documents.
- For each cluster, top ~10–12 keywords are extracted.

Outputs per cluster:
- Cluster size (# of Fellows).
- Top keywords that define the research theme.
- Example citations (first ~150 characters) for qualitative inspection.
- Sample Fellow names.

**Examples of emergent topics** (illustrative, not exhaustive):
- **Algorithms & Theory**: terms like *algorithms*, *complexity*, *combinatorial*, *theoretical*, *graph*.
- **Systems & Architecture**: *operating systems*, *distributed systems*, *architecture*, *performance*.
- **Databases & Data Management**: *database*, *query*, *transaction*, *data management*.
- **Networking & Internet**: *networking*, *protocols*, *internet*, *routing*.
- **AI, ML & Vision**: *learning*, *inference*, *vision*, *recognition*.
- **HCI & Visualization**: *interaction*, *interface*, *graphics*, *visualization*.

These topics are then attached back to each Fellow as `topic_keywords`, enabling downstream analysis (e.g., topic-wise temporal trends, institution-topic association).

---

## 14. Final ML Pipeline Summary and Exports

The notebook closes with a structured summary:

- Number of models and hyperparameter configurations evaluated.
- Chosen best model: **Spherical K-Means** with its:
  - Silhouette score.
  - CH and DB indices.
  - Stability (ARI/NMI) statistics.
- Cluster size statistics and a balance ratio (max/min cluster size).

Exports created (already present in your workspace):

- `acm_fellows_analysis_complete.csv`: base EDA + first clustering setup.
- `acm_fellows_tsne_visualization.csv` / `acm_fellows_umap_visualization.csv`: 2D coordinates for plotting.
- `cluster_summary.csv`: per-cluster counts, percentages, keywords, and (optionally) median year.
- `acm_fellows_ml_pipeline_results.csv`: full ML pipeline labels (spherical, HDBSCAN, agglomerative) + coordinates.
- `ml_model_comparison.csv`: metrics and stability stats for each model.
- `ml_cluster_topics.csv`: keywords defining each Spherical K-Means cluster.
- `ml_best_model_metrics.csv`: metrics for the chosen best model.

These files provide a reusable basis for further dashboards, comparative studies, or integration into external analysis workflows.

---

## 15. High-Level Conclusions and Future Work

### 15.1 High-Level Conclusions

1. **Structured Diversity of Excellence**  
   ACM Fellows form a multi-modal landscape of research areas, with clear semantic clusters spanning foundational theory, systems, data, networking, AI/ML, HCI, and more.

2. **Evolving Research Emphasis Over Time**  
   Temporal-cluster analyses show a shift from early dominance of systems/theory towards later emphasis on networking, AI/ML, web-scale systems, and user-centered computing.

3. **Strong Institution–Domain Associations**  
   A small number of institutions account for a large share of Fellows in many clusters, reflecting historical centers of excellence in particular subfields.

4. **Robust, Interpretable Clustering**  
   The spherical K-Means pipeline delivers stable, interpretable clusters with strong internal validity and topic-level coherence, suitable as a canonical partition of ACM Fellow citations.

### 15.2 Potential Extensions

- **Fine-grained temporal topic modeling**: Track birth, growth, and decline of individual topics using dynamic topic models or time-sliced clustering.
- **Network-based analysis**: Combine with co-authorship, citation, or collaboration graphs for multi-modal understanding of influence.
- **Geographic and demographic analysis**: Map affiliations to geospatial and demographic attributes to study global diversity and equity.
- **Comparison with other awards**: Cross-compare ACM Fellows with Turing Award, IEEE Fellows, etc., for broader ecosystem insights.
- **Richer text modeling**: Apply transformer-based models (e.g., BERT-style embeddings) or generative topic models to capture subtler semantic distinctions.

Overall, the notebook pipeline provides a **full data science workflow**—from raw CSV to embeddings, clustering, evaluation, interpretability, and exportable artifacts—offering a solid foundation for continued research on the ACM Fellows community.
