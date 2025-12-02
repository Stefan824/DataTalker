1) Spherical K-Means (cosine)

Why: fast, stable on text embeddings; great for clean, globular topical groups.

Train

L2-normalize embeddings (so cosine = dot product).

Sweep 
𝐾
∈
{
8
,
12
,
16
,
20
,
…
 
}
K∈{8,12,16,20,…}; multiple random inits.

Test / Evaluate

Primary: Silhouette (cosine).

Secondary: Calinski–Harabasz (↑), Davies–Bouldin (↓).

Stability: bootstrap resamples → ARI or NMI across runs (mean ± sd).

Report: chosen 
𝐾
K, metrics, cluster size distribution, 2–3 exemplar citations per cluster.

When it wins: coherent, roughly spherical topics; you want speed + interpretability.

2) HDBSCAN (density-based, auto-K + outliers)

Why: your topics can be uneven; this finds dense groups, marks “misc” as noise.

Train

(Optional) light dimensionality reduction (e.g., SVD to 100–300) if embeddings are very high-dim.

Sweep min_cluster_size (e.g., 10–50) and min_samples (None or ~min_cluster_size).

Test / Evaluate

Primary (density): DBCV (↑).

Model-internal: cluster stability/persistence (↑) from HDBSCAN.

Practical: % noise (↓ desired, but not zero), size distribution, exemplar citations.

When it wins: variable cluster densities, real outliers, natural “other” bin without forcing K.

3) Agglomerative Clustering (average linkage, cosine)

Why: hierarchical structure; you can browse coarse → fine clusters and pick the best cut.

Train

Use cosine distances (L2-normalized embeddings).

Build hierarchy with average (or complete) linkage.

Select a cut by target 
𝐾
K or distance threshold.

Test / Evaluate

For several cut levels: Silhouette (cosine), CH, DB.

Stability: bootstrap ARI/NMI across resamples/cuts.

Report: chosen cut (or 
𝐾
K), metrics, dendrogram snapshots (optional), exemplars.

When it wins: you want a hierarchy and interpretable splits, not just a flat partition.

Minimal full-pipeline checklist (for all three)

Preprocess: one embedding per citation; L2-normalize (cosine).

Train: fit the clustering method with a small hyperparameter sweep (K or density params).

Select: choose the setting that maximizes the primary metric (Spherical K-Means/Agglo → Silhouette; HDBSCAN → DBCV + stability, reasonable %noise).

Validate: stability via ARI/NMI on 3–5 bootstraps; qualitative exemplars per cluster.

Report: hyperparams, primary/secondary metrics, stability (mean±sd), size distribution, top terms/exemplars.

Metrics：
Core internal indices (no labels needed)

Silhouette (↑ better)
use cosine on L2-normalized embeddings. Reports how well points fit their own cluster vs. nearest other cluster. Range 
[
−
1
,
1
]
[−1,1].

Calinski–Harabasz / CH (↑)
separation vs. compactness; higher is better.

Davies–Bouldin / DB (↓)
average similarity between clusters; lower is better.

Density & structure (for density methods)

DBCV (↑)
density-based validity (for HDBSCAN/DBSCAN/OPTICS). Higher is better.

% Noise (↓)
fraction of points labeled as noise/outliers (e.g., −1 in HDBSCAN).

Stability / robustness

ARI (↑) or NMI (↑) across reruns/bootstraps
run the clustering multiple times (or on bootstrap samples) and compare labelings; higher means more stable.

Practical sanity checks

Cluster size distribution (avoid extreme micro-clusters unless expected).

Exemplar inspection (nearest examples per cluster).

(Optional) Top-terms coherence: for each cluster, TF-IDF top terms should look semantically coherent.

Recommended primary metrics by algorithm
1) Spherical K-Means (cosine)

Primary: Silhouette (cosine)

Secondary: CH (↑), DB (↓)

Stability: ARI/NMI across seeds or bootstraps

2) HDBSCAN

Primary: DBCV (↑)

Model-internal: cluster stability/persistence (↑) from HDBSCAN

Practical: % noise (↓), size distribution

(Optional) Silhouette on non-noise points

3) Agglomerative (average linkage, cosine)

Primary: Silhouette (cosine) across several cut levels

Secondary: CH, DB

Stability: ARI/NMI across bootstrapped samples and cut thresholds

How to use them (short and sweet)

Normalize embeddings (L2) for cosine metrics.

Sweep hyperparams (K for k-means/agglo; min_cluster_size for HDBSCAN).

Select the setting maximizing the primary metric (or a balanced tradeoff with secondary + stability).

Report: chosen hyperparams, primary/secondary scores, stability (mean±sd), % noise (if applicable), and a few exemplars per cluster.