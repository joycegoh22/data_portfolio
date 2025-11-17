## --- Import packages --- 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.clustering import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Suppress warnings 
sc.setLogLevel("ERROR")


## --- Load data ---
lines_rdd = sc.textFile("space.dat") 
rdd = lines_rdd.map(lambda line: [float(x) for x in line.split(",")]) # Split each line into 6 variables (6D)
rdd.take(5) # View first 5 lines to ensure data is correctly split 


## --- Convert the rdd into vectors for ML ---
vec_rdd = rdd.map(lambda x: Vectors.dense(x))
vec_rdd.take(5)


## --- Basic EDA ---
vec_rdd.count() # view number of lines
stats = Statistics.colStats(vec_rdd) 
print("mean  :", stats.mean())
print("var   :", stats.variance())     # population variance
print("nnz   :", stats.numNonzeros())

corr = Statistics.corr(vec_rdd) # correlation matrix 
print(corr)


## --- PCA ---

# Center the data first 
mean_vec = np.array(vec_rdd.mean()) 
centered_rdd = vec_rdd.map(lambda v: v - mean_vec)

mat = RowMatrix(centered_rdd)

svd = mat.computeSVD(6, computeU=False)

s = np.array(svd.s)
explained = (s**2) / np.sum(s**2)
print("Explained variance per PC:", explained)
print("Cumulative:", np.cumsum(explained))

# Extract the first 2 PCs, since they explain 99% of the variance 
V = svd.V.toArray()[:, :2]   # first 2 PCs
proj_rdd = rdd.map(lambda x: np.dot(np.array(x), V))

# Plot raw PCA scatter with a sample of the data points 
sample = proj_rdd.sample(False, 0.15).collect()
arr = np.array(sample)
df = pd.DataFrame(arr, columns=["PC1", "PC2"])

plt.figure(figsize=(6,6))
plt.scatter(df["PC1"], df["PC2"], s=8, alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection (PC1 vs PC2)")
plt.grid(True)
plt.savefig("pca_scatter.png")


## --- k-means clustering algorithm ---
proj_vec_rdd = proj_rdd.map(lambda x: Vectors.dense(x)) # turn PCA projections into Vectors

# Based on the raw PCA plot, we predict that the ideal k value is 4
# However, we will try k = 3 to 8 to validate and ensure clusters are correctly identified 

# Plot elbow plot to determine the ideal k value 
k_values = [3, 4, 5, 6, 7, 8]
wssses = []

def run_kmeans(k):
    # Run k-means on the RDD of feature vectors
    model = KMeans.train(proj_vec_rdd, k, maxIterations=20, initializationMode="k-means||", seed=42)

    # Compute WSSSE (Within Set Sum of Squared Errors)
    wssse = proj_vec_rdd.map(lambda point: (
        (Vectors.squared_distance(point, model.centers[model.predict(point)]))
    )).sum()

    # Compute cluster sizes
    cluster_assignments = proj_vec_rdd.map(lambda x: model.predict(x))
    sizes = cluster_assignments.countByValue()  # returns dict {cluster_id: count}

    return wssse, dict(sizes)

results = {}
for k in k_values:
    wssse, sizes = run_kmeans(k)
    results[k] = {"wssse": wssse, "sizes": sizes}
    wssses.append(wssse)

print("Summary:", results)

# Elbow plot 
plt.figure(figsize=(6,4))
plt.plot(k_values, wssses, marker='o')
plt.xticks(k_values)
plt.xlabel("k (number of clusters)")
plt.ylabel("WSSSE")
plt.title("Elbow Plot for KMeans Clustering (PC1–PC2)")
plt.grid(True)
plt.tight_layout()
plt.savefig("kmeans_elbow.png")
plt.close()

# Based on the elbow plot, k = 4 is indeed the ideal k value
# Re-plot PCA using k-means (k = 4) 

k = 4
model = KMeans.train(proj_vec_rdd, k, maxIterations=50) 
clustered = proj_rdd.map(lambda x: (x, model.predict(Vectors.dense(x)))) 
sample = clustered.sample(False, 0.15).collect()
arr = np.array([p for p, _ in sample]) 
labels = np.array([c for _, c in sample])

plt.figure(figsize=(6,6)) 
plt.scatter(arr[:,0], arr[:,1], c=labels, cmap="tab10", s=8) 
plt.xlabel("PC1"); plt.ylabel("PC2") 
plt.title("KMeans Clusters in PC1–PC2 Space (k = 4)") 
plt.grid(True) 
plt.savefig("pca_clusters.png")


## --- PCA within each cluster ---

# split the original 6D data into clusters using a cluster id 
clustered_6d = rdd.map(lambda x: (x, model.predict(Vectors.dense(np.dot(np.array(x), V)))))
clustered_6d = clustered_6d.cache()

cluster0 = clustered_6d.filter(lambda t: t[1] == 0).map(lambda t: t[0])
cluster1 = clustered_6d.filter(lambda t: t[1] == 1).map(lambda t: t[0])
cluster2 = clustered_6d.filter(lambda t: t[1] == 2).map(lambda t: t[0])
cluster3 = clustered_6d.filter(lambda t: t[1] == 3).map(lambda t: t[0])

# quick summary of the clusters 
k = 4
for c in range(k):
    pts = clustered_6d.filter(lambda t: t[1] == c).map(lambda t: t[0])
    print(f"Cluster {c} size:", pts.count())
    print("Example points:", pts.take(3))

# put cluster points into vector form for PCA 
cluster0_vecs = cluster0.map(lambda x: Vectors.dense(x))
cluster1_vecs = cluster1.map(lambda x: Vectors.dense(x))
cluster2_vecs = cluster2.map(lambda x: Vectors.dense(x))
cluster3_vecs = cluster3.map(lambda x: Vectors.dense(x))

clusters = [
    ("0", cluster0_vecs),
    ("1", cluster1_vecs),
    ("2", cluster2_vecs),
    ("3", cluster3_vecs),
]

# PCA plot for each cluster in 2D using a fraction of the data 
def pca_plot_2d_total(vec_rdd, cid, sample_frac=0.2):
    n = vec_rdd.count()
    if n == 0:
        print(f"Cluster {cid}: empty")
        return
    d = len(vec_rdd.first())

    # center
    mu = np.array(Statistics.colStats(vec_rdd).mean(), dtype=float)
    centered = vec_rdd.map(lambda v: Vectors.dense(np.array(v.toArray(), dtype=float) - mu)).cache()

    # full SVD to get total variance 
    k_full = max(1, min(d, n - 1))
    mat = RowMatrix(centered)
    svd_full = mat.computeSVD(k_full, computeU=False)

    # explained variance ratios w.r.t. total variance
    s_full = np.array(svd_full.s, dtype=float)
    total_var = (s_full**2).sum()
    Vfull = svd_full.V.toArray()
    if Vfull.shape[0] != d and Vfull.shape[1] == d:
        Vfull = Vfull.T

    # top-2 for plotting
    V2 = Vfull[:, :2]
    expl_total = (s_full[:2]**2) / total_var  # share of total 6D variance captured by PC1 and PC2

    # project sample and plot
    proj = centered.map(lambda v: np.dot(np.array(v.toArray(), dtype=float), V2))
    sample = proj.sample(False, sample_frac, seed=42).collect()
    if not sample:
        print(f"Cluster {cid}: no sample to plot")
        return
    A = np.array(sample, dtype=float)

    plt.figure(figsize=(6,6))
    plt.scatter(A[:,0], A[:,1], s=8, alpha=0.6)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"Cluster {cid} PCA 2D\n(PC1={expl_total[0]:.2%}, PC2={expl_total[1]:.2%} of total)")
    plt.grid(True); plt.tight_layout()
    out = f"cluster_{cid}_pca_2d.png"
    plt.savefig(out); plt.close()

    print(f"Cluster {cid}: n={n}, PC1={expl_total[0]:.4f}, PC2={expl_total[1]:.4f} (of total)")

# PCA plot for each cluster in 3D using a fraction of the data 
def pca_plot_3d_total(vec_rdd, cid, sample_frac=0.2):
    n = vec_rdd.count()
    if n == 0:
        print(f"Cluster {cid}: empty")
        return
    d = len(vec_rdd.first())

    mu = np.array(Statistics.colStats(vec_rdd).mean(), dtype=float)
    centered = vec_rdd.map(lambda v: Vectors.dense(np.array(v.toArray(), dtype=float) - mu)).cache()

    k_full = max(1, min(d, n - 1))
    mat = RowMatrix(centered)
    svd_full = mat.computeSVD(k_full, computeU=False)
    s_full = np.array(svd_full.s, dtype=float)
    total_var = (s_full**2).sum()

    Vfull = svd_full.V.toArray()
    if Vfull.shape[0] != d and Vfull.shape[1] == d:
        Vfull = Vfull.T
    if Vfull.shape[1] < 3 or k_full < 3:
        print(f"Cluster {cid}: fewer than 3 PCs available; skipping 3D.")
        return

    V3 = Vfull[:, :3]
    expl_total = (s_full[:3]**2) / total_var  # share of total 6D variance

    proj = centered.map(lambda v: np.dot(np.array(v.toArray(), dtype=float), V3))
    sample = proj.sample(False, sample_frac, seed=42).collect()
    if not sample:
        print(f"Cluster {cid}: no sample to plot")
        return
    A = np.array(sample, dtype=float)

    # PC axis lengths (min–max ranges) in PC space
    mins, maxs = A.min(axis=0), A.max(axis=0)
    lengths = maxs - mins
    print(f"Cluster {cid}: n={n}, PC1–3 (of total) = {np.round(expl_total, 4)}; "
          f"lengths = {np.round(lengths, 3)}")

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(A[:,0], A[:,1], A[:,2], s=8, alpha=0.6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"Cluster {cid} PCA 3D\n(PC1={expl_total[0]:.2%}, PC2={expl_total[1]:.2%}, PC3={expl_total[2]:.2%} of total)")
    plt.tight_layout()
    out = f"cluster_{cid}_pca_3d.png"
    plt.savefig(out); plt.close()

# plot PCA (in 2D and 3D)for each cluster 
for cid, vecs in clusters:
    pca_plot_2d_total(vecs, cid, sample_frac=0.2) # 2D with % of total
    
for cid, vecs in clusters: 
    pca_plot_3d_total(vecs, cid, sample_frac=0.2)  # 3D with % of total


# define function to determine the number of PCs that explain > 95% of the variance 
def cluster_pca_spectrum(vec_rdd, k_full=6, var_thresh=0.95):
    n = vec_rdd.count()
    mu = np.array(Statistics.colStats(vec_rdd).mean())
    centered = vec_rdd.map(lambda v: Vectors.dense(np.array(v) - mu))
    mat = RowMatrix(centered)
    svd = mat.computeSVD(k_full, computeU=False)
    s = np.array(svd.s, dtype=float)
    expl = (s**2) / np.sum(s**2)
    cum = np.cumsum(expl)
    d_hat = int(np.searchsorted(cum, var_thresh) + 1)
    return {"n": n, "explained": expl, "cum": cum, "d_hat": d_hat}

# table of results 
rows = []
for cid, vecs in clusters:
    res = cluster_pca_spectrum(vecs)
    rows.append({
        "cluster": cid,
        "n": res["n"],
        "d_hat (>=95%)": res["d_hat"],
        "PC1 %": round(float(res["explained"][0])*100, 2),
        "PC2 %": round(float(res["explained"][1])*100, 2),
        "PC1+PC2 %": round(float(res["explained"][0]+res["explained"][1])*100, 2),
    })

pca_df = pd.DataFrame(rows).sort_values("cluster")
print("\n=== PCA Shape Stats Table ===")
print(pca_df.to_string(index=False))


## --- Per-cluster geometry core statistics --- 

def cluster_core_stats(vec_rdd, k_full=6):
    n = vec_rdd.count()
    stats = Statistics.colStats(vec_rdd)
    mu = np.array(stats.mean())                    # 6D center
    centered = vec_rdd.map(lambda v: Vectors.dense(np.array(v)-mu))

    # Distances to center (for sphere test / radius)
    dists = centered.map(lambda v: float(np.linalg.norm(np.array(v))))
    # mean/std for radius
    dc = dists.stats()              
    rad_mean, rad_std = dc.mean(), math.sqrt(dc.variance())

    return {"n": n, "center": mu, "rad_mean": rad_mean, "rad_std": rad_std}

geo_rows = []

for cid, vecs in clusters:
    core = cluster_core_stats(vecs)

    geo_rows.append({
        "cluster": cid,
        "points": core["n"],
        "center": core["center"],
        "rad_mean": round(core["rad_mean"], 4),
        "rad_std": round(core["rad_std"], 4),
        "rad_std/mean": round(core["rad_std"] / core["rad_mean"], 4)
    })

geo_df = pd.DataFrame(geo_rows)
print("\n=== Geometry Stats Table ===")
print(geo_df.to_string(index=False))

