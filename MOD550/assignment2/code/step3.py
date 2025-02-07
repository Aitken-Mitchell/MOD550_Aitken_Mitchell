import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# Load the two CSV files
df1 = pd.read_csv("step2_oscillatory_data_clean.csv")
df2 = pd.read_csv("step2_oscillatory_data_noisy.csv")

# Extract 't' and 'u' columns from both datasets
t1, u1 = df1["t"].values, df1["u"].values
t2, u2 = df2["t"].values, df2["u"].values

# Stack both datasets vertically (concatenate along axis=0)
t = np.concatenate((t1, t2), axis=0)
u = np.concatenate((u1, u2), axis=0)

# Create a single combined NumPy array with shape (N, 2) where N is total samples
data = np.column_stack((t, u))

# Apply K-means clustering
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(data)

# Compute variance for each K-Means cluster
for cluster in range(6):
    cluster_data = data[kmeans_labels == cluster]  # Select points in the cluster
    variance = np.var(cluster_data, axis=0, ddof=1)  # Compute variance for each feature (t & u)
    

# Save clustered data to CSV
clustered_df = pd.DataFrame({"t": t, "u": u, "Cluster": kmeans_labels})
clustered_df.to_csv("step3_clustered_data.csv", index=False)
print("Clustered data saved to step3_clustered_data.csv, 6 x K-Means clusters due to elbow plot.")

# Plot the clustering results
plt.figure(figsize=(12, 6))

# Original Data
plt.subplot(1, 2, 1)
plt.scatter(t, u, s=5, alpha=0.6, color='black')
plt.xlabel("Time (t)")
plt.ylabel("Amplitude (u)")
plt.title("Original Data")

# K-Means Clustering
plt.subplot(1, 2, 2)
plt.scatter(t, u, c=kmeans_labels, cmap='viridis', s=5, alpha=0.6)
plt.xlabel("Time (t)")
plt.ylabel("Amplitude (u)")
plt.title("K-Means Clustering")

plt.tight_layout()
plt.savefig("step3_k-means_clustering.png")
plt.show()

# Define the range of k values to test
k_values = range(1, 11)  # Trying k from 1 to 10
inertia_values = []  # Store variance (inertia) for each k

# Run K-Means for different k values
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data)
    inertia_values.append(kmeans.inertia_)  # Inertia is the within-cluster variance

# Plot Variance (Inertia) vs Number of Clusters (k)
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o', linestyle='--', color='b')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Variance (Inertia)")
plt.title("Variance as a Function of Number of Clusters (K-Means)")
plt.xticks(k_values)
plt.grid(True)
plt.savefig("step3_variance_vs_#clusters.png")
plt.show()
