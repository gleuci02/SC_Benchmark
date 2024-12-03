from sklearn import cluster
from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP


# =================================================
# Create cluster objects
# =================================================
print('Begin clustering...')

# Baseline: non-subspace clustering methods
model_kmeans = cluster.KMeans(n_clusters=10)  # k-means as baseline

model_spectral = cluster.SpectralClustering(n_clusters=10,affinity='nearest_neighbors',n_neighbors=5)  # spectral clustering as baseline

# Our work: elastic net subspace clustering (EnSC)
# You et al., Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
model_ensc = ElasticNetSubspaceClustering(n_clusters=10,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9)

# Our work: sparse subspace clusterign by orthogonal matching pursuit (SSC-OMP)
# You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=10,affinity='symmetrize',n_nonzero=5,thr=1.0e-5)

clustering_algorithms = (
    ('KMeans', model_kmeans),
    ('Spectral Clustering', model_spectral),
    ('EnSC via active support solver', model_ensc),
    ('SSC-OMP', model_ssc_omp),
)