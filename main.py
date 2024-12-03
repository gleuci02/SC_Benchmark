from datasets.mnist import load_mnist
from datasets.cifar import load_cifar10, load_cifar100
from datasets.stl import load_stl10
from algorithms.kmeans import KMeansAlgorithm
from algorithms.EDSC import EDESC
from sklearn import cluster
from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from metrics import normalized_mutual_info_score, clustering_accuracy
from visualization import plot_metrics


ALGORITHMS = {
    "kmeans": KMeansAlgorithm(n_clusters=10),
    "elasticnet": ElasticNetSubspaceClustering(n_clusters=10,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9),
    "Sparse Subspace Clustering OMP": SparseSubspaceClusteringOMP(n_clusters=10,affinity='symmetrize',n_nonzero=5,thr=1.0e-5),
    "Spectral Clustering": cluster.SpectralClustering(n_clusters=10,affinity='nearest_neighbors',n_neighbors=5),
    "EDSC": EDESC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=1000,
        n_dec_1=1000,
        n_dec_2=500,
        n_dec_3=500,
        n_z=20,
        n_clusters=4)
}

DATASETS = {
    "MNIST": load_mnist,
    "CIFAR10": load_cifar10,
    "CIFAR100": load_cifar100,
    "STL10": load_stl10,
}

def run_experiment(dataset):

    algorithm_names = []
    acc_scores = []
    nmi_scores = []

    # Load dataset
    X, y_true = DATASETS[dataset]()


    for name in ALGORITHMS:
        # Initialize algorithm
        algo_class = ALGORITHMS[name]
        algorithm = algo_class
    
        # Fit and predict
        algorithm.fit(X)
        y_pred = algorithm.predict(X)

        print(f"dataset used: {dataset}")
        print(f"Results for {name}:")
        print("NMI:")
        nmi = normalized_mutual_info_score(y_true, y_pred)
        print(nmi)
        print("ACC:")
        acc = clustering_accuracy(y_true, y_pred)
        print(acc)

        # Store results
        algorithm_names.append(name)
        acc_scores.append(acc)
        nmi_scores.append(nmi)

    plot_metrics(algorithm_names, acc_scores, nmi_scores, dataset)

if __name__ == "__main__":

    dataset_loader = DATASETS["MNIST"]

    run_experiment("MNIST")
