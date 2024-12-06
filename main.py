from datasets.mnist import load_mnist, load_fashion_mnist
from datasets.cifar import load_cifar10, load_cifar100
from datasets.stl import load_stl10
from algorithms.EDSC import edesc
from sklearn import cluster
from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from metrics import normalized_mutual_info_score, clustering_accuracy
from visualization import plot_metrics

ALGORITHMS = {
    #"EDESC": edesc.EDESC(
    #    n_enc_1=500,
    #    n_enc_2=500,
    #    n_enc_3=1000,
    #    n_dec_1=1000,
    #    n_dec_2=500,
    #    n_dec_3=500,
    #    n_z=20,
    #    n_input=1000,
    #    n_clusters=10,
    #    num_sample=60000),
    "kmeans": cluster.KMeans(n_clusters=10),
    "elasticnet": ElasticNetSubspaceClustering(n_clusters=10,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9),
    "Sparse Subspace Clustering OMP": SparseSubspaceClusteringOMP(n_clusters=10,affinity='symmetrize',n_nonzero=5,thr=1.0e-5),
    "Spectral Clustering": cluster.SpectralClustering(n_clusters=10,affinity='nearest_neighbors',n_neighbors=5)
}

DATASETS = {
    "STL10": load_stl10,
    "CIFAR10": load_cifar10,
    "CIFAR100": load_cifar100,
    "MNIST": load_mnist,
    "FASHION_MNIST": load_fashion_mnist,
}

def run_experiment():
    algorithm_names = []
    acc_scores = []
    nmi_scores = []

    # Load dataset
    for dataset in DATASETS:
        X, y_true = DATASETS[dataset]()
        for name in ALGORITHMS:

            # Initialize algorithm
            algorithm = ALGORITHMS[name]

            if dataset == DATASETS["MNIST"]:
                algorithm.n_clusters = 10
            elif dataset == DATASETS["CIFAR10"]:
                algorithm.n_clusters = 10
            elif dataset == DATASETS["CIFAR100"]:
                algorithm.n_clusters = 100
            elif dataset == DATASETS["STL10"]:
                algorithm.n_clusters = 10
            
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

    #dataset_loader = DATASETS["MNIST"]

    run_experiment()