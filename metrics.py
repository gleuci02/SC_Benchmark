from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import _supervised

def clustering_accuracy(labels_true, labels_pred):
    """Clustering Accuracy between two clusterings.
    Clustering Accuracy is a measure of the similarity between two labels of
    the same data. Assume that both labels_true and labels_pred contain n 
    distinct labels. Clustering Accuracy is the maximum accuracy over all
    possible permutations of the labels, i.e.
    \max_{\sigma} \sum_i labels_true[i] == \sigma(labels_pred[i])
    where \sigma is a mapping from the set of unique labels of labels_pred to
    the set of unique labels of labels_true. Clustering accuracy is one if 
    and only if there is a permutation of the labels such that there is an
    exact match
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.
    
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    labels_pred : array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    
    Returns
    -------
    accuracy : float
       return clustering accuracy in the range of [0, 1]
    """
    labels_true, labels_pred = _supervised.check_clusterings(labels_true, labels_pred)
    # value = _supervised.contingency_matrix(labels_true, labels_pred, sparse=False)
    value = _supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)

def normalized_mutual_information(y_true, y_pred):
    """
    Calculate Normalized Mutual Information (NMI).

    Parameters:
    - y_true: array-like, Ground truth labels.
    - y_pred: array-like, Predicted cluster labels.

    Returns:
    - nmi: float, Normalized mutual information score.
    """
    return normalized_mutual_info_score(y_true, y_pred)