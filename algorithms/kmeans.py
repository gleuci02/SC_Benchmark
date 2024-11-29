from sklearn.cluster import KMeans

class KMeansAlgorithm():
    def __init__(self, n_clusters=10, random_state=42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)