from sklearn.linear_model import ElasticNet

class ElasticNetSubspaceClustering():
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    def fit(self, data):
        self.model.fit(data, data)  # ElasticNet can be used for dimensionality reduction

    def predict(self, data):
        return self.model.predict(data)