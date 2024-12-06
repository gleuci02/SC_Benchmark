from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

#def load_mnist():
#    mnist = fetch_openml('mnist_784', version=1)
#    X = StandardScaler().fit_transform(mnist.data)
#    y = mnist.target
#    return X, y

def load_mnist():
    mnist = datasets.MNIST('./data', train=True, download=True)
    X = StandardScaler().fit_transform(mnist.data.reshape(60000, 28*28).data)
    y = mnist.targets
    return X, y

def load_fashion_mnist():
    fashion_mnist = datasets.FashionMNIST('./data', train=True, download=True)
    X = StandardScaler().fit_transform(fashion_mnist.data.reshape(60000, 28*28).data)
    y = fashion_mnist.targets
    return X, y