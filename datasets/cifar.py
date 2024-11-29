import numpy as np
from torchvision import datasets
from sklearn.preprocessing import StandardScaler


def load_cifar10():
    cifar10 = datasets.CIFAR10('./data/cifar', train=True, download=True)
    X = StandardScaler().fit_transform(cifar10.data)
    y = cifar10.target
    return X, y

def load_cifar100():
    cifar100 = datasets.CIFAR100('./data/cifar', train=True, download=True)
    X = StandardScaler().fit_transform(cifar100.data)
    y = cifar100.target
    return X, y