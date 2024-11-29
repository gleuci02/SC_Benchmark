import numpy as np
from torchvision import datasets
from sklearn.preprocessing import StandardScaler


def load_stl10():
    stl10 = datasets.STL10('./data/stl', train=True, download=True)
    X = StandardScaler().fit_transform(stl10.data)
    y = stl10.target
    return X, y