import numpy as np
from sklearn.neighbors import NearestNeighbors


class evaluator:

    def __init__(self) -> None:
        self.neighbors = NearestNeighbors(n_neighbors=10,
                                          metric='euclidean')

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.neighbors.fit(x)

    def cmc(self, x: np.ndarray, y: np.ndarray):
        for label in np.unique(y):
            print("Label : ", label)
            indices = np.where(y == label)[0]
            x_label = x[indices]
            neighbors = np.array(self.neighbors.kneighbors(x_label))
            neighbors_labels = self.y[neighbors[1, :, :].astype(int)]
            print("Distances : ", neighbors[0])
            print("Neighbors : ", neighbors_labels)
            cmc = []
            for labels in neighbors_labels:
                cmc.append(label in labels)
            print("CMC : ", cmc)
