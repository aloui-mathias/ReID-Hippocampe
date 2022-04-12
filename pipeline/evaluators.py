from functions import originalname_to_customname
import numpy as np
from sklearn.neighbors import NearestNeighbors


class evaluator:

    def __init__(self, nb: int = 10) -> None:
        self.nb = nb
        self.neighbors = NearestNeighbors(n_neighbors=nb,
                                          metric='euclidean')

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.neighbors.fit(x)

    def eval(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        print("------------------------------")
        print("----------   EVAL   ----------")
        print()

        sum_cmcks = np.zeros(shape=(self.nb))
        for label in np.unique(y):
            print("Label : ", label)
            indices = np.where(y == label)[0]
            x_label = x[indices]
            names = z[indices]
            print(
                "Images : ",
                list(map(
                    str.encode,
                    names
                ))
            )
            neighbors = np.array(self.neighbors.kneighbors(x_label))
            neighbors_labels = self.y[neighbors[1, :, :].astype(int)]
            print("Distances : ", neighbors[0])
            print("Neighbors : ", neighbors_labels)
            cmcks = []
            for labels in neighbors_labels:
                cmck = np.zeros(shape=(self.nb))
                for i in range(self.nb):
                    if label == labels[i]:
                        for j in range(i, self.nb):
                            cmck[j] += 1
                        break
                cmcks.append(cmck)

            print("CMC@k : ", cmcks)
            print()

            sum_cmcks += np.sum(cmcks, axis=0)

        sum_cmcks = sum_cmcks / len(x)
        print()
        print("Mean CMC@k : ", sum_cmcks)

        print("------------------------------")
