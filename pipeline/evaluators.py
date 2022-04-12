import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


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
        sum_APks = np.zeros(shape=(self.nb))
        sum_APs = 0
        for label in tqdm(np.unique(y)):
            print("Label : ", label)
            indices = np.where(y == label)[0]
            x_label = x[indices]
            names = z[indices]
            gtp = len(np.where(self.y == label)[0])
            print("GTP : ", gtp)
            print(
                "Images : ",
                list(map(
                    str.encode,
                    names
                ))
            )
            neighbors = np.array(self.neighbors.kneighbors(x_label))
            all_neighbors = np.array(self.neighbors.kneighbors(
                x_label, n_neighbors=len(self.x)))
            neighbors_labels = self.y[neighbors[1, :, :].astype(int)]
            all_neighbors_labels = self.y[
                all_neighbors[1, :, :].astype(int)]
            print("Distances : ", neighbors[0])
            print("Neighbors : ", neighbors_labels)
            cmcks = []
            APks = []
            APs = []
            for labels in neighbors_labels:
                cmck = np.zeros(shape=(self.nb))
                APk = np.zeros(shape=(self.nb))
                for i in range(self.nb):
                    if label == labels[i]:
                        for j in range(i, self.nb):
                            APk[j] = 1/(i+1)
                            cmck[j] = 1
                        break
                cmcks.append(cmck)
                APks.append(APk)
            for labels in all_neighbors_labels:
                nb_found = 0
                AP = 0
                for i in range(len(self.y)):
                    if label == labels[i]:
                        nb_found += 1
                        AP += nb_found/(i+1)
                AP /= gtp
                APs.append(AP)

            print("CMC@k : ", cmcks)
            sum_cmck = np.sum(cmcks, axis=0)
            print("Mean CMC@k : ", sum_cmck/len(cmcks))
            print("AP@k : ", APks)
            sum_APk = np.sum(APks, axis=0)
            print("mAP@k : ", sum_APk/len(APks))
            print("AP : ", APs)
            sum_AP = np.sum(APs, axis=0)
            print("mAP : ", sum_AP/len(APs))
            print()

            sum_cmcks += sum_cmck
            sum_APks += sum_APk
            sum_APs += sum_AP

        mean_cmcks = sum_cmcks / len(x)
        mAPs1 = sum_APks / len(x)
        mAPs2 = sum_APs / len(x)
        print()
        print("Mean CMC@k : ", mean_cmcks)
        print("mAP@k : ", mAPs1)
        print("mAP : ", mAPs2)

        print("------------------------------")
