from evaluators import evaluator
from config import *
import numpy as np
from os import (
    environ,
    listdir
)
from os.path import join
import pandas as pd
from sys import argv


SIZE = (int)(argv[1])
DB_PATH = environ["DB_PATH"]

embeddings_train = []
embeddings_test = []
labels_test = []
labels_train = []
lables_profil_test = []
lables_profil_train = []

for indiv in listdir(DB_PATH):
    csv_path = join(
        DB_PATH, indiv,
        ".".join([indiv, str(SIZE), "csv"])
    )
    df = pd.read_csv(
        csv_path,
        dtype={"image": str, "profil": float},
        converters={
            "embedding": lambda x:
                list(map(
                    float,
                    x.strip("[]").replace("\n", '').split()
                ))
        }
    )

    for index, row in df.iterrows():
        embedding = row["embedding"]
        if row["profil"] < 0.5:
            key = "left"
        else:
            key = "right"
        test = not (bool)(row["test"])
        if test:
            embeddings_test.append(embedding)
            labels_test.append(indiv)
            lables_profil_test.append(".".join([indiv, key]))
        else:
            embeddings_train.append(embedding)
            labels_train.append(indiv)
            lables_profil_train.append(".".join([indiv, key]))


embeddings_test = np.array(embeddings_test)
labels_test = np.array(labels_test)
lables_profil_test = np.array(lables_profil_test)
embeddings_train = np.array(embeddings_train)
labels_train = np.array(labels_train)
lables_profil_train = np.array(lables_profil_train)

eval = evaluator()
eval.fit(embeddings_train, labels_train)
eval.cmc(embeddings_test, labels_test)
