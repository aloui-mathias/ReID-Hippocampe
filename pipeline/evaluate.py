from evaluators import evaluator
from config import *
from contextlib import redirect_stdout
import numpy as np
from os import (
    environ,
    listdir
)
from os.path import join
import pandas as pd
from sys import argv


SIZE = (int)(argv[1])
NAME = argv[2]
DB_PATH = environ["DB_PATH"]


with open(f'evaluate.{SIZE}{NAME}.txt', 'w') as f:
    with redirect_stdout(f):

        embeddings_train = []
        embeddings_test = []
        embeddings_new = []

        labels_test = []
        labels_train = []
        labels_new = []

        labels_profil_test = []
        labels_profil_train = []
        labels_profil_new = []

        names_test = []
        names_train = []
        names_new = []

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
                    profil = "left"
                else:
                    profil = "right"
                test = (bool)(row["test"])
                new = (bool)(row["new"])
                name = row["image"]
                if new:
                    embeddings_new.append(embedding)
                    labels_new.append(indiv)
                    labels_profil_new.append(".".join([indiv, profil]))
                    names_new.append(name)
                elif test:
                    embeddings_test.append(embedding)
                    labels_test.append(indiv)
                    labels_profil_test.append(".".join([indiv, profil]))
                    names_test.append(name)
                else:
                    embeddings_train.append(embedding)
                    labels_train.append(indiv)
                    labels_profil_train.append(".".join([indiv, profil]))
                    names_train.append(name)

        embeddings_test = np.array(embeddings_test)
        labels_test = np.array(labels_test)
        lables_profil_test = np.array(labels_profil_test)
        names_test = np .array(names_test)

        embeddings_train = np.array(embeddings_train)
        labels_train = np.array(labels_train)
        lables_profil_train = np.array(labels_profil_train)
        names_train = np .array(names_train)

        embeddings_new = np.array(embeddings_new)
        labels_new = np.array(labels_new)
        lables_profil_new = np.array(labels_profil_new)
        names_new = np .array(names_new)

        eval = evaluator(nb=10)
        eval.fit(embeddings_train, labels_train)
        eval.eval(embeddings_test, labels_test, names_test)
