from config import *
from functions import (
    isImage,
    originalname_to_customname
)
from os import (
    environ,
    listdir
)
from os.path import (
    isfile,
    join
)
import pandas as pd
from profil.classification import model as profilModel
from ReID.reid import model as reidModel
from tqdm import tqdm


DB_PATH = environ["DB_PATH"]
SIZES = environ["CROP_SIZES"].split(" ")


for size in SIZES:

    print(size)

    db_csv_path = join(DB_PATH, "ALL." + size + ".csv")

    db_df = {"indiv": [], "profil": [], "embedding": []}

    for indiv in tqdm(listdir(DB_PATH)):
        indiv_csv_path = join(DB_PATH, indiv, indiv + "." + size + ".csv")

        if not isfile(indiv_csv_path):
            raise Exception("DB has not been updated")
        else:
            indiv_df = pd.read_csv(indiv_csv_path)

        db_df["indiv"].append(indiv)
        db_df["profil"].append("both")

        db_df["indiv"].append(indiv)

        db_df["indiv"].append(indiv)
