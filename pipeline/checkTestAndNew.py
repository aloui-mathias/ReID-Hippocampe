from config import *
from functions import originalname_to_cropresizename
from os import (
    environ,
    listdir
)
from os.path import (
    isdir,
    join
)
from pandas import read_csv
from sys import argv
from tqdm import tqdm

if len(argv) < 2 or len(argv) > 3:
    print("Not enough or too many argv")
if len(argv) >= 2:
    SIZE = argv[1]
    NAME = None
if len(argv) == 3:
    NAME = argv[2]
DB_PATH = environ["DB_PATH"]
INDIV_PATH = environ["INDIV_PATH"]

for indiv in tqdm(listdir(DB_PATH)):
    test = listdir(join(INDIV_PATH, SIZE, "test", indiv))
    if isdir(join(INDIV_PATH, SIZE, "train", indiv)):
        new = 0
        train = listdir(
            join(INDIV_PATH, SIZE, "train", indiv))
    else:
        new = 1
        train = []
    if not NAME:
        csv_path = join(
            DB_PATH,
            indiv,
            ".".join([indiv, SIZE, "csv"])
        )
    else:
        csv_path = join(
            DB_PATH,
            indiv,
            ".".join([indiv, SIZE + NAME, "csv"])
        )
    df = read_csv(csv_path)

    for idx in df.index:
        image = df.at[idx, "image"]
        cropname = originalname_to_cropresizename(
            image,
            (int)(SIZE)
        )
        if cropname in test:
            val = 1
        elif cropname in train:
            val = 0
        else:
            raise Exception("Image not in indiv folder")
        df.at[idx, "test"] = val
        df.at[idx, "new"] = new

    df.to_csv(csv_path, index=False)
