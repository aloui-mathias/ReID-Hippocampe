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
from sys import argv
from tqdm import tqdm


DB_PATH = environ["DB_PATH"]
DATASET_PATH = environ["DATASET_PATH"]
if len(argv) > 3 or len(argv) == 1:
    print("Not enough or too many argv")
    exit()
if len(argv) >= 2:
    SIZE = (int)(argv[1])
if len(argv) == 3:
    NAME = argv[2]
else:
    NAME = None
REMAKE = True

paths = {}
for indiv in listdir(DB_PATH):
    indiv_path = join(DB_PATH, indiv)

    if isfile(indiv_path):
        continue

    images = []
    for filename in listdir(indiv_path):
        if isImage(filename):
            images.append(filename)

    paths[indiv] = images

profil_model = profilModel()

reid_model = reidModel(size=SIZE, name=NAME)

for indiv, images in tqdm(paths.items()):
    if not NAME:
        csv_path = join(DB_PATH, indiv, indiv + "." + str(SIZE) + ".csv")
    else:
        csv_path = join(DB_PATH, indiv, indiv + "." +
                        str(SIZE) + NAME + ".csv")

    if not isfile(csv_path):
        df = pd.DataFrame(columns=["image", "embedding"])
    else:
        df = pd.read_csv(csv_path)

    new_images = []
    new_paths = []
    for image in images:
        if image not in df["image"] or REMAKE:
            new_images.append(image)
            crop_name = originalname_to_customname(image)
            crop_name += ".crop.jpg"
            new_paths.append(join(DATASET_PATH, indiv, crop_name))

    new_embeddings = reid_model.predict(new_paths)
    new_profils = profil_model.predict(new_paths)

    if len(new_images) != 0 or REMAKE:
        new_df = pd.DataFrame(
            {
                "image": new_images,
                "embedding": new_embeddings,
                "profil": new_profils
            }
        )
        if REMAKE:
            df = new_df
        else:
            df = pd.concat([df, new_df])
        df.to_csv(csv_path, index=False)
