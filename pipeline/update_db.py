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
    isdir,
    isfile,
    join
)
import pandas as pd
from profil.classification import model as profilModel
from ReID.reid import model as reidModel
from tqdm import tqdm


DB_PATH = environ["DB_PATH"]
DATASET_PATH = environ["DATASET_PATH"]
# SIZES = environ["CROP_SIZES"].split(" ")
SIZES = ["380", "456", "528"]
REMAKE = True
project = environ["TEMP_FOLDER"]

paths = {}
for indiv in listdir(DB_PATH):
    indiv_path = join(DB_PATH, indiv)

    images = []
    for filename in listdir(indiv_path):
        if isImage(filename):
            images.append(filename)

    paths[indiv] = images

profil_model = profilModel()

for size in SIZES:

    print(size)

    reid_model = reidModel(size=size)

    for indiv, images in tqdm(paths.items()):
        csv_path = join(DB_PATH, indiv, indiv + "." + size + ".csv")

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
