from config import *
from os import (
    environ,
    listdir
)
from os.path import join
from utils import (
    get_date,
    isCustom,
    isImage,
    originalname_to_cropresizename
)
from tqdm import tqdm


SIZES = environ["CROP_SIZES"].split()
DATASET_PATH = environ["DATASET_PATH"]
OUT_PATH = environ["SPLIT_DATE_PROFIL_PATH"]


for size in ["224"]:

    print(size)

    size_path = join(OUT_PATH, size)

    paths = {}

    for indiv in tqdm(listdir(DATASET_PATH)):

        indiv_path = join(size_path, indiv)

        files = {}

        for filename in listdir(join(DATASET_PATH, indiv)):

            if isImage(filename) and not isCustom(filename):
                resize = originalname_to_cropresizename(filename, size)
                date = get_date(
                    join(DATASET_PATH, indiv),
                    filename
                )
                files[str(date)] = files.get(str(date), []) + [resize]

        paths[indiv] = files

    print(paths)
