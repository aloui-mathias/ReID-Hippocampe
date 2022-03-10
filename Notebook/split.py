from genericpath import isfile
from config import *
from os import (
    environ,
    mkdir
)
from os.path import (
    isdir,
    join
)
from numpy.random import random
from shutil import copyfile
from tqdm import tqdm
from utils import withoutExt


if not isdir(environ["SPLIT_PATH"]):
    mkdir(environ["SPLIT_PATH"])
else:
    print("split already exists")
    exit()

parts = ["train", "val", "test"]

for part in parts:
    mkdir(join(environ["SPLIT_PATH"], part))

with open("yolo_dataset.txt", "r") as trainfile:

    for line in tqdm(trainfile.read().splitlines()):

        line = line.split("/")
        indiv = line[0]
        filename = line[1]

        path = environ["SPLIT_PATH"]

        rnd = random()
        if rnd < 0.1:
            path = join(path, parts[1])
        elif rnd < 0.2:
            path = join(path, parts[2])
        else:
            path = join(path, parts[0])

        out_path = join(environ["DATASET_PATH"], indiv, filename)
        in_path = join(path, filename)

        copyfile(out_path, in_path)

        filename = withoutExt(filename) + ".txt"
        out_path = join(environ["DATASET_PATH"], indiv, filename)
        in_path = join(path, filename)

        copyfile(out_path, in_path)
