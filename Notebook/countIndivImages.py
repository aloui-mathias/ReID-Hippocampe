from config import *
from numpy import sort
from os import (
    environ,
    listdir
)
from os.path import join


ds_path = environ["DATASET_PATH"]

counts = {}

for indiv in listdir(ds_path):
    indiv_count = 0
    for filename in listdir(join(ds_path, indiv)):
        if filename.split(".")[-1] == "txt":
            indiv_count += 1
    counts[indiv_count] = counts.get(indiv_count, 0) + 1

keys = list(counts.keys())
keys = sort(keys)

tt_indiv = 0
tt_images = 0

print(",".join(["nbImages", "nbIndiv", "sumIndiv", "sumImages"]))

for key in keys:
    tt_indiv += counts[key]
    tt_images += counts[key]*key
    parts = list(map(str, [key, counts[key], tt_indiv, tt_images]))
    print(",".join(parts))
