import os
import shutil

PATH = "/home/data/indiv"

train = []
test = []

for indiv in os.listdir(os.path.join(PATH, "224")):

    indiv_path = os.path.join(PATH, "224", indiv)

    if len(os.listdir(indiv_path)) <= 19:
        test.append(indiv)
    else:
        train.append(indiv)

for size in ["224", "240", "260", "300"]:

    path = os.path.join(PATH, size)

    os.mkdir(os.path.join(path, "train"))
    os.mkdir(os.path.join(path, "test"))

    for indiv in train:
        shutil.move(os.path.join(path, indiv),
                    os.path.join(path, "train", indiv))

    for indiv in test:
        shutil.move(os.path.join(path, indiv),
                    os.path.join(path, "test", indiv))
