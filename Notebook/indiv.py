from config import *
from os import (
    environ,
    listdir,
    mkdir,
)
from os.path import (
    isdir,
    isfile,
    join
)
from shutil import copyfile
from utils import (
    isImage,
    isResize
)


if not isdir(environ["INDIV_PATH"]):
    mkdir(environ["INDIV_PATH"])

for indiv in listdir(environ["DATASET_PATH"]):
    
    out_path = join(environ["DATASET_PATH"], indiv)
    in_path = join(environ["INDIV_PATH"], indiv)

    if not isdir(in_path):
        mkdir(in_path)

    for filename in listdir(out_path):
        
        if isResize(filename) and isImage(filename) and not isfile(join(in_path, filename)):
        
            copyfile(join(out_path, filename), join(in_path, filename))
            