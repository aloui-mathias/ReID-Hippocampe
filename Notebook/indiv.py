from config import *
from os import (
    environ,
    listdir,
    mkdir,
)
from os.path import (
    isdir,
    join
)
import random
import sys
from tqdm import tqdm
from utils import (
    copyfiles,
    isImage,
    isResize
)

SIZE = sys.argv[1]

if not isdir(environ["INDIV_PATH"]):
    mkdir(environ["INDIV_PATH"])
if not isdir(join(environ["INDIV_PATH"], f"{SIZE}")):
    mkdir(join(environ["INDIV_PATH"], f"{SIZE}"))
    mkdir(join(environ["INDIV_PATH"], f"{SIZE}", "train"))
    mkdir(join(environ["INDIV_PATH"], f"{SIZE}", "test"))
else:
    print(f"indiv for {SIZE} already exits")
    exit()

for indiv in tqdm(listdir(environ["DATASET_PATH"])):
    
    out_path = join(environ["DATASET_PATH"], indiv)
    
    files = []

    for filename in listdir(out_path):
        
        if isResize(filename, SIZE) and isImage(filename):
            files.append(filename)
            
    if len(files) < 4:
        in_path = join(environ["INDIV_PATH"], f"{SIZE}", "test", indiv)
        mkdir(in_path)
        copyfiles(files, out_path, in_path)
    else:
        in_train = join(environ["INDIV_PATH"], f"{SIZE}", "train", indiv)
        mkdir(in_train)
        in_test = join(environ["INDIV_PATH"], f"{SIZE}", "test", indiv)
        mkdir(in_test)
        random.shuffle(files)
        split_id = (int)(len(files)*0.9)
        copyfiles(files[:split_id], out_path, in_train)
        copyfiles(files[split_id:], out_path, in_test)
    
    
    
            