from config import *

import os
import sys
from get_image_size import get_image_size


def get_min_max_size():
    max_size = (0, 0)
    min_size = (sys.maxsize, sys.maxsize)
    for folder in os.listdir(DATASET_PATH):
        for filename in os.listdir(DATASET_PATH + folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                size = get_image_size(DATASET_PATH + folder + "/" + filename)
                if max(size) > max(max_size):
                    max_size = size
                if min(size) < min(min_size):
                    min_size = size
            elif not filename.lower().endswith(('.csv')):
                print(filename)
    return min_size, max_size
