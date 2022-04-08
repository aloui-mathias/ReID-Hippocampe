from config import *
from os import environ, mkdir
from os.path import isdir, join, split
from shutil import copyfile, rmtree


def cleanTempFolder():
    rmtree(environ["TEMP_FOLDER"])


def copyImageToTempFolder(image_path: str) -> str:
    image_name = split(image_path)[1]
    new_name = originalname_to_customname(image_name)
    ext = new_name.split(".")[-1].lower()
    new_name = ".".join(new_name.split(".")[:-1] + [ext])
    new_path = join(environ["TEMP_FOLDER"], new_name)

    if isdir(environ["TEMP_FOLDER"]):
        cleanTempFolder()
    mkdir(environ["TEMP_FOLDER"])

    copyfile(image_path, new_path)
    return new_path


def fixYolov5():

    filepath = join(environ["ENV_PATH"],
                    "lib/site-packages/torch/nn/modules/upsampling.py")

    # Read in the file
    with open(filepath, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace(
        'recompute_scale_factor=self.recompute_scale_factor)\n',
        ('# recompute_scale_factor=self.recompute_scale_factor\n' +
         '                             )')
    )

    # Write the file out again
    with open(filepath, 'w') as file:
        file.write(filedata)


def originalname_to_customname(name: str) -> str:
    name = name.replace('©', '@')
    name = name.replace(' ', '_')
    return name


def imageToCropPath(image_path: str) -> str:
    path, image_name = split(image_path)
    return join(path, "yolo", image_name)
