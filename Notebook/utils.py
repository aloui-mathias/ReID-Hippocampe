from os.path import join
from shutil import copyfile
from typing import List, Optional


def copyfiles(files: List[str], out_path: str, in_path: str) -> None:
    for filename in files:
        copyfile(join(out_path, filename), join(in_path, filename))


def getExt(filename: str) -> str:
    return filename.split(".")[-1]


def isCrop(filename: str) -> bool:
    return getExt(withoutExt(filename)) == "crop"


def isCsv(filename: str) -> bool:
    return getExt(filename) == "csv"


def isCustom(filename: str) -> bool:
    return isCrop(filename) or isResize(filename) or isYoloImage(filename)


def isImage(filename: str) -> bool:
    return getExt(filename).lower() in ["jpg", "jpeg", "png"]


def isResize(filename: str, size: Optional[int] = None) -> bool:
    if size:
        return getExt(withoutExt(filename)) == f"resize{size}"
    return "resize" in getExt(withoutExt(filename))


def isTxt(filename: str) -> bool:
    return getExt(filename) == "txt"


def isYolo(filename: str) -> bool:
    return getExt(filename) == "yolo"


def isYoloImage(filename: str) -> bool:
    return isImage(filename) and isYolo(withoutExt(filename))


def originalname_to_cropname(name: str) -> str:
    name = originalname_to_customname(name)
    name += ".crop.jpg"
    return name


def originalname_to_cropresizename(name: str, size: int) -> str:
    name = originalname_to_customname(name)
    name += ".crop.resize"
    if size:
        name += f"{size}"
    name += ".jpg"
    return name


def originalname_to_customname(name: str) -> str:
    name = name.replace('©', '@')
    name = name.replace(' ', '_')
    return name


def originalname_to_yoloname(name: str) -> str:
    name = originalname_to_customname(name)
    name += ".yolo.jpg"
    return name


def withoutExt(filename: str) -> str:
    return ".".join(filename.split(".")[:-1])
