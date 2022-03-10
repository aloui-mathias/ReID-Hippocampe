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


def isResize(filename: str) -> bool:
    return getExt(withoutExt(filename)) == "resize"


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


def originalname_to_cropresizename(name: str) -> str:
    name = originalname_to_customname(name)
    name += ".crop.resize.jpg"
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
