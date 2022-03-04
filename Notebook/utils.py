def getExt(filename: str) -> str:
    return filename.split(".")[-1]


def isCsv(filename: str) -> bool:
    return getExt(filename) == "csv"


def isCustom(filename: str) -> bool:
    return isYolo(filename) or isCrop(filename) or isResize(filename)


def isImage(filename: str) -> bool:
    return getExt(filename).lower() in ["jpg", "jpeg", "png"]


def isTxt(filename: str) -> bool:
    return getExt(filename) == "txt"


def isYolo(filename: str) -> bool:
    return getExt(withoutExt(filename)) == "yolo"


def isCrop(filename: str) -> bool:
    return getExt(withoutExt(filename)) == "crop"


def isResize(filename: str) -> bool:
    return getExt(withoutExt(filename)) == "resize"


def orginalname_to_yoloname(name: str) -> str:
    name = name.replace('©', '@')
    name = name.replace(' ', '_')
    name += ".yolo.jpg"
    return name


def withoutExt(filename: str) -> str:
    return ".".join(filename.split(".")[:-1])


def yoloname_to_originalname(name: str) -> str:
    name = name.replace('@', '©')
    name = name.replace('_', ' ')
    return withoutExt(withoutExt(name))
