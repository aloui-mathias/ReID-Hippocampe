def orginalname_to_cropname(name: str) -> str:
    name = name.replace('©', '@')
    name = name.replace(' ', '_')
    return name


def cropname_to_originalname(name: str) -> str:
    name = name.replace('@', '©')
    name = name.replace('_', ' ')
    return name
