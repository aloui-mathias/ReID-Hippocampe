from functions import cleanTempFolder, copyImageToTempFolder, imageToCropPath
from profil.classification import main as profil
from ReID.reid import main as reid
from yolo.detect import main as detect
from yolo.detect import parse_opt


opt = parse_opt()

if not opt.source:
    opt.source = "D:/CEFE/Dataset/BF001/BF001_2021 ©P.Louisy 200608 DSC_0716.JPG"

opt.source = copyImageToTempFolder(opt.source)

detect(opt)

crop_path = imageToCropPath(opt.source)

print(profil(crop_path))

print(reid(crop_path))

input()

cleanTempFolder()
