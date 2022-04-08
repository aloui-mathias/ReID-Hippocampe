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

profil_score = profil(crop_path)

reid_score = reid(crop_path)


with open("result.txt", "w") as file:

    if profil_score < 0.45:
        file.write(f"Profil gauche : {profil_score[0][0]}\n")
    elif profil_score > 0.55:
        file.write(f"Profil droit : {profil_score[0][0]}\n")
    else:
        file.write(f"Profil indéterminé : {profil_score[0][0]}\n")

    file.write(f"Encodage de l'image : {reid_score[0]}\n")

input()

cleanTempFolder()
