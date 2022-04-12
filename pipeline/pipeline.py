from functions import cleanTempFolder, copyImageToTempFolder, imageToCropPath
from numpy import log10
from profil.classification import main as profil
from ReID.reid import main as reid
from yolo.detect import main as detect
from yolo.detect import parse_opt


def main(img_path: str = "D:/CEFE/Dataset/BF001/" +
         "BF001_2021 ©P.Louisy 200608 DSC_0716.JPG",
         stop: bool = False):

    opt = parse_opt()

    if not opt.source:
        opt.source = img_path

    opt.source = copyImageToTempFolder(opt.source)

    detect(opt)

    crop_path = imageToCropPath(opt.source)

    profil_score = profil([crop_path])[0]

    reid_score = reid([crop_path])[0]

    with open("pipeline.result.txt", "w") as file:

        file.write("Profil :\n")
        file.write(f"score = {profil_score}\n")
        file.write("prediction = ")
        if profil_score < 0.5:
            proba = log10(2-2*profil_score)*100/log10(2)
            file.write("Gauche\n")
        elif profil_score > 0.5:
            proba = log10(2*profil_score)*100/log10(2)
            file.write("Droit\n")
        else:
            file.write("Indetermine\n")

        if profil_score != 0.5:
            file.write(f"probabilite = {proba}%\n")

        file.write("ReID :\n")
        file.write(f"{reid_score}\n")

    if stop:
        input()

    cleanTempFolder()

    return [reid_score, profil_score]


if __name__ == "__main__":
    main(stop=True)
