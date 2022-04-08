from os import environ


environ["CUDA_VISIBLE_DEVICES"] = "-1"
environ["EMBEDDING"] = "128"
environ["ENV_PATH"] = "D:/Programmes/anaconda/envs/pipeline/Lib/site-packages"
environ["INPUT_SIZE"] = "224"
environ["PROFIL_SIZE"] = "224"
environ["PROFIL_WEIGHTS"] = "profil/weights/weights.h5"
environ["TEMP_FOLDER"] = "temp_data"
environ["WEIGHTS"] = "ReID/weights/" + environ["INPUT_SIZE"] + ".h5"
environ["YOLO_WEIGHTS"] = "yolo/weights/best.pt"
environ["YOLO_YAML"] = "yolo/hippo.yaml"

