from os import environ


# A garder pour préciser pas de GPU compatible
environ["CUDA_VISIBLE_DEVICES"] = "-1"
environ["EMBEDDING"] = "128"
environ["ENV_PATH"] = "/home/data/.conda/envs/pipeline/lib/python3.8/site-packages"
environ["INPUT_SIZE"] = "224"
environ["PROFIL_SIZE"] = "224"
environ["PROFIL_WEIGHTS"] = "profil/weights/weights.h5"
environ["TEMP_FOLDER"] = "temp_data"
environ["WEIGHTS"] = "ReID/weights/" + environ["INPUT_SIZE"] + ".h5"
environ["WEIGHTS_PATH"] = "ReID/weights/"
environ["YOLO_WEIGHTS"] = "yolo/weights/best.pt"
environ["YOLO_YAML"] = "yolo/hippo.yaml"
environ["DB_PATH"] = "D:\\CEFE\DB"
environ["CROP_SIZES"] = "224 240 260 300 380 456 528"
environ["DATASET_PATH"] = "D:\\CEFE\\Dataset"
environ["INDIV_PATH"] = "D:\\CEFE\\indiv"
