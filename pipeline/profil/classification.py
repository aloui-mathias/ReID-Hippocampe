from config import *
from imgaug import augmenters as iaa
import numpy as np
from os import environ
from PIL import Image
from tensorflow.keras import layers, models


def profilModel():
    return models.Sequential([
        # Note the input shape is the desired size of the image 200x200 with 3 bytes color
        # This is the first convolution
        layers.Conv2D(16, (3, 3), activation='relu',
                      input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        # The second convolution
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        # The third convolution
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        # The fourth convolution
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        # # The fifth convolution
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        layers.Flatten(),
        # 512 neuron hidden layer
        layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('left') and 1 for the other ('right')
        layers.Dense(1, activation='sigmoid')
    ])


def main(image_paths: str):

    SIZE = (int)(environ["PROFIL_SIZE"])

    model = profilModel()
    model.load_weights(environ["PROFIL_WEIGHTS"])

    resize = iaa.Resize(
        {"longer-side": SIZE, "shorter-side": "keep-aspect-ratio"})
    padding = iaa.PadToSquare(
        pad_mode="constant", pad_cval=0, position="center")

    imgs = []
    for image_path in image_paths:
        img = Image.open(image_path)
        img = np.array(img)
        img = padding(image=resize(image=img)).reshape((1, SIZE, SIZE, 3))
        imgs.append(img)

    preds = []
    for img in imgs:
        preds.append(model.predict(img)[0][0])

    return preds
