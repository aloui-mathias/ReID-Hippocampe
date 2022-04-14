from config import *
from imgaug import augmenters as iaa 
import numpy as np
from os import environ
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7
)


class model:
    
    def __init__(self, size: str = None, name: str = None) -> None:
        if not size:
            self.SIZE = (int)(environ["INPUT_SIZE"])
            WEIGHTS = environ["WEIGHTS"]
        elif not name:
            self.SIZE = (int)(size)
            WEIGHTS = environ["WEIGHTS_PATH"] + str(self.SIZE) + ".h5"
        else :
            self.SIZE = (int)(size)
            WEIGHTS = environ["WEIGHTS_PATH"] + str(self.SIZE) + name + ".h5"
        DIM = (int)(environ["EMBEDDING"])
        
        if self.SIZE == 224:
            premodel = EfficientNetB0(
                weights ='imagenet',
                include_top = False,
                input_shape = (self.SIZE, self.SIZE, 3)
            )
        elif self.SIZE == 240:
            premodel = EfficientNetB1(
                weights ='imagenet',
                include_top = False,
                input_shape = (self.SIZE, self.SIZE, 3)
            )
        elif self.SIZE == 260:
            premodel = EfficientNetB2(
                weights ='imagenet',
                include_top = False,
                input_shape = (self.SIZE, self.SIZE, 3)
            )
        elif self.SIZE == 300:
            premodel = EfficientNetB3(
                weights ='imagenet',
                include_top = False,
                input_shape = (self.SIZE, self.SIZE, 3)
            )
        elif self.SIZE == 380:
            premodel = EfficientNetB4(
                weights ='imagenet',
                include_top = False,
                input_shape = (self.SIZE, self.SIZE, 3)
            )
        elif self.SIZE == 456:
            premodel = EfficientNetB5(
                weights ='imagenet',
                include_top = False,
                input_shape = (self.SIZE, self.SIZE, 3)
            )
        elif self.SIZE == 528:
            premodel = EfficientNetB6(
                weights ='imagenet',
                include_top = False,
                input_shape = (self.SIZE, self.SIZE, 3)
            )
        elif self.SIZE == 600:
            premodel = EfficientNetB7(
                weights ='imagenet',
                include_top = False,
                input_shape = (self.SIZE, self.SIZE, 3)
            )
        else:
            print("Wrong SIZE")
            exit()

        def embedding_model(pretrained_model):
            # Create the placeholder for the input image and run it through our pre-trained
            # VGG16 model. Note that we will be fine-tuning this feature extractor but it will still
            # run in *inference model* making sure the BatchNorm layers are not training.
            inputs = layers.Input(shape=pretrained_model.input_shape[1:])
            features = pretrained_model(inputs, training=False)
            
            # Create the upper half of our model where we pool out the extracted features,
            # pass it through a mini fully-connected network. We also force the embeddings 
            # to remain on a unit hypersphere space.
            x = layers.GlobalAveragePooling2D()(features)
            x = layers.Dense(2048, activation='relu')(x)
            x = layers.Dense(DIM)(x)
            outputs = layers.Lambda(lambda a: tf.math.l2_normalize(a, axis=1))(x)
            
            # Create the final model
            model = models.Model(inputs, outputs)

            return model

        # Initialize the model. Here we are extracting 128-dimensional embedding vectors. This is
        # a hyperparameter you might want to experiment with. 
        self.model = embedding_model(premodel)
        
        self.model.load_weights(WEIGHTS)

    def predict(self, image_paths: str):

        imgs = []
        
        resize = iaa.Resize({"longer-side": self.SIZE, "shorter-side": "keep-aspect-ratio"})
        padding = iaa.PadToSquare(pad_mode="constant", pad_cval=0, position="center")
        
        for image_path in image_paths:
            img = Image.open(image_path)
            img = np.array(img)
            img = padding(image=resize(image=img)).reshape((1,self.SIZE,self.SIZE,3))
            imgs.append(img)
        
        preds = []
        
        for img in imgs:
            pred = self.model.predict(img)
            preds.append(pred[0])

        return preds
        
        