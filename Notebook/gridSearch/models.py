from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3
)


class Models:

    @staticmethod
    def getList():
        return [
            "EfficientNetB0",
            "EfficientNetB1",
            "EfficientNetB2",
            "EfficientNetB3"
        ]

    @staticmethod
    def getModel(model_name):
        if model_name == "EfficientNetB0":
            return EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
        if model_name == "EfficientNetB1":
            return EfficientNetB1(
                weights='imagenet',
                include_top=False,
                input_shape=(240, 240, 3)
            )
        if model_name == "EfficientNetB2":
            return EfficientNetB2(
                weights='imagenet',
                include_top=False,
                input_shape=(206, 260, 3)
            )
        if model_name == "EfficientNetB3":
            return EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=(300, 300, 3)
            )

    @staticmethod
    def getSize(model_name):
        if model_name == "EfficientNetB0":
            return 224
        if model_name == "EfficientNetB1":
            return 240
        if model_name == "EfficientNetB2":
            return 260
        if model_name == "EfficientNetB3":
            return 300
