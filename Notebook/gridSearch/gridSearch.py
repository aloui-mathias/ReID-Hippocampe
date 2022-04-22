import imgaug as ia
from imgaug import augmenters as iaa 
from kerasgen.balanced_image_dataset import balanced_image_dataset_from_directory
from matplotlib import pyplot as plt
from models import Models
import numpy as np
from os import (
    listdir,
    mkdir
)
from os.path import (
    isfile,
    join
)
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tqdm import tqdm


# Fix imgaug color change


filepath = "/home/data/.conda/envs/pipeline/lib/python3.8/site-packages/imgaug/augmenters/color.py"

with open(filepath, 'r') as file:
    filedata = file.read()

filedata = filedata.replace(
    '+ interpolation_factors\n', '+ interpolation_factors[:,None]\n')

with open(filepath, 'w') as file:
    file.write(filedata)


# Set the fix variables between models

EMBEDDING_SIZE = 128
NUM_CLASSES_PER_BATCH = 3
NUM_IMAGES_PER_CLASSE = 10
DATASET_PATH = "/home/data/indiv"


# Function to train and evaluate a model


def train(premodel, dropout, triplet_distance, train_ds, val_ds, path):

    # Get input shape
    input_size = premodel.input_shape[1:]

    premodel.trainable = True

    inputs = keras.layers.Input(shape=input_size)
    features = premodel(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(features)
    x = keras.layers.Dense(2048, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(EMBEDDING_SIZE)(x)
    # Checked on github, even cosine need l2_normalization
    outputs = keras.layers.Lambda(
        lambda a: tf.math.l2_normalize(a, axis=1))(x)
    model = keras.models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tfa.losses.TripletSemiHardLoss(distance_metric=triplet_distance)
    )

    es = keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=10)

    mc = keras.callbacks.ModelCheckpoint(
        join(path, 'best_model.h5'),
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=100,
                        callbacks=[es, mc])

    return model, history


def plot(history, path):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(join(path, "plot.png"))


def eval(model, eval_disance, test_path):
    x = []
    y = []
    for indiv in listdir(test_path):
        for pic_name in listdir(join(test_path, indiv)):
            img = Image.open(join(test_path, indiv, pic_name))
            img = np.array(img)
            x.append(model.predict(img))
            y.append(indiv)
    neighbors = NearestNeighbors(n_neighbors=5,
                                 metric=eval_disance)
    neighbors.fit(x, y)
    sum_cmcks = np.zeros(shape=(5))
    sum_APks = np.zeros(shape=(5))
    for label in tqdm(np.unique(y)):
        indices = np.where(y == label)[0]
        x_label = x[indices]
        neighbors = np.array(
            neighbors.kneighbors(x_label))
        neighbors_labels = y[
            neighbors[1, :, :].astype(int)]
        cmcks = []
        APks = []
        for labels in neighbors_labels:
            cmck = np.zeros(shape=(5))
            APk = np.zeros(shape=(5))
            for i in range(5):
                if label == labels[i]:
                    for j in range(i, 5):
                        APk[j] = 1/(i+1)
                        cmck[j] = 1
                    break
            cmcks.append(cmck)
            APks.append(APk)
        sum_cmck = np.sum(cmcks, axis=0)
        sum_APk = np.sum(APks, axis=0)
        sum_cmcks += sum_cmck
        sum_APks += sum_APk
    mean_cmcks = sum_cmcks / len(x)
    mAPs = sum_APks / len(x)
    return mean_cmcks, mAPs


def evals(model, test_path):
    CMCks = []
    mAPks = []
    for eval_distance in ["l2", "euclidean", "cosine"]:
        CMCk, mAPk = eval(model, eval_distance, test_path)
        CMCks.append(CMCk)
        mAPks.append(mAPk)
    return ["l2", "euclidean", "cosine"], CMCks, mAPks


def search(
        model_name, dropout, triplet_distance,
        train_ds, val_ds, test_path):

    premodel = Models.getModel(model_name)

    path = ".".join([model_name, str(dropout), triplet_distance])

    mkdir(path)

    model, history = train(
        premodel, dropout, triplet_distance, train_ds, val_ds, path)

    plot(history, path)

    eval_distances, CMCks, mAPks = evals(model, test_path)

    return pd.DataFrame({
        "model_name": [model_name]*3,
        "dropout": [dropout] * 3,
        "triplet_distance": [triplet_distance] * 3,
        "eval_distance": eval_distances,
        "CMC@k": CMCks,
        "mAP@k": mAPks
    })


# Create Dataframe to store results to csv

if not isfile("results.csv"):
    results = pd.DataFrame(
        columns=[
            "model_name",
            "dropout",
            "triplet_distance",
            "eval_distance"
        ]
    )
else:
    results = pd.read_csv("results.csv")


# Augmentations


sometimes = lambda aug: iaa.Sometimes(0.5, aug)

class MyParameter(ia.parameters.StochasticParameter):
    
    def __init__(self, lb, ub, mid):
        self.lb = lb
        self.ub = ub
        self.mid = mid

    def _draw_samples(self, size, random_state):
        samples = []
        for i in range(size[0]):
            if np.random.random() < 0.5:
                samples.append(np.random.uniform(self.lb, self.mid))
            else:
                samples.append(np.random.uniform(self.mid, self.ub))
        return np.array(samples).reshape(size)

seq = iaa.Sequential(
    [
        sometimes(iaa.Salt((0.001, 0.05))),
        iaa.Rotate((-180,180)),
        sometimes(iaa.AverageBlur(k=(2,3))),
        sometimes(iaa.ChangeColorTemperature(MyParameter(4000, 20000, 6600))),
        sometimes(iaa.WithBrightnessChannels(iaa.Add((-30, 30))))
    ]
)

def augment(images, labels):
    img_dtype = images.dtype
    img_shape = tf.shape(images)
    images = tf.numpy_function(seq.augment_images,
                                [tf.cast(images, np.uint8)],
                                np.uint8)
    images = tf.cast(images, img_dtype)
    images = tf.reshape(images, shape = img_shape)
    return images, labels


# Iterate over all parameters

for model_name in [Models.getList()[3]]:

    # Load Dataset

    size = Models.getSize(model_name)

    path = join(DATASET_PATH, str(size))

    train_path = join(path, "train")
    test_path = join(path, "test")

    train_ds = balanced_image_dataset_from_directory(
        train_path,
        num_classes_per_batch=NUM_CLASSES_PER_BATCH,
        num_images_per_class=NUM_IMAGES_PER_CLASSE,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        image_size=(size, size),
        shuffle=True,
        seed=555,
        validation_split=0.1,
        subset='training',
        safe_triplet=True,
        samples_per_epoch=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    train_ds = train_ds.map(augment)

    val_ds = balanced_image_dataset_from_directory(
        train_path,
        num_classes_per_batch=NUM_CLASSES_PER_BATCH,
        num_images_per_class=NUM_IMAGES_PER_CLASSE,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        image_size=(size, size),
        shuffle=True,
        seed=555,
        validation_split=0.1,
        subset='validation',
        safe_triplet=True,
        samples_per_epoch=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    for dropout in [0.1, 0.3, 0.5, 0.7]:
        for triplet_distance in ["L2", "squared-L2", "angular"]:
            if (
                    (results["model_name"] == model_name) &
                    (results["dropout"] == dropout) &
                (results["triplet_distance"] == triplet_distance)
            ).any():
                print(" - ".join([model_name, str(dropout),
                      triplet_distance]), " : Already done")
            else:
                result = search(
                    model_name,
                    dropout,
                    triplet_distance,
                    train_ds,
                    val_ds,
                    test_path
                )
                results = pd.concat([results, result])
                results.to_csv("results.csv", index=False)
            break
        break
    break
