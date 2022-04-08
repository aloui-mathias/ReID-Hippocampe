import imgaug as ia
from imgaug import augmenters as iaa 
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sys
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

#Fix imgaug color change
filepath = "/home/data/.conda/envs/reid/lib/python3.8/site-packages/imgaug/augmenters/color.py"

# Read in the file
with open(filepath, 'r') as file :
  filedata = file.read()

# Replace the target string
filedata = filedata.replace('+ interpolation_factors\n', '+ interpolation_factors[:,None]\n')

# Write the file out again
with open(filepath, 'w') as file:
  file.write(filedata)

SIZE = (int)(sys.argv[1])
INDIV_PATH = "/home/data/indiv/" + str(SIZE) + "/"

test_dataset = []

# Iterate over the folders (individuals)
for indiv in os.listdir(os.path.join(INDIV_PATH, "test")):
  for picturename in os.listdir(os.path.join(INDIV_PATH, "test", indiv)):
    test_dataset.append({'individual':indiv, 'path': os.path.join(INDIV_PATH, "test", indiv, picturename)})

# Prepare a dataframe and preview it
test_dataset = pd.DataFrame(test_dataset)

train_dataset = []

# Iterate over the folders (individuals)
for indiv in os.listdir(os.path.join(INDIV_PATH, "train")):
  for picturename in os.listdir(os.path.join(INDIV_PATH, "train", indiv)):
    train_dataset.append({'individual':indiv, 'path': os.path.join(INDIV_PATH, "train", indiv, picturename)})

# Prepare a dataframe and preview it
train_dataset = pd.DataFrame(train_dataset)

min_images_par_class = np.min(train_dataset["individual"].value_counts())

min_images_par_class = 2

from kerasgen.balanced_image_dataset import balanced_image_dataset_from_directory

batch_size = 20

train_ds = balanced_image_dataset_from_directory(
    os.path.join(INDIV_PATH, "train"),
    num_classes_per_batch=batch_size, 
    num_images_per_class=min_images_par_class,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    image_size=(SIZE, SIZE),
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


val_ds = balanced_image_dataset_from_directory(
    os.path.join(INDIV_PATH, "train"),
    num_classes_per_batch=batch_size, 
    num_images_per_class=min_images_par_class,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    image_size=(SIZE, SIZE),
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

train_ds = train_ds.map(augment)

images, labels = [], []

# Iterate over the rows of the newly created dataframe
for (_, row) in test_dataset.iterrows():
    # Read the image from the disk, resize it, and scale the pixel
    # values to [0,1] range
    image = plt.imread(row['path'])
    images.append(image)

    # Parse the name of the person
    labels.append(row['individual'])

images = np.array(images)
labels = np.array(labels)

def embedding_model(embedding_dim, pretrained_model):
    # Create the placeholder for the input image and run it through our pre-trained
    # VGG16 model. Note that we will be fine-tuning this feature extractor but it will still
    # run in *inference model* making sure the BatchNorm layers are not training.
    inputs = keras.layers.Input(shape=pretrained_model.input_shape[1:])
    features = pretrained_model(inputs, training=False)
    
    # Create the upper half of our model where we pool out the extracted features,
    # pass it through a mini fully-connected network. We also force the embeddings 
    # to remain on a unit hypersphere space.
    x = keras.layers.GlobalAveragePooling2D()(features)
    x = keras.layers.Dense(2048, activation='relu')(x)
    x = keras.layers.Dense(embedding_dim)(x)
    outputs = keras.layers.Lambda(lambda a: tf.math.l2_normalize(a, axis=1))(x)
    
    # Create the final model
    model = keras.models.Model(inputs, outputs)

    return model

# Define the LR schedule constants
start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005
rampup_epochs = 5
sustain_epochs = 0
exp_decay = .8

# Define the LR schedule as a callback
def lrfn(epoch):
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        if epoch < rampup_epochs:
            lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        else:
            lr = (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
        return lr
    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

# Define a `EarlyStopping` callback so that our model does overfit 
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, verbose=2, mode='auto',
    restore_best_weights=True
)

if SIZE == 224:
    premodel = tf.keras.applications.EfficientNetB0(
        weights ='imagenet',
        include_top = False,
        input_shape = (SIZE, SIZE, 3)
    )
elif SIZE == 240:
    premodel = tf.keras.applications.EfficientNetB1(
        weights ='imagenet',
        include_top = False,
        input_shape = (SIZE, SIZE, 3)
    )
elif SIZE == 260:
    premodel = tf.keras.applications.EfficientNetB2(
        weights ='imagenet',
        include_top = False,
        input_shape = (SIZE, SIZE, 3)
    )
elif SIZE == 300:
    premodel = tf.keras.applications.EfficientNetB3(
        weights ='imagenet',
        include_top = False,
        input_shape = (SIZE, SIZE, 3)
    )
elif SIZE == 380:
    premodel = tf.keras.applications.EfficientNetB4(
        weights ='imagenet',
        include_top = False,
        input_shape = (SIZE, SIZE, 3)
    )
elif SIZE == 456:
    premodel = tf.keras.applications.EfficientNetB5(
        weights ='imagenet',
        include_top = False,
        input_shape = (SIZE, SIZE, 3)
    )
elif SIZE == 528:
    premodel = tf.keras.applications.EfficientNetB6(
        weights ='imagenet',
        include_top = False,
        input_shape = (SIZE, SIZE, 3)
    )
elif SIZE == 600:
    premodel = tf.keras.applications.EfficientNetB7(
        weights ='imagenet',
        include_top = False,
        input_shape = (SIZE, SIZE, 3)
    )
else:
    print("Wrong SIZE")
    exit()


premodel.trainable = True # We will be fine-tuning feature extractor network

# Initialize the model. Here we are extracting 128-dimensional embedding vectors. This is
# a hyperparameter you might want to experiment with. 
model = embedding_model(128, premodel)

# Compile the model with TripletLoss
model.compile(optimizer=tf.keras.optimizers.Adam(), 
    # According to the FaceNet paper, Semi-hard Triplet Mining yielded good results. 
    loss=tfa.losses.TripletSemiHardLoss() 
)

# Train the model and visualize the training progress
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=50,
                    callbacks=[lr_callback, es])

model.save_weights("/home/data/save_models/" + str(SIZE) + ".h5")

def evalutate(model, images, train_labels):

    test_features = model.predict(images)

    neighbors = NearestNeighbors(n_neighbors=11,
        algorithm='brute',
        metric='euclidean').fit(test_features)

    reid = []
    newid = []

    for id, feature in enumerate(test_features):

        distances, indices = neighbors.kneighbors([feature])

        similar_labels = [labels[indices[0][i]] for i in range(1, 11)]

        if labels[id] in train_labels:
            result = (id, None, None)
            for id2, similar_label in enumerate(similar_labels):
                if similar_label == labels[id]:
                    result = (id, distances[0][id2+1], id2+1)
                    break
            reid.append(result)
        else:
            newid.append((id, distances[0][1]))

    reid_df = pd.DataFrame(reid, columns=("id", "distance", "top"))
    newid_df = pd.DataFrame(newid, columns=("id", "distance"))

    with open(f"result{SIZE}aug.txt", "w") as f:
        f.write("New ID Distances :\n")
        distances = newid_df["distance"]
        f.write("min = " + str(min(distances)) + "\n")
        f.write("max = " + str(max(distances)) + "\n")
        f.write("mean = " + str(np.mean(distances)) + "\n")
        for q in range(1, 10):
            f.write(f"{q/10} quantile = " + str(np.quantile(distances, q/10)) + "\n")

        f.write("-----------------------------------------\n")

        f.write("ReID Distances :\n")
        distances = reid_df["distance"].dropna()
        f.write("min = " + str(min(distances)) + "\n")
        f.write("max = " + str(max(distances)) + "\n")
        f.write("mean = " + str(np.mean(distances)) + "\n")
        for q in range(1, 10):
            f.write(f"{q/10} quantile = " + str(np.quantile(distances, q/10)) + "\n")

        f.write("-----------------------------------------\n")

        f.write("ReID Tops :\n")
        tops = reid_df["top"].dropna()
        f.write("min = " + str(min(tops)) + "\n")
        f.write("max = " + str(max(tops)) + "\n")
        f.write("mean = " + str(np.mean(tops)) + "\n")
        for q in range(1, 10):
            f.write(f"{q/10} quantile = " + str(np.quantile(tops, q/10)) + "\n")

        f.write("-----------------------------------------\n")

        f.write("ReID Fails (not in top-10) = " + str(len(reid_df[reid_df["top"].isna()])/len(reid_df["top"])) + "\n")

evalutate(model, images, list(train_dataset["individual"]))