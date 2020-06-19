from getData import getData, getUserFile
import os
from printImages import printImages, plotAccuracy, showDatasetExamples
import tensorflow as tf
import numpy as np
from numba import jit, cuda
from stopTraining import stopTraining, stopOnOptimalAccuracy, manualStop
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import concurrent.futures # fire learning in separate thread to be able to stop it manually


"""
This file executes script that trains neural network
to determine objects of two given types: type_a and type_b

Types are defined from given input directory
that contains two folders named respectively.

E.g. to work with dataset cats vs dogs the given directory
must contain two folders:
dogs - with training pictures of dogs
cats - with training pictures of cats
"""

SHOW_DATASET_EXAMPLE = False
USE_TEST_FOLDERS = True
TARGET_SIZE= 150

if USE_TEST_FOLDERS: # for saving time during tests - give hardcoded folders
    a_dir = 'resources/cats_and_dogs_filtered/train/cats'
    b_dir = 'resources/cats_and_dogs_filtered/train/dogs'
    a_dir_validation = 'resources/cats_and_dogs_filtered/validation/cats'
    b_dir_validation = 'resources/cats_and_dogs_filtered/validation/dogs'
else:
    a_dir, b_dir, a_dir_validation, b_dir_validation = getData()

VALIDATE = a_dir_validation is not None  # do not validate if validation set is not provided
a_label = os.path.basename(a_dir)[:-1].capitalize()  # cut last letter "s". I.e. dogs will become dog
b_label = os.path.basename(b_dir)[:-1].capitalize()

work_dir = os.path.dirname(a_dir)
work_dir_validation = None
if VALIDATE:
    work_dir_validation = os.path.dirname(a_dir_validation)
print('total training {} images: {}'.format(a_label, len(os.listdir(a_dir))))
print('total training {} images: {}'.format(b_label, len(os.listdir(b_dir))))
if VALIDATE:
    print('total validation {} images: {}'.format(a_label, len(os.listdir(a_dir_validation))))
    print('total validation {} images: {}'.format(b_label, len(os.listdir(b_dir_validation))))

# display dataset examples as a separate threads to avoid blocking main script with plot windows
# TKinter does not like multi-thread and plt.show(block=False) does not work
# TODO: figure out a way to show pictures non-blocking way - need to try out concurrent.futures
# if SHOW_DATASET_EXAMPLE:
#
#     start_new_thread(showDatasetExamples, (), {
#         'train_dirs': [a_dir, b_dir],
#         'validation_dirs': [a_dir_validation, b_dir_validation],
#         'number_of_images': 10
#     })

if SHOW_DATASET_EXAMPLE:
    showDatasetExamples([a_dir, b_dir], [a_dir_validation, b_dir_validation])

# define TensorFlow model
model = tf.keras.models.Sequential([
    # 1
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(TARGET_SIZE, TARGET_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # 2
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    # 3
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    # 4 and 5 are removed because size of pictures decreased
    #  tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    #  tf.keras.layers.MaxPooling2D(2, 2),
    # 5
    #  tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    #  tf.keras.layers.MaxPooling2D(2, 2),
    # flatten the image
    tf.keras.layers.Flatten(),
    # 512 connected neurons
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    # 0 - horses, 1 - humans
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

])

model.summary()
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

# all images will be rescaled by 1.0 / 255
# add augmentation to increase dataset size
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,  # random rotation from 0 to 40 degrees
    width_shift_range=0.2,  #random shift width-wise from 0 to 0.2
    height_shift_range=0.2, #random shift height-wise
    shear_range=0.2,    #shear - i.e. transform image
    zoom_range=0.2,     #random zooming
    horizontal_flip=True,
    fill_mode='nearest' #how to fill lost pixels
)
validation_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# flow training images in batches of 20
train_generator = train_datagen.flow_from_directory(
    work_dir,  # directory with all images
    target_size=(TARGET_SIZE, TARGET_SIZE),  # all images will be resized to ...
    batch_size=16,
    class_mode='binary'  # binary labels
)
validation_generator = None
if VALIDATE:
    validation_generator = validation_datagen.flow_from_directory(
        work_dir_validation,
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=16,
        class_mode='binary'
    )


# training:
def train():
    callback = stopTraining(accuracy=0.9)
    # if VALIDATE:
    #     callback = stopOnOptimalAccuracy()
    # else:
    #     callback = stopTraining(accuracy=0.8)
    return model.fit(train_generator,
                     steps_per_epoch=16,
                     epochs=1000,
                     callbacks=[callback],
                     verbose=1,
                     validation_data=validation_generator,
                     validation_steps=8
           )

def concurTrain():
    with concurrent.futures.ThreadPoolExecutor() as e:
        e.submit(manualStop, model)
        future = e.submit(train)
        return future.result()

# Try to use videocard for processing
@jit(target='cuda')
def trainWithGPU():
    print('Start training using GPU')
    return concurTrain()

try:
    history = trainWithGPU()
except:
    print('Start training using CPU')
    history = concurTrain()

# printIntermediateRepresentations(show_images, model)

# plot how the accuracy evolves during the training
plotAccuracy(history, VALIDATE)

print('Recognize user images: ')
img_paths = getUserFile(a_label, b_label)

while img_paths:
    labels = list()
    for file in img_paths:
        try:
            img = image.load_img(
                file, target_size=(TARGET_SIZE, TARGET_SIZE)
            )
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)
            # print(classes[0])
            if classes[0] > 0.5:
                # print(file + '\nis a human')
                labels.append(a_label)
            else:
                # print(file + '\nis a horse')
                labels.append(b_label)
        except:  # it is not an image - skip it
            continue
    if len(img_paths) == len(labels):
        printImages(img_paths, 'Neural network guesses', labels)
    img_paths = getUserFile(a_label, b_label)
