from getData import getData, getUserFile
import os
from printImages import printImages, plotAccuracy, showDatasetExamples
import tensorflow as tf
import numpy as np
from numba import jit, cuda
from stopTraining import stopTraining, stopOnOptimalAccuracy
from keras_preprocessing import image
from _thread import start_new_thread

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
# TODO: figure out a way to show pictures non-blocking way
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
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # 2
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    # 3
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    # 4 and 5 are tremoved because size of pictures decreased
    # tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # 5
    # tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # flatten the image
    tf.keras.layers.Flatten(),
    # 512 connected neurons
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    # 0 - horses, 1 - humans
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

])

model.summary()
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

# all images will be rescaled by 1.0 / 255
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

# flow training images in batches of 20
train_generator = train_datagen.flow_from_directory(
    work_dir,  # directory with all images
    target_size=(150, 150),  # all images will be resized to ...
    batch_size=20,
    class_mode='binary'  # binary labels
)
validation_generator = None
if VALIDATE:
    validation_generator = validation_datagen.flow_from_directory(
        work_dir_validation,
        target_size=(150, 150),
        batch_size=20,
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
                     #steps_per_epoch=100,
                     epochs=15,
                     callbacks=[callback],
                     verbose=1,
                     validation_data=validation_generator,
                     validation_steps=50)


# Try to use videocard for processing
@jit(target='cuda')
def trainWithGPU():
    print('Start training using GPU')
    return train()


try:
    history = trainWithGPU()
except:
    print('Start training using CPU')
    history = train()

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
                file, target_size=(150, 150)
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
