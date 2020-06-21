from getdata import get_data, get_user_file
import os
from printimages import printImages, plot_accuracy, show_dataset_examples
import tensorflow as tf
import numpy as np
from stoptraining import StopTraining
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from get_keras_model import get_new_model, train_model, get_pre_trained_model


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

# input parameters for different workflows
SHOW_DATASET_EXAMPLE = False
USE_TEST_FOLDERS = False
TARGET_SIZE= 150
USE_PRE_TRAINED_MODEL = True
AUGMENT = False
#

if USE_TEST_FOLDERS: # for saving time during tests - give hardcoded folders
    a_dir = 'resources/cats_and_dogs_filtered/train/cats'
    b_dir = 'resources/cats_and_dogs_filtered/train/dogs'
    a_dir_validation = 'resources/cats_and_dogs_filtered/validation/cats'
    b_dir_validation = 'resources/cats_and_dogs_filtered/validation/dogs'
else:
    a_dir, b_dir, a_dir_validation, b_dir_validation = get_data()

VALIDATE = a_dir_validation is not None  # do not validate if validation set is not provided
a_label = os.path.basename(a_dir)[:-1].capitalize()  # cut last letter "s". I.e. dogs will become dog
b_label = os.path.basename(b_dir)[:-1].capitalize()

work_dir = os.path.dirname(a_dir)
print('Training dir: ', work_dir)
work_dir_validation = None
if VALIDATE:
    work_dir_validation = os.path.dirname(a_dir_validation)
print('total training {} images: {}'.format(a_label, len(os.listdir(a_dir))))
print('total training {} images: {}'.format(b_label, len(os.listdir(b_dir))))
if VALIDATE:
    print('total validation {} images: {}'.format(a_label, len(os.listdir(a_dir_validation))))
    print('total validation {} images: {}'.format(b_label, len(os.listdir(b_dir_validation))))

if SHOW_DATASET_EXAMPLE:
    show_dataset_examples([a_dir, b_dir], [a_dir_validation, b_dir_validation])

# define TensorFlow model - refactored into function to be able to use pre-trained model
if USE_PRE_TRAINED_MODEL:
    model = get_pre_trained_model(TARGET_SIZE, TARGET_SIZE)
else:
    model = get_new_model(TARGET_SIZE, TARGET_SIZE)

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

# all images will be rescaled by 1.0 / 255
# add augmentation to increase dataset size
if AUGMENT:
    train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=40,  # random rotation from 0 to 40 degrees
        width_shift_range=0.2,  # random shift width-wise from 0 to 0.2
        height_shift_range=0.2,  # random shift height-wise
        shear_range=0.2,  # shear - i.e. transform image
        zoom_range=0.2,  # random zooming
        horizontal_flip=True,
        fill_mode='nearest'  # how to fill lost pixels
    )
else:
    train_datagen = ImageDataGenerator(rescale=1 / 255.0)

validation_datagen = ImageDataGenerator(rescale=1 / 255.0)

# flow training images in batches of 20
train_generator = train_datagen.flow_from_directory(
    work_dir,  # directory with all images
    target_size=(TARGET_SIZE, TARGET_SIZE),  # all images will be resized to ...
    batch_size=32,
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


#training process refactored into function
history = train_model(model,
                      train_generator=train_generator,
                      validation_generator=validation_generator)

# printIntermediateRepresentations(show_images, model)

# plot how the accuracy evolves during the training
plot_accuracy(history, VALIDATE)

print('Recognize user images: ')


while True:
    img_paths = get_user_file(a_label, b_label)
    if not img_paths:
        break
    labels = list()
    for file in img_paths:
        # print(file)
        try:
            img = image.load_img(
                file, target_size=(TARGET_SIZE, TARGET_SIZE)
            )
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)
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