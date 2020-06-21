import tensorflow as tf
import concurrent.futures  # fire learning in separate thread to be able to stop it manually
from stopTraining import manualStop
from numba import jit, cuda
import requests
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3


def getPreTrainedModel(image_width, image_height):
    local_weights_file = 'resources/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    if not os.path.exists(local_weights_file):
        print('Downloading file from URL')
        url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        try:
            r = requests.get(url, allow_redirects=True)
            open(local_weights_file, 'wb').write(r.content)
        except Exception as e:
            print('Unable to get model')
            print(e)
            exit(1)
        if os.path.exists(local_weights_file) and os.path.getsize(local_weights_file) > 0:
            print('Model downloaded successfully')
    pre_trained_model = InceptionV3(
        input_shape=(image_width, image_height, 3),
        include_top=False,
        weights=None
    )

    pre_trained_model.load_weights(local_weights_file)
    for layer in pre_trained_model.layers:
        # print(layer.name, layer.output_shape)
        layer.trainable = True
    last_layer = pre_trained_model.get_layer('mixed6')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    # flatten the ouput layer to 1 dimension
    x = tf.keras.layers.Flatten()(last_output)

    # add a fully connected layer with 1024 hidden units
    x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)

    # add a final sigmoid layer for classification
    x = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(pre_trained_model.input, x)

    return model

def getNewModel(image_width, image_height):
    model = tf.keras.models.Sequential([
        # 1
        tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(image_width, image_height, 3)),
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
    return model

# training:
def train(model, train_generator, validation_generator):
    return model.fit(train_generator,
                     steps_per_epoch=8,
                     epochs=40,
                     callbacks=None,
                     verbose=1,
                     validation_data=validation_generator
                     #validation_steps=8
                     )


def concurTrain(model, train_generator, validation_generator):
    with concurrent.futures.ThreadPoolExecutor() as e:
        cancel = e.submit(manualStop, model)
        future = e.submit(train, model, train_generator, validation_generator)
        cancel.cancel()
        return future.result()


# Try to use videocard for processing
@jit(target='cuda')
def trainWithGPU(model, train_generator, validation_generator):
    print('Start training using GPU')
    return concurTrain(model, train_generator, validation_generator)


def trainModel(model, train_generator, validation_generator):
    try:
        return trainWithGPU(model, train_generator, validation_generator)
    except Exception as e:
        print(e)
        print('Start training using CPU')
        return concurTrain(model, train_generator, validation_generator)
