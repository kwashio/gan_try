# coding: utf-8
"""
dcgan + feature matchingの実装、CIFAR10で実験
"""

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout
import math
import numpy as np
import os
from keras.datasets import cifar10
from keras.optimizers import Adam
from PIL import Image
from keras import backend as K
from keras.losses import binary_crossentropy, mean_squared_error

BATCH_SIZE = 256
NUM_EPOCH = 20
GENERATED_IMAGE_PATH = 'generated_images/'  # 生成画像の保存先


def generator_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128 * 8 * 8))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((8, 8, 128)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, 5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, 5, padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5),
                     strides=(2, 2),
                     padding='same',
                     input_shape=(32, 32, 3)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (5, 5), strides=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256, name='target_hidden_layer'))  # ここでfeature matching
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def combine_images(generated_images):
    total = generated_images.shape[0]
    print(generated_images.shape)
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total) / cols)
    width, height, channel = generated_images.shape[1:]
    print(width, height, channel)
    combined_image = np.zeros((height * rows, width * cols, channel),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        combined_image[width * i:width * (i + 1), height * j:height * (j + 1), 0:3] = image[:, :, 0:3]
    return combined_image


def feature_matching_loss(y_true, y_pred):
    target_feature = K.mean(y_true, axis=0)
    predicted_feature = K.mean(y_pred, axis=0)
    l2 = K.sum((target_feature - predicted_feature) ** 2)
    return l2


def train():
    (X_train, y_train), (_, _) = cifar10.load_data()
    X_train = (X_train.astype(np.float32) / 127.5) - 1

    discriminator = discriminator_model()
    d_opt = Adam(lr=1e-5, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    discriminator.trainable = False
    generator = generator_model()

    # DCGAN with feature matching
    discriminator_hidden_model = Model(inputs=discriminator.input,
                                       outputs=discriminator.get_layer('target_hidden_layer').output)
    discriminator_hidden_model.trainable = False
    noise_inputs = Input(shape=(100,))
    generated = generator(noise_inputs)
    main_output = discriminator(generated)
    aux_output = discriminator_hidden_model(generated)
    dcgan = Model(inputs=noise_inputs, outputs=[main_output, aux_output])
    g_opt = Adam(lr=2e-4, beta_1=0.5)

    dcgan.compile(optimizer=g_opt,
                  loss=['binary_crossentropy', feature_matching_loss],
                  loss_weights=[1.0, 1.0])

    num_batches = int(X_train.shape[0] / BATCH_SIZE)

    print('Number of batches:', num_batches)
    for epoch in range(num_batches):
        for index in range(num_batches):
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)

            if index % 500 == 0:
                image = combine_images(generated_images)
                image = (image + 1) * 127.5
                if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.mkdir(GENERATED_IMAGE_PATH)
                Image.fromarray(image.astype(np.uint8)) \
                    .save(GENERATED_IMAGE_PATH + "{0}_{1}.png".format(epoch, index))

            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            expected_feature = discriminator_hidden_model.predict(image_batch)
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])

            labels = [1] * BATCH_SIZE
            g_loss = dcgan.train_on_batch(noise, [np.array(labels), expected_feature])
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss[0], d_loss))
        generator.save_weights('generator.h5')
        discriminator.save_weights('discriminator.h5')


if __name__ == '__main__':
    train()
