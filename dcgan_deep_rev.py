# coding: utf-8
"""
reverserの実装、CIFAR10で実験
モデル構成をNVIDIAのブログのものに
https://devblogs.nvidia.com/parallelforall/photo-editing-generative-adversarial-networks-2/
dcgan_deep_with_fm.pyでgeneratorとdiscriminatorを学習した後にうごかす。
"""

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten
import math
import numpy as np
import os
from keras.datasets import cifar10
from keras.optimizers import Adam
from PIL import Image
from keras import backend as K

BATCH_SIZE = 64
NUM_EPOCH = 60
GENERATED_IMAGE_PATH = 'encoded_images/'  # 生成画像の保存先


def generator_model():
    # input
    # (N, 100)
    model = Sequential()
    model.add(Dense(8192, input_dim=100))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(Activation('relu'))

    # (N, 4, 4, 512)
    model.add(Reshape((4, 4, 512)))

    # (N, 8, 8, 256)
    model.add(Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(Activation('relu'))

    # (N, 16, 16, 128)
    model.add(Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(Activation('relu'))

    # (N, 32, 32, 64)
    model.add(Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(Activation('relu'))

    # output
    # (N, 32, 32, 3)
    model.add(Conv2D(filters=3, kernel_size=5, padding='same'))
    model.add(Activation('tanh'))

    return model


def discriminator_model():
    # (N, 32, 32, 3) input
    model = Sequential()

    # (N, 32, 32, 64)
    model.add(Conv2D(filters=64, kernel_size=5, padding='same', input_shape=(32, 32, 3)))
    model.add(LeakyReLU(0.2))

    # (N, 16, 16, 128)
    model.add(Conv2D(filters=128, kernel_size=5, padding='same', strides=2))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(LeakyReLU(0.2))

    # (N, 8, 8, 256)
    model.add(Conv2D(filters=256, kernel_size=5, padding='same', strides=2))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(LeakyReLU(0.2))

    # (N, 4, 4, 512)
    model.add(Conv2D(filters=512, kernel_size=5, padding='same', strides=2))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(LeakyReLU(0.2))

    # (N, 8192)
    model.add(Flatten(name='last_hidden_layer'))

    # (N, 1) output
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def encoder_model(discriminator_hidden_model):
    image_input = Input(shape=(32, 32, 3))
    feature = discriminator_hidden_model(image_input)
    predicted_z = Dense(100)(feature)
    model = Model(inputs=image_input, outputs=predicted_z)
    return model


def reverser_model(encoder, generator):
    image_inputs = Input(shape=(32, 32, 3))
    encoded_z = encoder(image_inputs)
    generated = generator(encoded_z)
    model = Model(inputs=image_inputs, outputs=generated)
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
    (x_train, y_train), (_, _) = cifar10.load_data()
    x_train = (x_train.astype(np.float32) / 127.5) - 1

    discriminator = discriminator_model()
    # d_opt = Adam(lr=1e-5, beta_1=0.1)
    # discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)
    discriminator.load_weights('discriminator.h5')

    discriminator.trainable = False
    generator = generator_model()
    generator.load_weights('generator.h5')
    generator.trainable = False
    discriminator_hidden_model = Model(inputs=discriminator.input,
                                       outputs=discriminator.get_layer('last_hidden_layer').output)
    discriminator_hidden_model.trainable = False

    # encoder
    encoder = encoder_model(discriminator_hidden_model)

    # reverser
    reverser = reverser_model(encoder, generator)

    e_opt = Adam(lr=5e-4)

    reverser.compile(optimizer=e_opt,
                     loss='mean_squared_error')
    #
    # noise_inputs = Input(shape=(100,))
    # generated = generator(noise_inputs)
    # main_output = discriminator(generated)
    # aux_output = discriminator_hidden_model(generated)
    # dcgan = Model(inputs=noise_inputs, outputs=[main_output, aux_output])
    # g_opt = Adam(lr=2e-4, beta_1=0.5)
    #
    # dcgan.compile(optimizer=g_opt,
    #               loss=['binary_crossentropy', feature_matching_loss],
    #               loss_weights=[1.0, 1.0])

    num_batches = int(x_train.shape[0] / BATCH_SIZE)

    print('Number of batches:', num_batches)

    for epoch in range(NUM_EPOCH):
        X_train = np.random.permutation(x_train)
        for index in range(num_batches):
            # noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_images = reverser.predict(image_batch, verbose=0)
            #
            if index % 500 == 0:
                image = combine_images(image_batch)
                image = (image + 1) * 127.5
                if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.mkdir(GENERATED_IMAGE_PATH)
                Image.fromarray(image.astype(np.uint8)) \
                    .save(GENERATED_IMAGE_PATH + "{0}_{1}_true.png".format(epoch, index))

                image = combine_images(generated_images)
                image = (image + 1) * 127.5
                if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.mkdir(GENERATED_IMAGE_PATH)
                Image.fromarray(image.astype(np.uint8)) \
                    .save(GENERATED_IMAGE_PATH + "{0}_{1}.png".format(epoch, index))

            # X = np.concatenate((image_batch, generated_images))
            # y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            # d_loss = discriminator.train_on_batch(X, y)

            # expected_feature = discriminator_hidden_model.predict(image_batch)
            # noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])

            # labels = [1] * BATCH_SIZE
            # g_loss = dcgan.train_on_batch(noise, [np.array(labels), expected_feature])
            # print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss[0], d_loss))
            # generator.save_weights('generator.h5')
            # discriminator.save_weights('discriminator.h5')
            e_loss = reverser.train_on_batch(x=image_batch, y=image_batch)
            print("epoch: %d, batch: %d, e_loss: %f" % (epoch, index, e_loss))
        encoder.save_weights('encoder.h5')
        reverser.save_weights('reverser.h5')


if __name__ == '__main__':
    train()
