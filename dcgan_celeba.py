# coding: utf-8
"""
dcgan + feature matchingの実装、CelebAで実験
モデル構成をNVIDIAのブログのものに
https://devblogs.nvidia.com/parallelforall/photo-editing-generative-adversarial-networks-2/
"""

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten
from keras.initializers import TruncatedNormal
import math
import numpy as np
import os
from keras.optimizers import Adam
from PIL import Image
from keras import backend as K


BATCH_SIZE = 64
NUM_EPOCH = 60
GENERATED_IMAGE_PATH = 'celeb_generated_images/'  # 生成画像の保存先
DATA_PATH = ''
init = TruncatedNormal(stddev=0.02)


def generator_model():
    # input
    # (N, 100)
    model = Sequential()
    model.add(Dense(8192, input_dim=100, kernel_initializer=init))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(Activation('relu'))

    # (N, 4, 4, 512)
    model.add(Reshape((4, 4, 512)))

    # (N, 8, 8, 256)
    model.add(Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same',
                              kernel_initializer=init))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(Activation('relu'))

    # (N, 16, 16, 128)
    model.add(Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same',
                              kernel_initializer=init))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(Activation('relu'))

    # (N, 32, 32, 64)
    model.add(Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same',
                              kernel_initializer=init))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(Activation('relu'))

    # output
    # (N, 64, 64, 3)
    model.add(Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same',
                              kernel_initializer=init))
    model.add(Activation('tanh'))

    return model


def discriminator_model():
    # (N, 64, 64, 3) input
    model = Sequential()

    # (N, 32, 32, 64)
    model.add(Conv2D(filters=64, kernel_size=5, padding='same', strides=2, input_shape=(64, 64, 3),
                     kernel_initializer='TruncatedNormal'))
    model.add(LeakyReLU(0.2))

    # (N, 16, 16, 128)
    model.add(Conv2D(filters=128, kernel_size=5, padding='same', strides=2,
                     kernel_initializer=init))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(LeakyReLU(0.2))

    # (N, 8, 8, 256)
    model.add(Conv2D(filters=256, kernel_size=5, padding='same', strides=2,
                     kernel_initializer=init))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(LeakyReLU(0.2))

    # (N, 4, 4, 512)
    model.add(Conv2D(filters=512, kernel_size=5, padding='same', strides=2,
                     kernel_initializer=init))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(LeakyReLU(0.2))

    # (N, 8192)
    model.add(Flatten(name='last_hidden_layer'))

    # (N, 1) output
    model.add(Dense(1, kernel_initializer=init))
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
    x_train = np.load('cropped_celeba.npy')
    x_train = (x_train.astype(np.float32) / 127.5) - 1

    discriminator = discriminator_model()
    d_opt = Adam(lr=5e-4)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    discriminator.trainable = False
    generator = generator_model()

    # DCGAN with feature matching
    discriminator_hidden_model = Model(inputs=discriminator.input,
                                       outputs=discriminator.get_layer('last_hidden_layer').output)
    discriminator_hidden_model.trainable = False
    noise_inputs = Input(shape=(100,))
    generated = generator(noise_inputs)
    main_output = discriminator(generated)
    aux_output = discriminator_hidden_model(generated)
    dcgan = Model(inputs=noise_inputs, outputs=[main_output, aux_output])
    g_opt = Adam(lr=5e-4)

    dcgan.compile(optimizer=g_opt,
                  loss=['binary_crossentropy', feature_matching_loss],
                  loss_weights=[1.0, 1.0])

    num_batches = int(x_train.shape[0] / BATCH_SIZE)

    print('Number of batches:', num_batches)

    for epoch in range(NUM_EPOCH):
        X_train = np.random.permutation(x_train)
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

            d_loss_real = discriminator.train_on_batch(image_batch, [1] * BATCH_SIZE)
            d_loss_fake = discriminator.train_on_batch(generated_images, [0] * BATCH_SIZE)
            d_loss = (d_loss_real + d_loss_fake) / 2

            expected_feature = discriminator_hidden_model.predict(image_batch)
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])

            labels = [1] * BATCH_SIZE
            g_loss = dcgan.train_on_batch(noise, [np.array(labels), expected_feature])
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f, fm_loss: %f" % (epoch, index, g_loss[0], d_loss, g_loss[1]))
        generator.save_weights('celeb_generator.h5')
        discriminator.save_weights('celeb_discriminator.h5')


if __name__ == '__main__':
    train()
