from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

from data import Data

import sys
import os

import numpy as np


class GAN:
    def __init__(self):
        self.img_rows = 1
        self.img_cols = 1
        self.img_shape = (self.img_rows, self.img_cols)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(32, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=32, sample_interval=50):

        # Load the dataset
        data = Data()
        data.load_data()
        data.normalize()
        X_train = data.train_data

        X_train = np.expand_dims(X_train, axis=1)
        X_train = np.expand_dims(X_train, axis=1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if (epoch + 1) % sample_interval == 0:
                if not os.path.exists("saved_model"):
                    os.makedirs("saved_model")
                self.generator.save_weights("saved_model/G_model_%d.hdf5" % epoch, True)
                self.discriminator.save_weights("saved_model/D_model_%d.hdf5" % epoch, True)

    def test(self):
        data = Data()
        data.load_data()
        max_num = data.get_max()

        self.generator.load_weights("saved_model/G_model_29999.hdf5", True)
        self.discriminator.load_weights("saved_model/D_model_29999.hdf5", True)

        test_data = []
        test_label = data.test_label

        threshold_list = [i for i in np.linspace(0.001550, 0.001650, 21)]
        precision = []

        for item in data.test_data:
            temp = [item]
            temp = np.expand_dims(temp, axis=1)
            temp = np.expand_dims(temp, axis=1)
            test_data.append(temp)

        for j in range(len(threshold_list)):
            mount = 0  # 正确判断的个数
            for i in range(len(test_data)):
                res = self.discriminator.predict(test_data[i]/max_num)

                # 计算差距
                diff = abs(res - 0.5) / 0.5

                # 如果差距大于 threshold，则为假，否则为真
                if diff < threshold_list[j]:
                    label = 1
                else:
                    label = 0
                print("判别数据：", test_data[i], "判别结果：", label, "真实结果：", test_label[i])
                if label == test_label[i]:
                    mount += 1

            print("准确率：", mount / len(test_data))
            precision.append(mount / len(test_data))

        plt.plot(threshold_list, precision, marker="*")
        plt.xticks(rotation=80)
        plt.show()

    def generator_data(self):
        self.generator.load_weights("saved_model/G_model_29999.hdf5", True)
        self.discriminator.load_weights("saved_model/D_model_29999.hdf5", True)

        result = []

        data = Data()
        data.load_data()
        data.normalize()

        for i in range(10000):
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            result.append(gen_imgs[0][0][0] * 44.52)

        print(result)
        plt.hist(result, bins=40)
        plt.show()


if __name__ == '__main__':
    gan = GAN()
    # gan.train(epochs=30000, batch_size=32, sample_interval=15000)
    gan.test()
    # gan.generator_data()
