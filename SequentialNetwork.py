import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from DataSet import *

class SequentialNetwork(object):
    def __init__(self, data_set):
        self.data_set = data_set

    def train(self):
        # Train an mnist model.
        if self.data_set == 'mnist':
            batch_size = 128
            num_classes = 10
            epochs = 50
            img_rows, img_cols = 28, 28

            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

            # Add the channel dimension to inputs to work with other DeepGame modules
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
            
            # plt.imshow(x_train[1].squeeze(axis=2), vmin = 0, vmax = 1, cmap='gray')
            # plt.show()

            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                Conv2D(32, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(200, activation='relu'),
                Dropout(0.5),
                Dense(200, activation='relu'),
                Dense(num_classes, activation=None),
                Softmax()
            ])

            model.summary()

            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_test, y_test))

            score = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])

            self.model = model

        # Train a cifar10 model.
        elif self.data_set == 'cifar10':
            batch_size = 128
            num_classes = 10
            epochs = 50
            img_rows, img_cols, img_chls = 32, 32, 3
            data_augmentation = True

            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_chls)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chls)
            input_shape = (img_rows, img_cols, img_chls)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255

            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            model = Sequential([
                Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dense(num_classes, activation=None),
                Softmax()
            ])

            model.summary()

            opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])

            if not data_augmentation:
                print("Not using data augmentation.")
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test),
                          shuffle=True)
            else:
                print("Using real-time data augmentation.")
                datagen = ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=0,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=False)

                datagen.fit(x_train)
                model.fit(datagen.flow(x_train, y_train,
                                                 batch_size=batch_size),
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    workers=4)

            scores = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", scores[0])
            print("Test accuracy:", scores[1])

            self.model = model

            # Train a gtsrb model.
        elif self.data_set == 'gtsrb':
            downscale = True
            batch_size = 128
            num_classes = 43
            epochs = 50
            if downscale:
                img_rows, img_cols, img_chls = 24, 24, 3
            else:
                img_rows, img_cols, img_chls = 48, 48, 3
            data_augmentation = True

            train = DataSet('gtsrb', 'training')
            x_train, y_train = train.x, train.y
            test = DataSet('gtsrb', 'test')
            x_test, y_test = test.x, test.y
            input_shape = (img_rows, img_cols, img_chls)

            model = Sequential([
                Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(256,activation='relu'),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dense(num_classes, activation=None),
                Softmax()
            ])

            model.summary()

            opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])

            if not data_augmentation:
                print("Not using data augmentation.")
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test),
                          shuffle=True)
            else:
                print("Using real-time data augmentation.")
                datagen = ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=0,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=False)

                datagen.fit(x_train)
                model.fit(datagen.flow(x_train, y_train,
                                                 batch_size=batch_size),
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    workers=4)

            scores = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", scores[0])
            print("Test accuracy:", scores[1])

            self.model = model

        else:
            print("Unsupported dataset %s. Try 'mnist' or 'cifar10' or 'gtsrb'." % self.data_set)
            return
