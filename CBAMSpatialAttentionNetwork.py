import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CBAMSpatialAttentionNetwork(object):
    def __init__(self, data_set):
        self.data_set = data_set

    # To train a neural network.
    def train(self):
        # Train an mnist model.
        if self.data_set == 'mnist' :
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

            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            input = keras.Input(shape=input_shape)
            conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(input)
            conv2 = Conv2D(32, (3, 3), activation='relu')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
            attn = SpatialModule()(pool1)
            res = Add()([pool1, attn]) # Residual connection
            conv4 = Conv2D(64, (3, 3), activation='relu')(res)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
            flat = Flatten()(pool2)
            dense1 = Dense(200, activation='relu')(flat)
            drop = Dropout(0.5)(dense1)
            dense2 = Dense(200, activation='relu')(drop)
            output = Dense(num_classes, activation='softmax')(dense2)

            model = keras.Model(inputs = input, outputs = output)
            
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
        else:
            print("Unsupported dataset %s. Try 'mnist'." % self.data_set)
            return
    
def SpatialModule():
    def spatial_module_on(inputs):
        (B, H, W, C) = inputs.shape

        # Pooling over the channel dimension
        avrg_pool = tf.math.reduce_mean(inputs, axis=3)
        print(avrg_pool.shape)
        avrg_pool = tf.reshape(avrg_pool, [-1, H, W, 1])
        max_pool = tf.math.reduce_max(inputs, axis=3)
        max_pool = tf.reshape(max_pool, [-1, H, W, 1])
        pools = tf.concat([avrg_pool, max_pool], axis=3)
        weights = Conv2D(filters=1, kernel_size=(7, 7), activation='sigmoid', padding='same')(pools)
        # [B, H, W, 1]

        broadcast = tf.tile(weights, [1, 1, 1, C])  # Broadcast the weights to all channels
        outputs = tf.multiply(inputs, broadcast) # Element-wise multiplication
        return outputs
    return spatial_module_on

# mnist
# 50 epochs
# Test loss: 0.2611531615257263
# Test accuracy: 0.92330002784729