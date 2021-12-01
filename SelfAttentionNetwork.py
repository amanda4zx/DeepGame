import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SelfAttentionNetwork(object):
    # Specify which dataset at initialisation.
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
            conv3 = AugmentedConv2D(64, (3, 3), activation='relu', dk = 8, dv = 8)(pool1)
            # conv3 = BatchNormalization()(conv3) # apply batch normalisation after augmented conv as described in the paper
            conv4 = Conv2D(64, (3, 3), activation='relu')(conv3)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
            flat = Flatten()(pool2)
            dense1 = Dense(200, activation='relu')(flat)
            drop = Dropout(0.5)(dense1)
            dense2 = Dense(200, activation='relu')(drop)
            dense3 = Dense(num_classes, activation=None)(dense2)
            output = Softmax()(dense3)

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

# Attention module that applies the dot-product self-attention on a feature map inputs
def SelfAttn(dk, dv):
    def self_attn_on(inputs):
        (_, H, W, F_in) = inputs.shape
        # Matrix multiplications as 1x1 convolutions
        kqv = Conv2D(filters = 2 * dk + dv, kernel_size = 1)(inputs)
        k0, q0, v0 = tf.split(kqv, [dk, dk, dv], axis=3)

        flatten_hw = lambda x, d: tf.reshape(x, [-1, H * W, d])
        k = flatten_hw(k0, dk)
        q = flatten_hw(q0, dk)
        v = flatten_hw(v0, dv)

        # Scaled dot-product
        weights = tf.linalg.matmul(q, k, transpose_b = True) * (dk ** -0.5)
        weights = tf.nn.softmax(weights)

        attn_out = tf.linalg.matmul(weights, v)
        attn_out = tf.reshape(attn_out, [-1, H, W, dv])
        attn_out = Conv2D(filters=dv, kernel_size=1)(attn_out)
        return attn_out
    return self_attn_on

def AugmentedConv2D(filters, kernel_size, activation, dk, dv):
    def aug_conv2d_on(inputs):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')(inputs)
        attn = SelfAttn(dk, dv)(inputs)
        return tf.concat([conv, attn], axis = 3)
    return aug_conv2d_on
