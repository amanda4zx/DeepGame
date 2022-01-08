"""
Construct a NeuralNetwork class to include operations
related to various datasets and corresponding models.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

# import cv2
# import copy
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as Image

from basics import assure_path_exists
from SequentialNetwork import *
from SelfAttentionNetwork import *
from CBAMSpatialAttentionNetwork import *
# from DataSet import *


# Define a Neural Network class.
class NeuralNetwork:
    # Specify which dataset at initialisation.
    def __init__(self, data_set, network_type):
        self.data_set = data_set
        self.network_type = network_type
        self.dir = network_type + '/'
        assure_path_exists(self.dir + "%s_pic/" % self.data_set)

    def predict(self, image):
        image = np.expand_dims(image, axis=0)
        predict_value = self.model.predict(image)
        new_class = np.argmax(np.ravel(predict_value))
        confident = np.amax(np.ravel(predict_value))
        return new_class, confident

    # To train a neural network and save it
    def train_network(self):
        if self.network_type == 'seq':
            network = SequentialNetwork(self.data_set)
        elif self.network_type == 'self_attn':
            network = SelfAttentionNetwork(self.data_set)
        elif self.network_type == 'cbam_spatial_attn':
            network = CBAMSpatialAttentionNetwork(self.data_set)
        
        network.train()
        self.model = network.model
        self.save_network()

    # To save the neural network to disk.
    def save_network(self):
        if self.data_set == 'mnist':
            self.model.save(self.dir + 'models/' + 'mnist')
            print("Neural network saved to disk.")

        elif self.data_set == 'cifar10':
            self.model.save(self.dir + 'models/' + 'cifar10')
            print("Neural network saved to disk.")

        elif self.data_set == 'gtsrb':
            self.model.save(self.dir + 'models/'+ 'gtsrb')
            print("Neural network saved to disk.")

        else:
            print("save_network: Unsupported dataset.")

    # To load a neural network from disk.
    def load_network(self):
        if self.data_set == 'mnist':
            self.model = models.load_model(self.dir + 'models/' + 'mnist')
            print("Neural network loaded from disk.")

        elif self.data_set == 'cifar10':
            self.model = models.load_model(self.dir + 'models/' + 'cifar10')
            print("Neural network loaded from disk.")

        elif self.data_set == 'gtsrb':
            try:
                self.model = models.load_model(self.dir + 'models/' + 'gtsrb')
                print("Neural network loaded from disk.")
            except (IOError, OSError):
                self.train_network()

        else:
            print("load_network: Unsupported dataset.")

    def save_input(self, image, filename):
        image = Image.array_to_img(image.copy())
        plt.imsave(self.dir + filename, image, cmap='gray')
        # causes discrepancy
        # image_cv = copy.deepcopy(image)
        # cv2.imwrite(filename, image_cv * 255.0, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def get_label(self, index):
        if self.data_set == 'mnist':
            labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        elif self.data_set == 'cifar10':
            labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.data_set == 'gtsrb':
            labels = ['speed limit 20 (prohibitory)', 'speed limit 30 (prohibitory)',
                      'speed limit 50 (prohibitory)', 'speed limit 60 (prohibitory)',
                      'speed limit 70 (prohibitory)', 'speed limit 80 (prohibitory)',
                      'restriction ends 80 (other)', 'speed limit 100 (prohibitory)',
                      'speed limit 120 (prohibitory)', 'no overtaking (prohibitory)',
                      'no overtaking (trucks) (prohibitory)', 'priority at next intersection (danger)',
                      'priority road (other)', 'give way (other)', 'stop (other)',
                      'no traffic both ways (prohibitory)', 'no trucks (prohibitory)',
                      'no entry (other)', 'danger (danger)', 'bend left (danger)',
                      'bend right (danger)', 'bend (danger)', 'uneven road (danger)',
                      'slippery road (danger)', 'road narrows (danger)', 'construction (danger)',
                      'traffic signal (danger)', 'pedestrian crossing (danger)', 'school crossing (danger)',
                      'cycles crossing (danger)', 'snow (danger)', 'animals (danger)',
                      'restriction ends (other)', 'go right (mandatory)', 'go left (mandatory)',
                      'go straight (mandatory)', 'go right or straight (mandatory)',
                      'go left or straight (mandatory)', 'keep right (mandatory)',
                      'keep left (mandatory)', 'roundabout (mandatory)',
                      'restriction ends (overtaking) (other)', 'restriction ends (overtaking (trucks)) (other)']
        else:
            print("LABELS: Unsupported dataset.")
        return labels[index]

    # Get softmax logits, i.e., the inputs to the softmax function of the classification layer,
    # as softmax probabilities may be too close to each other after just one pixel manipulation.
    def softmax_logits(self, manipulated_images, batch_size=512):
        model = self.model

        # func = K.function([model.layers[0].input] + [K.learning_phase()],
        #                   [model.layers[model.layers.__len__() - 1].output.op.inputs[0]])

        # partial_model = keras.Model(model.inputs, model.layers[model.layers.__len__() - 1].output.op.inputs[0])

        # Assume that the last layer is Softmax()
        partial_model = keras.Model(model.inputs, model.layers[model.layers.__len__() - 2].output)
        # print("Extracted partial model for softmax logits")
        # partial_model.summary()

        if len(manipulated_images) >= batch_size:
            softmax_logits = []
            batch, remainder = divmod(len(manipulated_images), batch_size)
            for b in range(batch):
                logits = partial_model.predict(manipulated_images[b * batch_size:(b + 1) * batch_size])
                softmax_logits.append(logits)
            softmax_logits = np.asarray(softmax_logits)
            softmax_logits = softmax_logits.reshape(batch * batch_size, model.output_shape[1])

            if remainder > 0:
                logits = partial_model.predict(manipulated_images[batch * batch_size:len(manipulated_images)])
                softmax_logits = np.concatenate((softmax_logits, logits), axis=0)
        else:
            # softmax_logits = func([manipulated_images, 0])[0]
            softmax_logits = partial_model.predict(manipulated_images)

        # softmax_logits = func([manipulated_images, 0])[0]
        # print(softmax_logits.shape)
        return softmax_logits

# nn = NeuralNetwork('gtsrb', 'self_attn')
# nn.train_network()
