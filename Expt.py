# Code for random sampling of input indices

import matplotlib.pyplot as plt
import numpy as np
import random
import os
from DataSet import *
from NeuralNetwork import *
from DataCollection import *

class Expt:
    def __init__(self):
        self.getIndicesForMnist()
        self.getIndicesForGTSRB()
        return

    # Randomly sample nsamples indices of each class such that all types of networks classified them correctly
    # and store the sampled indices in a file
    def getIndicesForMnist(self):
        nsamples = 5
        ds = DataSet('mnist', 'test')
        dc = DataCollection('mnist_sample_indices')

        seqnn = NeuralNetwork('mnist', 'seq')
        selfnn = NeuralNetwork('mnist', 'self_attn')
        cbamnn = NeuralNetwork('mnist', 'cbam_spatial_attn')
        seqnn.load_network()
        selfnn.load_network()
        cbamnn.load_network()

        # print("#dims of test data for mnist = ", ds.y.shape)
        # print(np.argmax(ds.y, axis = 1).shape)
        ys = np.argmax(ds.y, axis = 1)  # extract labels from one-hot encoding
        gts = range(np.amin(ys), np.amax(ys)+1) # ground truths
        allIndices = []
        for gt in gts:
            indices = []
            gt_ys = np.nonzero(ys == gt)[0] # indices with ground truth = gt
            while len(indices) < nsamples:
                i = random.choice(gt_ys)
                if i in indices:
                    continue
                image = ds.get_input(i)
                seq_y, c1 = seqnn.predict(image)
                self_y, c2 = selfnn.predict(image)
                cbam_y, c3 = cbamnn.predict(image)
                # print(i, seq_y, c1, self_y, c2, cbam_y, c3)
                if seq_y != gt or self_y != gt or cbam_y != gt:
                    continue
                else:
                    indices.append(i)
            allIndices.append(indices)
        for gt in range(len(allIndices)):
            dc.addComment('Ground truth: %s; Label: [%s]\n' % (gt, seqnn.get_label(gt)))
            dc.addComment('%s\n\n' % allIndices[gt])
        return

    def getIndicesForGTSRB(self):
        nsamples = 5
        ds = DataSet('gtsrb', 'test')
        dc = DataCollection('gtsrb_sample_indices')

        seqnn = NeuralNetwork('gtsrb', 'seq')
        selfnn = NeuralNetwork('gtsrb', 'self_attn')
        cbamnn = NeuralNetwork('gtsrb', 'cbam_spatial_attn')
        seqnn.load_network()
        selfnn.load_network()
        cbamnn.load_network()

        # print("#test data for gtsrb = ", ds.y.shape)
        # print(np.argmax(ds.y, axis = 1).shape)
        ys = np.argmax(ds.y, axis = 1)  # extract labels from one-hot encoding
        gts = range(np.amin(ys), np.amax(ys)+1) # ground truths
        allIndices = []
        for gt in gts:
            indices = []
            gt_ys = np.nonzero(ys == gt)[0] # indices with ground truth = gt
            while len(indices) < nsamples:
                i = random.choice(gt_ys)
                if i in indices:
                    continue
                image = ds.get_input(i)
                seq_y, c1 = seqnn.predict(image)
                self_y, c2 = selfnn.predict(image)
                cbam_y, c3 = cbamnn.predict(image)
                # print(i, seq_y, c1, self_y, c2, cbam_y, c3)
                if seq_y != gt or self_y != gt or cbam_y != gt:
                    continue
                else:
                    indices.append(i)
            allIndices.append(indices)
        for gt in range(len(allIndices)):
            dc.addComment('Ground truth: %s; Label: [%s]\n' % (gt, seqnn.get_label(gt)))
            dc.addComment('%s\n\n' % allIndices[gt])
        return

# ex = Expt()
