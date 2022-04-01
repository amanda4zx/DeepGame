from __future__ import print_function
from tensorflow.keras import backend as K
import sys
import os
from multiprocessing import Process, Lock
import tensorflow as tf

from NeuralNetwork import *
from DataSet import *
from DataCollection import *
from upperbound import upperbound
from lowerbound import lowerbound

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

lock = Lock()

def makeArgs(dataSetName, tau, gameType, indices, eta):
    return [(dataSetName, tau, gameType, index, eta, network_type, lock)
        for index in indices
        for network_type in ['seq', 'self_attn', 'cbam_spatial_attn']]

# the first way of defining parameters
numArgs = len(sys.argv)
if numArgs >= 9:

    if sys.argv[1] == 'mnist' or sys.argv[1] == 'cifar10' or sys.argv[1] == 'gtsrb':
        dataSetName = sys.argv[1]
    else:
        print("please specify as the 1st argument the dataset: mnist or cifar10 or gtsrb")
        exit

    if sys.argv[2] == 'ub' or sys.argv[2] == 'lb':
        bound = sys.argv[2]
    else:
        print("please specify as the 2nd argument the bound: ub or lb")
        exit

    if sys.argv[3] == 'cooperative' or sys.argv[3] == 'competitive':
        gameType = sys.argv[3]
    else:
        print("please specify as the 3nd argument the game mode: cooperative or competitive")
        exit

    if sys.argv[4].isnumeric():
        image_index = int(sys.argv[4])
    elif sys.argv[4] != 'multi' and numArgs > 9:
        print("please specify as the 4th argument the index of the image: [int], or 'multi' in the case of multiple inputs")
        exit

    if sys.argv[5] == 'L0' or sys.argv[5] == 'L1' or sys.argv[5] == 'L2':
        distanceMeasure = sys.argv[5]
    else:
        print("please specify as the 5th argument the distance measure: L0, L1, or L2")
        exit

    if isinstance(float(sys.argv[6]), float):
        distance = float(sys.argv[6])
    else:
        print("please specify as the 6th argument the distance: [int/float]")
        exit
    eta = (distanceMeasure, distance)

    if isinstance(float(sys.argv[7]), float):
        tau = float(sys.argv[7])
    else:
        print("please specify as the 7th argument the tau: [int/float]")
        exit
    
    if sys.argv[8] == 'seq' or sys.argv[8] == 'self_attn' or sys.argv[8] == 'cbam_spatial_attn':
        network_type = sys.argv[8]
    elif sys.argv[8] != 'multi' or numArgs == 9:
        print("please specify as the 8th argument the type of neural network: seq for Sequential, self_attn for dot-product self attention, cbam_spatial_attn for spatial attention in CBAM, or multi in the case of multiprocessing")
        exit

    if numArgs > 9:
        indices = []
        for i in range(9,numArgs):
            if sys.argv[i].isnumeric():
                indices.append(int(sys.argv[i]))
            else:
                print("please specify input image indices as the arguments after the 8th argument")

elif len(sys.argv) == 1:
    # the second way of defining parameters
    dataSetName = 'cifar10'
    bound = 'lb'
    gameType = 'cooperative'
    image_index = 213
    eta = ('L2', 10)
    tau = 1
    network_type = 'seq'

# calling algorithms
# dc = DataCollection("%s_%s_%s_%s_%s_%s_%s" % (dataSetName, bound, tau, gameType, image_index, eta[0], eta[1]))
# dc.initialiseIndex(image_index)
try:
    if bound == 'ub':
        if numArgs > 9:
            allArgs = makeArgs(dataSetName, tau, gameType, indices, eta)
            print(allArgs)
            ps = [Process(target=upperbound, args=a) for a in allArgs]
            for p in ps:
                p.start()
            for p in ps:
                p.join()
        else:
            p = Process(target=upperbound, args=(dataSetName, tau, gameType, image_index, eta, network_type, lock))
            p.start()
            p.join()
    elif bound == 'lb':
        if numArgs > 9:
            allArgs = makeArgs(dataSetName, tau, gameType, indices, eta)
            print(allArgs)
            ps = [Process(target=lowerbound, args=a) for a in allArgs]
            for p in ps:
                p.start()
            for p in ps:
                p.join()
        else:
            p = Process(target=lowerbound, args=(dataSetName, tau, gameType, image_index, eta, network_type, lock))
            p.start()
    else:
        print("Unrecognised bound setting.\n"
            "Try 'ub' for upper bound or 'lb' for lower bound.\n")
        exit
except KeyboardInterrupt:
    pass


# dc.provideDetails()
# dc.summarise()
# dc.close()

K.clear_session()
