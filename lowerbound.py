#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Construct a 'lowerbound' function to compute
the lower bound of Player I’s minimum adversary distance
while Player II being cooperative, or Player I's maximum
adversary distance whilst Player II being competitive.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

from CooperativeAStar import *
from CompetitiveAlphaBeta import *
from NeuralNetwork import *
from DataSet import *
from DataCollection import *
from multiprocessing import Lock

from basics import *


def lowerbound(dataset_name, tau, game_type,image_index, eta, network_type, lock):
    NN = NeuralNetwork(dataset_name, network_type)
    lock.acquire()
    NN.load_network()
    lock.release()
    nprint("Dataset is %s." % NN.data_set)
    NN.model.summary()

    lock.acquire()
    dataset = DataSet(dataset_name, 'testing')
    lock.release()
    image = dataset.get_input(image_index)
    (label, confidence) = NN.predict(image)
    label_str = NN.get_label(int(label))
    nprint("Working on input with index %s, whose class is '%s' and the confidence is %s."
          % (image_index, label_str, confidence))
    nprint("The second player is being %s." % game_type)

    path = "%s_pic/idx_%s_label_[%s]_with_confidence_%s.png" % (
        dataset_name, image_index, label_str, confidence)
    NN.save_input(image, path)

    dc = DataCollection("%s_lb_%s_%s_%s_%s_%s_%s" % (dataset_name, game_type, image_index, eta[0], eta[1], tau, network_type))

    is_interrupted = False
    if game_type == 'cooperative':
        tic = time.time()
        cooperative = CooperativeAStar(dataset_name, image_index, image, NN, eta, tau)
        try:
            cooperative.play_game(image)
        except KeyboardInterrupt:
            is_interrupted = True
            pass

        elapsed = time.time() - tic
        if cooperative.ADVERSARY_FOUND is True:
            adversary = cooperative.ADVERSARY
            adv_label, adv_confidence = NN.predict(adversary)
            adv_label_str = NN.get_label(int(adv_label))

            nprint("\nFound an adversary within pre-specified bounded computational resource. "
                  "\nThe following is its information: ")
            nprint("difference between images: %s" % (diffImage(image, adversary)))
            l2dist = l2Distance(image, adversary)
            l1dist = l1Distance(image, adversary)
            l0dist = l0Distance(image, adversary)
            percent = diffPercent(image, adversary)
            nprint("L2 distance %s" % l2dist)
            nprint("L1 distance %s" % l1dist)
            nprint("L0 distance %s" % l0dist)
            nprint("manipulated percentage distance %s" % percent)
            nprint("class is changed into '%s' with confidence %s\n" % (adv_label_str, adv_confidence))

            dc.addComment("Found an adversarial example\n")
            dc.addComment("Class is changed into '%s' with confidence %s\n" % (adv_label_str, adv_confidence))
            dc.addComment("Difference between images: %s\n" % (diffImage(image, adversary)))

            path = "%s_pic/idx_%s_modified_into_[%s]_with_confidence_%s.png" % (
                dataset_name, image_index, adv_label_str, adv_confidence)
            NN.save_input(adversary, path)
            if eta[0] == 'L0':
                dist = l0dist
            elif eta[0] == 'L1':
                dist = l1dist
            elif eta[0] == 'L2':
                dist = l2dist
            else:
                nprint("Unrecognised distance metric.")
            path = "%s_pic/idx_%s_modified_diff_%s=%s_time=%s.png" % (
                dataset_name, image_index, eta[0], dist, elapsed)
            NN.save_input(np.absolute(image - adversary), path)
        else:
            if is_interrupted:
                nprint("\nInterrupted after %s minutes\n" % (elapsed/60))
                dc.addComment("Interrupted after %s minutes\n" % (elapsed/60))
            else:
                nprint("Adversarial distance exceeds distance budget.")
                dc.addComment("Adversarial distance exceeds distance budget.\n")

            newimage = cooperative.CURRENT_BEST_IMAGE
            new_label, new_confidence = NN.predict(newimage)
            l2dist = l2Distance(image, newimage)
            l1dist = l1Distance(image, newimage)
            l0dist = l0Distance(image, newimage)
            percent = diffPercent(image, newimage)
            dc.addComment("Current best safe manipulation has confidence %s\n" % new_confidence)
            # nprint("difference between images: %s" % (diffImage(image, newimage)))
            # dc.addComment("Difference between images: %s\n" % (diffImage(image, newimage)))

            path = "%s_pic/idx_%s_safe_with_%s_distance_%s.png" % (
                dataset_name, image_index, eta[0], cooperative.CURRENT_SAFE[-1])
            NN.save_input(newimage, path)

        nprint("\nNumber of iterations: %s\n" % cooperative.NITERS)
        dc.addComment("Number of iterations: %s\n" % cooperative.NITERS)
        dc.addRunningTime(elapsed)
        dc.addl2Distance(l2dist)
        dc.addl1Distance(l1dist)
        dc.addl0Distance(l0dist)
        dc.addManipulationPercentage(percent)
        dc.addComment("Progress: %s\n" % cooperative.PROGRESS)

    elif game_type == 'competitive':
        competitive = CompetitiveAlphaBeta(image, NN, eta, tau)
        competitive.play_game(image)

    else:
        nprint("Unrecognised game type. Try 'cooperative' or 'competitive'.")
