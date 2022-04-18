# Perform data analysis

import matplotlib.pyplot as plt

from Result import *
from basics import *

# the folder storing data
dataFolder = '../data/'

# plot the progress in bounds for all three models in the same axes
def plotBasic(dataSetName, bound, tau, gameType, image_index, eta, featureExtraction, numOfFeatures, explorationRate):
    basicDir = getDir(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate)
    seqRes = Result(basicDir, dataSetName, bound, tau, gameType, image_index, eta, 'seq')
    selfRes = Result(basicDir, dataSetName, bound, tau, gameType, image_index, eta, 'self_attn')
    cbamRes = Result(basicDir, dataSetName, bound, tau, gameType, image_index, eta, 'cbam_spatial_attn')
    minIters = min(seqRes.iters, selfRes.iters, cbamRes.iters)
    plt.clf()
    plotList(preprocess(seqRes.progress, minIters), 'seq', 'orange')
    plotList(preprocess(selfRes.progress, minIters), 'self', 'blue')
    plotList(preprocess(cbamRes.progress, minIters), 'cbam', 'green')

    plt.legend()
    plt.yscale('log')
    plt.ylabel(eta[0] + ' distance')
    plt.xlabel('iterations')
    plt.title('%s_%s_%s_%s_%sfeatures_%sratio' % (dataSetName, image_index, bound, featureExtraction, numOfFeatures, explorationRate))
    path = '%sanalysis/%s_%s_%s_%s_%s/' % (dataFolder, dataSetName, featureExtraction, numOfFeatures, explorationRate, bound)
    assure_path_exists(path)
    plt.savefig(path + '%s' % image_index)
    # plt.show()

# compute information about which model obtained the best bounds for a given input
def collateBestBounds(dataSetName, bound, tau, gameType, image_index, eta, featureExtraction, numOfFeatures, explorationRate):
    basicDir = getDir(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate)
    seqRes = Result(basicDir, dataSetName, bound, tau, gameType, image_index, eta, 'seq')
    selfRes = Result(basicDir, dataSetName, bound, tau, gameType, image_index, eta, 'self_attn')
    cbamRes = Result(basicDir, dataSetName, bound, tau, gameType, image_index, eta, 'cbam_spatial_attn')
    minIters = min(seqRes.iters, selfRes.iters, cbamRes.iters)

    seqProg = [y for (x, y) in preprocess(seqRes.progress, minIters)]
    selfProg = [y for (x, y) in preprocess(selfRes.progress, minIters)]
    cbamProg = [y for (x, y) in preprocess(cbamRes.progress, minIters)]
    # print(image_index)
    # print(seqProg)

    # the number of iterations in which each model has the best bound
    numBestIters = {'seq':0, 'self_attn':0, 'cbam_spatial_attn':0}

    if(bound == 'ub'):
        for i in range(len(seqProg)):
            if(seqProg[i] <= selfProg[i] and seqProg[i] <= cbamProg[i]):
                numBestIters['seq'] += 1
            if(selfProg[i] <= seqProg[i] and selfProg[i] <= cbamProg[i]):
                numBestIters['self_attn'] += 1
            if(cbamProg[i] <= seqProg[i] and cbamProg[i] <= selfProg[i]):
                numBestIters['cbam_spatial_attn'] += 1
    elif(bound == 'lb'):
        for i in range(len(seqProg)):
            if(seqProg[i] >= selfProg[i] and seqProg[i] >= cbamProg[i]):
                numBestIters['seq'] += 1
            if(selfProg[i] >= seqProg[i] and selfProg[i] >= cbamProg[i]):
                numBestIters['self_attn'] += 1
            if(cbamProg[i] >= seqProg[i] and cbamProg[i] >= selfProg[i]):
                numBestIters['cbam_spatial_attn'] += 1

    # the model(s) that obtained at least as good a bound as the other models
    bestBoundModels = []
    if(bound == 'ub'):
        if(seqProg[-1] <= selfProg[-1] and seqProg[-1] <= cbamProg[-1]):
            bestBoundModels.append('seq')
        if(selfProg[-1] <= seqProg[-1] and selfProg[-1] <= cbamProg[-1]):
            bestBoundModels.append('self_attn')
        if(cbamProg[-1] <= seqProg[-1] and cbamProg[-1] <= selfProg[-1]):
            bestBoundModels.append('cbam_spatial_attn')
    elif(bound == 'lb'):
        if(seqProg[-1] >= selfProg[-1] and seqProg[-1] >= cbamProg[-1]):
            bestBoundModels.append('seq')
        if(selfProg[-1] >= seqProg[-1] and selfProg[-1] >= cbamProg[-1]):
            bestBoundModels.append('self_attn')
        if(cbamProg[-1] >= seqProg[-1] and cbamProg[-1] >= selfProg[-1]):
            bestBoundModels.append('cbam_spatial_attn')

    mostAdvModels = []
    if(bound == 'ub'):
        seqNumAdv = seqRes.numOfAdversary
        selfNumAdv = selfRes.numOfAdversary
        cbamNumAdv = cbamRes.numOfAdversary
        mostAdv = max(seqNumAdv, selfNumAdv, cbamNumAdv)
        if(seqNumAdv == mostAdv):
            mostAdvModels.append('seq')
        if(selfNumAdv == mostAdv):
            mostAdvModels.append('self_attn')
        if(cbamNumAdv == mostAdv):
            mostAdvModels.append('cbam_spatial_attn')

    # print(numBestIters)
    # print(bestBoundModels)
    # print(mostAdv)
    return numBestIters, bestBoundModels, mostAdvModels

# collate data about the best bound across different number of features for a given input and a given model
def collateForFeatures(dataSetName, bound, tau, gameType, image_index, eta, featureExtraction, explorationRate, network_type):
    nums = [2,4,6,8,10]
    dirs = [getDir(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate) for numOfFeatures in nums]
    ress = [Result(d, dataSetName, bound, tau, gameType, image_index, eta, network_type) for d in dirs]
    numIters = [r.iters for r in ress]
    minIters = min(numIters)
    progs = [[y for (x, y) in preprocess(r.progress, minIters)] for r in ress]
    finalBounds = [progress[-1] for progress in progs]
    best = min(finalBounds)
    bestBoundNums = []
    for i in range(len(nums)):
        if finalBounds[i] == best:
            bestBoundNums.append(i)

    mostAdvNums = []
    numAdvs = [r.numOfAdversary for r in ress]
    mostAdv = max(numAdvs)
    for i in range(len(nums)):
        if(numAdvs[i] == mostAdv):
            mostAdvNums.append(i)

    # print(minIters)
    # print(bestBoundNums)
    # print(mostAdvNums)

    return bestBoundNums, mostAdvNums

# collate data about the best bound across different exploration-exploitation ratios for a given input and a given model
def collateForRatios(dataSetName, bound, tau, gameType, image_index, eta, featureExtraction, numOfFeatures, network_type):
    ratios = [0.5, 1.0, 1.41, 2.0, 4.0]
    dirs = [getDir(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate) for explorationRate in ratios]
    ress = [Result(d, dataSetName, bound, tau, gameType, image_index, eta, network_type) for d in dirs]
    numIters = [r.iters for r in ress]
    minIters = min(numIters)
    progs = [[y for (x, y) in preprocess(r.progress, minIters)] for r in ress]
    finalBounds = [progress[-1] for progress in progs]
    best = min(finalBounds)
    # print(image_index)
    # print(finalBounds)
    # print(best)
    bestBoundRatios = []
    for i in range(len(ratios)):
        if finalBounds[i] == best:
            bestBoundRatios.append(i)    # record the index

    mostAdvRatios = []
    numAdvs = [r.numOfAdversary for r in ress]
    mostAdv = max(numAdvs)
    for i in range(len(ratios)):
        if(numAdvs[i] == mostAdv):
            mostAdvRatios.append(i)

    # print(minIters)
    # print(bestBoundRatios)
    # print(mostAdvRatios)
    # print("===")
    return bestBoundRatios, mostAdvRatios

# collate data about the best bound across different feature extraction methods for a given input and a given model
def collateForExtractions(dataSetName, bound, tau, gameType, image_index, eta, numOfFeatures, explorationRate, network_type):
    extractions = ['saliency', 'sift']
    dirs = [getDir(dataSetName, bound, tau, gameType, eta, ex, numOfFeatures, explorationRate) for ex in extractions]
    ress = [Result(d, dataSetName, bound, tau, gameType, image_index, eta, network_type) for d in dirs]
    numIters = [r.iters for r in ress]
    minIters = min(numIters)
    progs = [[y for (x, y) in preprocess(r.progress, minIters)] for r in ress]
    finalBounds = [progress[-1] for progress in progs]
    best = min(finalBounds)

    bestBoundExtractions = []
    for i in range(len(extractions)):
        if finalBounds[i] == best:
            bestBoundExtractions.append(i)    # record the index

    mostAdvExtractions = []
    numAdvs = [r.numOfAdversary for r in ress]
    mostAdv = max(numAdvs)
    for i in range(len(extractions)):
        if(numAdvs[i] == mostAdv):
            mostAdvExtractions.append(i)

    return bestBoundExtractions, mostAdvExtractions

# plot the partitions (i.e., features) generated from the feature extraction, showing the number of manipulated channels on manipulated pixels
def plotManipulatedFeatures(dataSetName, bound, tau, gameType, image_index, eta, featureExtraction, numOfFeatures, explorationRate, network_type):
    basicDir = getDir(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate)
    res = Result(basicDir, dataSetName, bound, tau, gameType, image_index, eta, network_type)

    if featureExtraction == 'saliency':
        pattern = 'grey-box'
    elif featureExtraction == 'sift':
        pattern = 'black-box'
    plt.clf()
    res.plotManipulatedFeatures(pattern, numOfFeatures)
    plt.title('%s_%s_%s_%s_%sfeatures_%sratio' % (dataSetName, image_index, bound, featureExtraction, numOfFeatures, explorationRate))
    path = '%sfeaturePlots/%s_%s_%s_%s_%s/' % (dataFolder, dataSetName, featureExtraction, numOfFeatures, explorationRate, bound)
    assure_path_exists(path)
    plt.savefig(path + '%s_%s' % (image_index, network_type))

def plotList(ls, label, color):
    # ls is a list of pairs (#iterations, best bound)
    xs = [x[0] for x in ls]
    ys = [x[1] for x in ls]
    plt.plot(xs,ys,label = label, color = color)

# Plot one point for each iteration, so the lack of progress is shown by a flat line segment
# If there are (x,y1) and (x,y2) due to multipling samplings of the same iteration x,
# take the last one for x which is the best
# If no (x,_), use the one before x
def preprocess(ls, iters):
    ls2 = []
    j = 0
    for i in range(iters + 1):  # Start from the 0-th iteration
        if j >= len(ls) or ls[j][0] > i:
            ls2.append((i, ls2[-1][1]))
        else:
            while(j < len(ls)-1 and ls[j+1][0] == i):
                j += 1
            ls2.append(ls[j])
            j += 1
    return ls2

def getDir(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate):
    if featureExtraction == 'saliency':
        return '%s%s_%s_%sfeatures_%sratio_%s/dataCollection/' % (dataFolder, dataSetName, featureExtraction, int(numOfFeatures), float(explorationRate), bound)
    elif featureExtraction == 'sift':
        return '%s%s_%s_%sratio_%s/dataCollection/' % (dataFolder, dataSetName, featureExtraction, float(explorationRate), bound)

def getSampledIndices(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate):
    basicDir = getDir(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate)
    f = open(basicDir + 'gtsrb_sample_indices.txt')
    ls = f.readlines()
    f.close()
    ids = []
    for l in ls:
        if l[0] == '[':
            # We only used the first sampled index from each class
            idx = int(l.strip('[]').split(', ')[0])
            ids.append(idx)
    return ids

# for each sampled index, plot the convergence of bounds
def plotAllBasic(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate):
    ids = getSampledIndices(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate)
    for idx in ids:
        try:
            plotBasic(dataSetName, bound, tau, gameType, idx, eta, featureExtraction, numOfFeatures, explorationRate)
        except:
            print("Some model has no progress for image index %s" % idx)

# collate data about which model obtained the best bounds for all input samples
def collateAllBestBounds(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate):
    ids = getSampledIndices(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate)
    mostBestItersIds = {'seq':[], 'self_attn':[], 'cbam_spatial_attn':[]}
    bestBoundIds = {'seq':[], 'self_attn':[], 'cbam_spatial_attn':[]}
    mostAdvIds = {'seq':[], 'self_attn':[], 'cbam_spatial_attn':[]}
    classId = 0
    total = 0
    for idx in ids:
        try:
            numBestIters, bestBoundModels, mostAdvModels = collateBestBounds(dataSetName, bound, tau, gameType, idx, eta, featureExtraction, numOfFeatures, explorationRate)
            if(numBestIters['seq'] >= numBestIters['self_attn'] and numBestIters['seq'] >= numBestIters['cbam_spatial_attn']):
                mostBestItersIds['seq'].append((classId, idx))
            if(numBestIters['self_attn'] >= numBestIters['seq'] and numBestIters['self_attn'] >= numBestIters['cbam_spatial_attn']):
                mostBestItersIds['self_attn'].append((classId, idx))
            if(numBestIters['cbam_spatial_attn'] >= numBestIters['seq'] and numBestIters['cbam_spatial_attn'] >= numBestIters['self_attn']):
                mostBestItersIds['cbam_spatial_attn'].append((classId, idx))
            for network_type in bestBoundModels:
                bestBoundIds[network_type].append((classId, idx))
            for network_type in mostAdvModels:
                mostAdvIds[network_type].append((classId, idx))
            total += 1
        except TypeError:
            print("Some model has no progress for image index %s" % idx)
        classId += 1

    filename = '%sanalysis/collated/%s_%s_%s_%s_%s.txt' % (dataFolder, dataSetName, featureExtraction, numOfFeatures, explorationRate, bound)
    f = open(filename, "w")
    f.write("Classes and indices where each model had best bounds in the most iterations:\n")
    for k in mostBestItersIds.keys():
        f.write("%s: %s\n" % (k, mostBestItersIds[k]))
    f.write("\nClasses and indices where each model obtained the best bounds in the end:\n")
    for k in bestBoundIds.keys():
        f.write("%s: %s\n" % (k, bestBoundIds[k]))
    if(bound == 'ub'):
        f.write("\nClasses and indices where each model obtained the most adversarial examples:\n")
        for k in mostAdvIds.keys():
            f.write("%s: %s\n" % (k, mostAdvIds[k]))

    seqLen1 = len(mostBestItersIds['seq'])
    selfLen1 = len(mostBestItersIds['self_attn'])
    cbamLen1 = len(mostBestItersIds['cbam_spatial_attn'])
    f.write("\nPercentage of times seq was the best in number of best iterations: %s\n" % (seqLen1/total))
    f.write("Percentage of times self_attn was the best in number of best iterations: %s\n" % (selfLen1/total))
    f.write("Percentage of times cbam_spatial_attn was the best in number of best iterations: %s\n" % (cbamLen1/total))

    seqLen2 = len(bestBoundIds['seq'])
    selfLen2 = len(bestBoundIds['self_attn'])
    cbamLen2 = len(bestBoundIds['cbam_spatial_attn'])
    f.write("\nPercentage of times seq obtained the best bounds in the end: %s\n" % (seqLen2/total))
    f.write("Percentage of times self_attn obtained the best bounds in the end: %s\n" % (selfLen2/total))
    f.write("Percentage of times cbam_spatial_attn obtained the best bounds in the end: %s\n" % (cbamLen2/total))

    if(bound == 'ub'):
        seqLen3 = len(mostAdvIds['seq'])
        selfLen3 = len(mostAdvIds['self_attn'])
        cbamLen3 = len(mostAdvIds['cbam_spatial_attn'])
        f.write("\nPercentage of times seq obtained the most adversarial examples: %s\n" % (seqLen3/total))
        f.write("Percentage of times self_attn obtained the most adversarial examples: %s\n" % (selfLen3/total))
        f.write("Percentage of times cbam_spatial_attn obtained the most adversarial examples: %s\n" % (cbamLen3/total))
    f.close()

# collate data about which number of features obtained the best bounds for all input samples for a model
def collateAllForFeatures(dataSetName, bound, tau, gameType, eta, featureExtraction, explorationRate, network_type):
    ids = getSampledIndices(dataSetName, bound, tau, gameType, eta, featureExtraction, 10, explorationRate)
    nums = [2, 4, 6, 8, 10]
    bestBoundIds = {0:[], 1:[], 2:[], 3:[], 4:[]}
    mostAdvIds = {0:[], 1:[], 2:[], 3:[], 4:[]}
    total = 0
    classId = 0
    for idx in ids:
        try:
            bestBoundNums, mostAdvNums = collateForFeatures(dataSetName, bound, tau, gameType, idx, eta, featureExtraction, explorationRate, network_type)
            for numId in bestBoundNums:
                bestBoundIds[numId].append((classId, idx))
            for numId in mostAdvNums:
                mostAdvIds[numId].append((classId, idx))
            total += 1
        except TypeError:
            print("Some model has no progress for image index %s" % idx)
        classId += 1
    assure_path_exists('%sanalysis/collatedForFeatures/' % dataFolder)
    filename = '%sanalysis/collatedForFeatures/%s_%s_%s_%s_%s.txt' % (dataFolder, dataSetName, featureExtraction, explorationRate, bound, network_type)
    f = open(filename, "w")
    f.write("Classes and indices where each number of features had best final bounds:\n")
    for k in bestBoundIds.keys():
        f.write("%s: %s\n" % (nums[k], bestBoundIds[k]))
    f.write("\nClasses and indices where each number of features had the most adversarial examples:\n")
    for k in mostAdvIds.keys():
        f.write("%s: %s\n" % (nums[k], mostAdvIds[k]))

    f.write("\n")

    lens = [len(bestBoundIds[i]) for i in range(len(nums))]
    for i in range(len(nums)):
        f.write("Percentage of times %s was the best in the final bound: %s\n" % (nums[i], (lens[i]/total)))
    lens = [len(mostAdvIds[i]) for i in range(len(nums))]
    f.write("\n")
    for i in range(len(nums)):
        f.write("Percentage of times %s had the most adversarial examples: %s\n" % (nums[i], (lens[i]/total)))

# collate data about which ratio obtained the best bounds for all input samples for a model
def collateAllForRatios(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, network_type):
    ids = getSampledIndices(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, 1.41)
    ratios = [0.5, 1.0, 1.41, 2.0, 4.0]
    bestBoundIds = {0:[], 1:[], 2:[], 3:[], 4:[]}
    mostAdvIds = {0:[], 1:[], 2:[], 3:[], 4:[]}
    total = 0
    classId = 0
    for idx in ids:
        try:
            bestBoundRatios, mostAdvRatios = collateForRatios(dataSetName, bound, tau, gameType, idx, eta, featureExtraction, numOfFeatures, network_type)
            for ratioId in bestBoundRatios:
                bestBoundIds[ratioId].append((classId, idx))
            for ratioId in mostAdvRatios:
                mostAdvIds[ratioId].append((classId, idx))
            total += 1
        except TypeError:
            print("Some model has no progress for image index %s" % idx)
        classId += 1
    assure_path_exists('%sanalysis/collatedForRatios/' % dataFolder)
    filename = '%sanalysis/collatedForRatios/%s_%s_%s_%s_%s.txt' % (dataFolder, dataSetName, featureExtraction, numOfFeatures, bound, network_type)
    f = open(filename, "w")
    f.write("Classes and indices where each ratio had best final bounds:\n")
    for k in bestBoundIds.keys():
        f.write("%s: %s\n" % (ratios[k], bestBoundIds[k]))
    f.write("\nClasses and indices where each ratio had the most adversarial examples:\n")
    for k in mostAdvIds.keys():
        f.write("%s: %s\n" % (ratios[k], mostAdvIds[k]))

    f.write("\n")

    lens = [len(bestBoundIds[i]) for i in range(len(ratios))]
    for i in range(len(ratios)):
        f.write("Percentage of times %s was the best in the final bound: %s\n" % (ratios[i], (lens[i]/total)))
    lens = [len(mostAdvIds[i]) for i in range(len(ratios))]
    f.write("\n")
    for i in range(len(ratios)):
        f.write("Percentage of times %s had the most adversarial examples: %s\n" % (ratios[i], (lens[i]/total)))

# collate data about which feature extraction method obtained the best bounds for all input samples for a model
def collateAllForExtractions(dataSetName, bound, tau, gameType, eta, numOfFeatures, explorationRate, network_type):
    ids = getSampledIndices(dataSetName, bound, tau, gameType, eta, 'saliency', numOfFeatures, explorationRate)
    extractions = ['saliency', 'sift']
    bestBoundIds = {0:[], 1:[]}
    mostAdvIds = {0:[], 1:[]}
    total = 0
    classId = 0
    for idx in ids:
        try:
            bestBoundExtractions, mostAdvExtractions = collateForExtractions(dataSetName, bound, tau, gameType, idx, eta, numOfFeatures, explorationRate, network_type)
            for exId in bestBoundExtractions:
                bestBoundIds[exId].append((classId, idx))
            for exId in mostAdvExtractions:
                mostAdvIds[exId].append((classId, idx))
            total += 1
        except TypeError:
            print("Some model has no progress for image index %s" % idx)
        classId += 1
    assure_path_exists('%sanalysis/collatedForExtractions/' % dataFolder)
    filename = '%sanalysis/collatedForExtractions/%s_%s_%s_%s_%s.txt' % (dataFolder, dataSetName, numOfFeatures, explorationRate, bound, network_type)
    f = open(filename, "w")
    f.write("Classes and indices where each extraction had best final bounds:\n")
    for k in bestBoundIds.keys():
        f.write("%s: %s\n" % (extractions[k], bestBoundIds[k]))
    f.write("\nClasses and indices where each extraction had the most adversarial examples:\n")
    for k in mostAdvIds.keys():
        f.write("%s: %s\n" % (extractions[k], mostAdvIds[k]))

    f.write("\n")

    lens = [len(bestBoundIds[i]) for i in range(len(extractions))]
    for i in range(len(extractions)):
        f.write("Percentage of times %s was the best in the final bound: %s\n" % (extractions[i], (lens[i]/total)))
    lens = [len(mostAdvIds[i]) for i in range(len(extractions))]
    f.write("\n")
    for i in range(len(extractions)):
        f.write("Percentage of times %s had the most adversarial examples: %s\n" % (extractions[i], (lens[i]/total)))


def plotAllManipulatedFeatures(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate):
    ids = getSampledIndices(dataSetName, bound, tau, gameType, eta, featureExtraction, numOfFeatures, explorationRate)
    for idx in ids:
        for network_type in ['seq', 'self_attn', 'cbam_spatial_attn']:
            try:
                plotManipulatedFeatures(dataSetName, bound, tau, gameType, idx, eta, featureExtraction, numOfFeatures, explorationRate, network_type)
            except:
                print("%s has no progress for image index %s" % (network_type, idx))
