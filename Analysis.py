import matplotlib.pyplot as plt

from Result import *
from basics import *

basicDir = '../data/gtsrb_saliency_10features_1.41ratio_MCTS/dataCollection/'

def plotBasic(dataSetName, bound, tau, gameType, image_index, eta, featureExtraction, numOfFeatures, explorationRate):
    seqRes = Result(basicDir, dataSetName, bound, tau, gameType, image_index, eta, 'seq')
    selfRes = Result(basicDir, dataSetName, bound, tau, gameType, image_index, eta, 'self_attn')
    cbamRes = Result(basicDir, dataSetName, bound, tau, gameType, image_index, eta, 'cbam_spatial_attn')
    minIters = seqRes.iters
    if(minIters > selfRes.iters):
        minIters = selfRes.iters
    if(minIters > cbamRes.iters):
        minIters = cbamRes.iters
    plt.clf()
    plotList(preprocess(seqRes.progress, minIters), 'seq_ub', 'orange')
    plotList(preprocess(selfRes.progress, minIters), 'self_ub', 'blue')
    plotList(preprocess(cbamRes.progress, minIters), 'cbam_ub', 'green')

    #TODO: plot lower bounds?

    plt.legend()
    plt.yscale('log')
    plt.ylabel(eta[0] + ' distance')
    plt.xlabel('iterations')
    plt.title('%s, input index %s' % (dataSetName, image_index))
    path = '../%s_%s_%s_%s/' % (dataSetName, featureExtraction, numOfFeatures, explorationRate)
    assure_path_exists(path)
    plt.savefig(path + '%s' % image_index)
    # plt.show()



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


f = open(basicDir + 'gtsrb_sample_indices.txt')
ls = f.readlines()
f.close()
for l in ls:
    if l[0] == '[':
        idx = int(l.strip('[]').split(', ')[0])
        try:
            plotBasic('gtsrb', 'ub', float(1), 'cooperative', idx, ('L2', float(10)), 'saliency', 10, 1.41)
        except:
            print(str(idx) + " problem!")