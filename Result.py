# Extract data from a file containing the experiment log
from FeatureExtraction import *
from DataSet import *
from NeuralNetwork import *
from matplotlib import pyplot as plt

class Result:
    def __init__(self, dir, dataSetName, bound, tau, gameType, image_index, eta, network_type):
        self.fileName = dir + "%s_%s_%s_%s_%s_%s_%s_%s.txt" % (dataSetName, bound, gameType, image_index, eta[0], eta[1], tau, network_type)
        f = open(self.fileName, 'r')
        self.lines = f.readlines()
        f.close()

        self.dataSetName = dataSetName
        self.image_index = image_index
        self.network_type = network_type

        if bound == 'ub':
            self.iters = int(self.extract('Number of iterations: '))
            self.numOfSampling = int(self.extract('Number of sampling: '))

            foundAdversary0 = self.extract('Found')
            if foundAdversary0 == None:
                self.foundAdversary = False
            else:
                self.foundAdversary = True

            self.numOfAdversary = 0
            if(self.foundAdversary):
                self.numOfAdversary = int(self.extract('Number of adversarial examples: '))

            self.runningTime = float(self.extract('Running time: '))

            if(self.foundAdversary):
                self.l2 = float(self.extract('L2 distance: '))
                self.l1 = float(self.extract('L1 distance: '))
                self.l0 = float(self.extract('L0 distance: '))
            
            self.numOfRemovals = int(self.extract('Number of removals due to no useful child: '))

            self.progress = []
            if(self.foundAdversary):
                progress0 = self.extract('Progress: ')
                progress0 = ''.join(filter(lambda s: s not in "()[]\n", progress0)).split(', ')
                for i in range(0, len(progress0), 2):
                    self.progress.append((int(progress0[i]), float(progress0[i+1])))
            else:
                self.progress.append((0, eta[1]))
            
            self.diffBetweenImages = []
            if(self.foundAdversary):
                diff0 = self.extract('Difference between images: ')
                diff0 = diff0.strip('[]\n').split(' ')
                diff0 = [int(s.strip('(),')) for s in diff0]
                self.diffBetweenImages = [(diff0[i], diff0[i+1], diff0[i+2]) for i in range(0, len(diff0), 3)]
        else:
            self.iters = int(self.extract('Number of iterations: '))

            self.runningTime = float(self.extract('Running time: '))

            self.progress = []
            progress0 = self.extract('Progress: ')
            progress0 = ''.join(filter(lambda s: s not in "()[]\n", progress0)).split(', ')
            for i in range(0, len(progress0), 2):
                self.progress.append((int(progress0[i]), float(progress0[i+1])))
    
    def extract(self, s):
        for l in self.lines:
            if self.match(s, l):
                return l[len(s):]
        return None

    def match(self, str1, str2):
        l1 = len(str1)
        l2 = len(str2)
        if(l1 < l2):
            l = l1
        else:
            l = l2
        same = True
        for i in range(l):
            if(str1[i] != str2[i]):
                same = False
        return same

    def plotFeatures(self, pattern='grey-box', num_partitions=10):
        ex = FeatureExtraction(pattern)
        dataset = DataSet(self.dataSetName, 'testing')
        self.image = dataset.get_input(self.image_index)
        self.nn = NeuralNetwork(self.dataSetName, self.network_type)
        self.nn.load_network()
        kps = ex.get_key_points(self.image, num_partitions)
        partitions = ex.get_partitions(self.image, self.nn, num_partitions)
        ex.plot_saliency_map(self.image, partitions, 'dataCollection/featMap.jpg')

    def plotManipulatedFeatures(self, pattern='grey-box', num_partitions=10):
        self.plotFeatures(pattern, num_partitions)
        num_manips = {}
        for (x, y, c) in self.diffBetweenImages:
            if (x, y) in num_manips.keys():
                num_manips[(x,y)] += 1
            else:
                num_manips[(x,y)] = 1
        # Show the number of manipulations at each manipulated pixel
        for (x, y) in num_manips.keys():
            plt.text(x, y, num_manips[(x,y)],ha="center", va="center", color="w")
        # plt.savefig('dataCollection/featMap.jpg')

# r = Result('dataCollection/','gtsrb', 'ub', float(1), 'cooperative', 8432, ('L2', float(10)), 'seq')
# print(r.iters)
# print(r.numOfSampling)
# print(r.foundAdversary)
# print(r.numOfAdversary)
# print(r.runningTime)
# print(r.l2)
# print(r.l1)
# print(r.l0)
# print(r.numOfRemovals)
# print(r.progress)
# print(r.diffBetweenImages)
# r.plotManipulatedFeatures()
