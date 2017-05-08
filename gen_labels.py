import numpy as np
import pickle
import sys
import os
from numpy import array
from random import shuffle
labelDict = {
    
    'classical' :   0,
    'club'      :   1,
    'dance'     :   2,
    'edm'       :   3,
    'heavy-metal'      :   4,
    'tango'     :   5,
}

if __name__ == "__main__":

    data = []
    labels = []

    print("Path : " + sys.argv[1])
    path = sys.argv[1]

    labelSize = len(labelDict)
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.mel'):
                filepath = os.path.join(dirpath, filename)

                with open(filepath, 'rb') as f:
                        inputFeatures = pickle.load(f)
                        featureMatrix = np.asarray(inputFeatures)
                        data.append(featureMatrix)
                        parentDir = os.path.dirname(filepath)
                        dirname = os.path.basename(parentDir) 
                        labelMatrix = [0] * labelSize
                        labelMatrix[labelDict[dirname]] = 1
                        labels.append(labelMatrix)

    combined = list(zip(data, labels))
    shuffle(combined)

    data[:], labels[:] = zip(*combined)
    size = len(data)
    trainsize = int(0.7 * size)
    traindata = {}
    for i in range(0,trainsize):
        traindata[i] = data[i]

    testdata = {}
    for i in range(trainsize,size):
        testdata[i] = data[i]

    trainlabel = labels[:trainsize]
    testlabel = labels[trainsize:]

    with open(sys.argv[2], 'wb') as f:
        f.write(pickle.dumps(traindata))

    with open(sys.argv[3], 'wb') as f:
        f.write(pickle.dumps(array(trainlabel)))

    with open(sys.argv[2] + "_test", 'wb') as f:
        f.write(pickle.dumps(testdata))

    with open(sys.argv[3] + "_test", 'wb') as f:
        f.write(pickle.dumps(array(testlabel)))
