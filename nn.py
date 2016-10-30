from sklearn.cross_validation import train_test_split


import numpy as np
from ocr import ocr

train_indices, indices = train_test_split(list(range(5000)))

#small test function; the inner loop
# tests each combination of # hidden nodes
# a 100 times, and the outerloop takes the
# average of those 100 times, and uses that
# as the general accuracy for that many hidden nodes in the ANN
def test(dm, dl, indices, nn):
    avg = 0
    for i in xrange(100):
        rightguesses = 0
        for j in indices:
            test = dm[j]
            prediction = nn.predict(test)
            if dl[j] == prediction:
                rightguesses += 1

        avg += (rightguesses / float(len(indices)))
    return avg / 100

#opens the data sets
dm = np.loadtxt(open('data.csv', 'rb'), delimiter = ',').tolist()
dl = np.loadtxt(open('dataLabels.csv', 'rb')).tolist()

#goes through every combination of # of hidden nodes and checks
# the accuracy of the ANN (prints it out)
for i in xrange(5, 100, 10):
    nn = ocr(i, indices, dl, dm, False)
    performance = test(dm, dl, indices, nn)
    print "" + str(i) + " Hidden Nodes --> " + str(performance) 


