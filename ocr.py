import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import matrix
from math import pow
from collections import namedtuple
import math
import random
import os
import json
from PIL import Image

#By Kushal Tirumala and Bryan Chen
class ocr:
    lr = 0.1
    width = 20
    filepath = 'nn.json'
    use_file = True

    def __init__(self, hiddenNodesNumber, trindices, labelData, dmatrix, use_file):
        self.sigmoid = np.vectorize(self.sigmoidScalar)
        self.sigmoid_prime = np.vectorize(self.sigmoidPrime)
        self.useFile = use_file
        self.dmatrix = dmatrix
        self.labelData = labelData

        if (not os.path.isfile(ocr.filepath) or not use_file):
            self.val1 = self.weightInit(400, hiddenNodesNumber)

            self.val2 = self.weightInit(hiddenNodesNumber, 10)

            self.ilBias = self.weightInit(1, hiddenNodesNumber)

            self.hlBias = self.weightInit(1, 10)

            td = namedtuple('td', ['y0', 'label'])

            self.train([td(self.data_matrix[i], int(self.labelData[i])) for i in trindices])


            self.save()

        else:

            self._load()


    #draws the image
    def draw(self, sample):
        pixelArray = [sample[j:j+self.width] for j in xrange(0, len(sample), self.width)]
        plt.imshow(zip(*pixelArray), cmap = cm.Greys_r, interpolation="nearest")
        plt.show()


    
    #activation function 
    def sigmoidScalar(self, z):
        return math.exp(-np.logaddexp(0, -z))

    
    #trains the ANN with the csv files
    def train(self, tdArray):
        for data in tdArray:
            y1 = np.dot(np.mat(self.val1), np.mat(data[0]).T)
            sum1 =  y1 + np.mat(self.ilBias) 
            y1 = self.sigmoid(sum1)

            y2 = np.dot(np.array(self.val2), y1)
            y2 = np.add(y2, self.hlBias) 
            y2 = self.sigmoid(y2)

            valActuals = [0] * 10
            valActuals[data[1]] = 1
            errorOutput = np.mat(valActuals).T - np.mat(y2)
            errorHidden = np.multiply(np.dot(np.mat(self.val2).T, errorOutput), self.sigmoid_prime(sum1))

            self.val1 += self.lr * np.dot(np.mat(errorHidden), np.mat(data[0]))
            self.val2 += self.lr * np.dot(np.mat(errorOutput), np.mat(y1).T)
            self.hlBias += self.lr * errorOutput
            self.ilBias += self.lr * errorHidden


    #saves the current ANN to the nn.json file
    def save(self):
        if not self.useFile:
            return

        json_neural_network = {
            "val1":[np_mat.tolist()[0] for np_mat in self.val1],
            "val2":[np_mat.tolist()[0] for np_mat in self.val2],
            "b1":self.ilBias[0].tolist()[0],
            "b2":self.hlBias[0].tolist()[0]
        };
        with open(ocr.NN_FILE_PATH,'w') as nnFile:
            json.dump(json_neural_network, nnFile)


    #loads the weights + biases from the previous ANN
    def _load(self):
        if not self.useFile:
            return

        with open(ocr.filepath) as nnFile:
            nn = json.load(nnFile)
        self.val1 = [np.array(li) for li in nn['val1']]
        self.val2 = [np.array(li) for li in nn['val2']]
        self.ilBias = [np.array(nn['b1'][0])]
        self.hlBias = [np.array(nn['b2'][0])]


    #returns the derivative of the sigmoid function for a given z
    def sigmoidPrime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))


    #flips the input image
    def flip(self, img):
        width = img.size[0]
        height = img.size[1]
        for y in range(height):
            for x in range(width // 2):
                left = img.getpixel((x, y))
                right = img.getpixel((width - 1 - x, y))
                img.putpixel((width - 1 - x, y), left)
                img.putpixel((x, y), right)


    #predicts a letter based off the 1-D array (of size 400 for a 20px by 20px pic)
    def predict(self, test):
        print((test))

        y1 = np.dot(np.mat(self.val1), np.mat(test).T)
        y1 =  y1 + np.mat(self.ilBias) 
        y1 = self.sigmoid(y1)
        print(y1)

        y2 = np.dot(np.array(self.val2), y1)
        y2 = np.add(y2, self.hlBias) 
        y2 = self.sigmoid(y2)
        print(y2)

        results = y2.T.tolist()[0]
        print(results)
        return results.index(max(results))

    #initalizes the weights in the ANN            
    def weightInit(self, input, out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(out, input)]


#Test
HIDDEN_NODE_COUNT = 15
data_matrix = np.loadtxt(open('data.csv', 'rb'), delimiter = ',')
labelData = np.loadtxt(open('dataLabels.csv', 'rb'))

data_matrix = data_matrix.tolist()
labelData = labelData.tolist()

nn = ocr(HIDDEN_NODE_COUNT, list(range(5000)), labelData, data_matrix, True);
im = Image.open('pic7.png')
im = im.rotate(-90);
nn.flip(im)
gry = im.convert('L')
bw = np.asarray(gry).copy()

#basically kinda what it does -->
# bw[bw < 128] = 1 
# bw[bw >= 128] = 0
pic = bw.ravel()

nn.draw(pic)
ans = nn.predict(((pic)))
print(ans);

