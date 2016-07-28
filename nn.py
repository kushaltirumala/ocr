import numpy as np
from ocr import OCRNeuralNetwork
from sklearn.cross_validation import train_test_split

def test(data_matrix, data_labels, test_indices, nn):
    avg_sum = 0
    for j in xrange(100):
        correct_guess_count = 0
        for i in test_indices:
            test = data_matrix[i]
            prediction = nn.predict(test)
            if data_labels[i] == prediction:
                correct_guess_count += 1

        avg_sum += (correct_guess_count / float(len(test_indices)))
    return avg_sum / 100


data_matrix = np.loadtxt(open('data.csv', 'rb'), delimiter = ',').tolist()
data_labels = np.loadtxt(open('dataLabels.csv', 'rb')).tolist()

train_indices, test_indices = train_test_split(list(range(5000)))



for i in xrange(5, 100, 5):
    nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
    performance = str(test(data_matrix, data_labels, test_indices, nn))
    print "{i} Hidden Nodes: {val}".format(i=i, val=performance)
