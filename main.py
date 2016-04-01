import numpy as np
import BL
import ES
import BT
import time
import knn

np.random.seed(12345678)

def splitFeaturesAndLabels(data):
	n_features = len(data[0]) - 1
	return data[:, 0:n_features], data[:, n_features]

test = np.load("partitions/wdbc0test.npy")
training = np.load("partitions/wdbc0training.npy")

training_data = training[:,0:30]
training_labels = training[:,30]
test_data = test[:,0:30]
test_labels = test[:,30]

'''
D, L = splitFeaturesAndLabels(training)

print D[0]
print L - training_labels
'''

start = time.time()
sol, sol_training_score =BT.BT(training_data, training_labels)
end = time.time()

print "Final solution: ", sol
print "Numero de caracteristicas: ", len(sol)
print "Number of selected features: ", len(sol[sol == True])
print "Final solution's training score: ", sol_training_score
print "Final solution's test score: ", 100*knn.getKNNClasiffierScore(training_data[:, sol], training_labels, test_data[:, sol], test_labels)
print "ES' execution time in seconds: ", end-start
