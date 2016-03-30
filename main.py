import numpy as np
import BL
import time
import knn

test = np.load("partitions/wdbc0test.npy")
training = np.load("partitions/wdbc0training.npy")

training_data = training[:,0:30]
training_labels = training[:,30]
test_data = test[:,0:30]
test_labels = test[:,30]

start = time.time()
sol = BL.BL(training_data, training_labels)
end = time.time()

print "Final solution: ", sol
print "Numero de caracteristicas: ", len(sol)
print "Number of selected features: ", len(sol[sol == True])
#print "Final solution's training score: ", sol_score
print "Final solution's test score: ", 100*knn.getKNNClasiffierScore(training_data[:, sol], training_labels, test_data[:, sol], test_labels)
print "BL' execution time in seconds: ", end-start
