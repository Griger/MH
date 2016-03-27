import time
import numpy as np
import knn

np.random.seed(12345678)

'''
data_libras = np.load("data_libras.npy")
data = data_libras[:,0:90]
labels = data_libras[:,90]
n_data = len(labels)
n_features = len(data[0])

perm = np.random.permutation(n_data)
training_perm = perm[:n_data/2]
test_perm = perm[n_data/2:]

training_data = data[training_perm]
training_labels = labels[training_perm]
test_data = data[test_perm]
test_labels = labels[test_perm]
'''

'''
data_wdbc = np.load("data_wdbc.npy")
data = data_wdbc[:,0:30]
labels = data_wdbc[:,30]
n_data = len(labels)
n_features = len(data[0])

perm = np.random.permutation(n_data)
training_perm = perm[:n_data/2]
test_perm = perm[n_data/2:]

training_data = data[training_perm]
training_labels = labels[training_perm]
test_data = data[test_perm]
test_labels = labels[test_perm]
'''

'''
data_arr = np.load("data_arrhythmia.npy")
data = data_arr[:,0:278]
labels = data_arr[:,278]
n_data = len(labels)
n_features = len(data[0])

perm = np.random.permutation(n_data)
training_perm = perm[:n_data/2]
test_perm = perm[n_data/2:]

training_data = data[training_perm]
training_labels = labels[training_perm]
test_data = data[test_perm]
test_labels = labels[test_perm]
'''
n_features = 30
arr_test = np.load("partitions/wdbc1test.npy")
arr_training = np.load("partitions/wdbc1training.npy")

training_data = arr_training[:,0:30]
training_labels = arr_training[:,30]
test_data = arr_test[:,0:30]
test_labels = arr_test[:,30]

def getFeatureClassificationScore(current_sol, idx):
	new_sol = np.array(current_sol)
	new_sol[idx] = True
	return 100*knn.getKNNClasiffierTrainingScore(training_data[:, new_sol], training_labels)

vgetFeatureClassificationScore = np.vectorize(getFeatureClassificationScore, excluded = ['current_sol'], otypes=[np.ndarray])

#Function that implements the SFS algorithm

def SFS():
	start = time.time()
	finish = False
	sol = np.repeat(False, n_features)
	sol_score = 0
	c_idx = range(0, n_features)

	while (not finish and len(c_idx) != 0):
		scores = vgetFeatureClassificationScore(current_sol = sol, idx = c_idx)
		#print scores, len(scores)
		max_idx = np.argmax(scores)
		max_score = scores[max_idx]
		print "La nueva sol tiene una tasa de acierto: ", max_score
		#print max_score, max_idx
		if max_score > sol_score:
			sol[c_idx[max_idx]] = True
			sol_score = max_score
			del c_idx[max_idx]
		else:
			finish = True

	end = time.time()
	print "Final solution: ", sol
	print "Numero de caracteristicas: ", len(sol)
	print "Number of selected features: ", len(sol[sol == True])
	print "Final solution's training score: ", sol_score
	print "Final solution's test score: ", 100*knn.getKNNClasiffierScore(training_data[:, sol], training_labels, test_data[:, sol], test_labels)
	print "SFS' execution time in seconds: ", end-start

SFS()
