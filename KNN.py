import numpy as np
import knn

def KNN(training_data, training_labels):
	print "Ejecutando KNN"
	n = len(training_data[0]) #number of features
	s = np.repeat(True, n) 
	s_score = knn.getKNNClasiffierTrainingScore(training_data[:, s], training_labels)

	return s, s_score
