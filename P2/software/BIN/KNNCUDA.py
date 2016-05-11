import numpy as np

def KNNCUDA(train_data, train_labels, knnGPU):
	print("Ejecutando KNNCUDA")
	n = len(train_data[0]) #number of features
	s = np.repeat(True, n)
	s_score = knnGPU.scoreSolution(train_data[:, s], train_labels)

	return s, s_score
