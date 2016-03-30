import numpy as np
import time
import knn

def flip(s,i):
	new_s = np.array(s)
	new_s[i] = not new_s[i]
	return new_s


def BL(training_data, training_labels):
	n = len(training_data[0]) #number of features
	s = np.random.choice([True, False], n) #initial solution
	s_score = 100*knn.getKNNClasiffierTrainingScore(training_data[:, s], training_labels)
	n_generated_sols = 0

	idx = random.sample(range(0,n), n)

	for i in idx:
		if n_generated_sols == 15000:
			return s

		s_i = flip(current_sol, i)
		s_i_score = 100*knn.getKNNClasiffierTrainingScore(training_data[:, s_i], training_labels)
		n_generated_sols = n_generated_sols + 1

		if(s_i_score > s_score):
			s[i] = not s[i]
			s_score = s_i_score
			break

		return s
