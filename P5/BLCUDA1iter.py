import numpy as np
from knnLooGPU import *

def flip(s,i):
	new_s = np.array(s)
	new_s[i] = not new_s[i]
	return new_s


def BLCUDA1iter(training_data, training_labels, knnGPU, initial_sol):
	n = len(training_data[0]) #number of features
	s = np.array(initial_sol)
	s_score = knnGPU.scoreSolution(training_data[:, s], training_labels)
	n_generated_sols = 0

	idx = np.random.choice(range(0,n), n, False)
	found_better_sol = False

	for i in idx:
		s[i] = not s[i]
		s_i_score = knnGPU.scoreSolution(training_data[:, s], training_labels)
		n_generated_sols += 1

		if(s_i_score > s_score):
			found_better_sol = True
			s_score = s_i_score
		else:
			s[i] = not s[i] #revert change

		if found_better_sol:
			break

	return s, s_score, n_generated_sols
