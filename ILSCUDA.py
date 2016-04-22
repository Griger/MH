import numpy as np
from knnLooGPU import *
from BLCUDA import *
from math import *

def mutation(s):
	n = len(s)
	mutant_s = np.array(s)
	t = int(ceil(0.1*n))
	mutant_idx = np.random.choice(range(0,n), t, False)
	mutant_s[mutant_idx] = np.logical_not(mutant_s[mutant_idx])
	return mutant_s

def ILSCUDA(train_data, train_labels, knnGPU):
	n = len(train_data[0])
	s = np.random.choice([True, False], n)
	s, s_score = BLCUDA(train_data, train_labels, knnGPU, s)
	best_s, best_score = None, 0.0

	for _ in range(0,24):
		if best_score < s_score:
			best_s, best_score = s, s_score

		s = mutation(best_s)
		s, s_score = BLCUDA(train_data, train_labels, knnGPU, s)

	return best_s, best_score
