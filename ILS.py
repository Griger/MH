import numpy as np
import knn
from BL import *

def mutation(s):
	n = len(s)
	mutant_s = np.array(s)
	t = int(ceil(0.1*n))
	mutant_idx = np.random.choice(range(0,n), t, False)
	mutant_s[mutant_idx] = not mutant_s[mutant_idx]
	return mutant_s

def ILS (train_data, train_labels):
	s = np.random.choice([True, False], n)
	s, s_score = BL(train_data, train_labels, s)
	best_s, best_score = None, 0.0

	for _ in range(0,24):
		if best_score < s_score:
			best_s, best_score = s, s_score

		s = mutation(best_s)
		s, s_score = BL(s)

	return best_s, best_score
