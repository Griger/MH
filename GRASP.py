from randSFS import *
from BL import *

def GRASP (train_data, train_labels):
	best_s = None
	best_score = 0.0

	for _ in range(0,25):
		s_greedy = randSFS(train_data, train_labels)
		s, s_score = BL(train_data, train_labels, s_greedy)

		if s_score > best_score:
			best_s, best_score = s, s_score

	return best_s, best_score
