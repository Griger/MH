from randSFSCUDA import *
from BLCUDA import *
from knnLooGPU import *

def GRASPCUDA (train_data, train_labels, knnGPU):
	best_s = None
	best_score = 0.0

	for _ in range(0,25):
		s_greedy, s_greedy_score = randSFSCUDA(train_data, train_labels, knnGPU)
		s, s_score = BLCUDA(train_data, train_labels, knnGPU, s_greedy)

		if s_score > best_score:
			best_s, best_score = s, s_score

	return best_s, best_score
