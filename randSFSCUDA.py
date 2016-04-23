import numpy as np
from knnLooGPU import *

#Function that computes the training classification score of a solution
def getFeatureClassificationScore(current_sol, train_data, train_labels, idx, knnGPU):
	new_sol = np.array(current_sol)
	new_sol[idx] = True
	return knnGPU.scoreSolution(train_data[:, new_sol], train_labels)	

#vectorized version of previous function
vgetFeatureClassificationScore = np.vectorize(getFeatureClassificationScore, excluded = ['current_sol','train_data','train_labels', 'knnGPU'], otypes=[np.ndarray])

#Function that implements the random SFS algorithm
def randSFSCUDA(data, labels, knnGPU):
	n_features = len(data[0])
	finish = False
	sol = np.repeat(False, n_features)
	sol_score = 0.0
	CL = range(0, n_features) #candidate list
	alpha = 0.3

	#while we get profit and we can add new features
	while (not finish and len(CL) != 0):
		#compute individual feature scores in relation to the current set of selected features
		scores = vgetFeatureClassificationScore(current_sol = sol, train_data = data, train_labels = labels, idx = CL, knnGPU = knnGPU)

		max_score = np.max(scores)
		min_score = np.min(scores)
		mu = max_score - alpha*(max_score - min_score)

		RCL_idx = np.where(scores >= mu)[0] #restricted candidate list
		random_candidate_idx = np.random.choice(RCL_idx)
		random_candidate = CL[random_candidate_idx]
		random_candidate_score = scores[random_candidate_idx]

		if random_candidate_score >= sol_score:
			sol[random_candidate] = True
			sol_score = random_candidate_score
			del CL[random_candidate_idx]
		else:
			finish = True

	return sol, sol_score
