import numpy as np
import knn

#Function that computes the training classification score of a solution
def getFeatureClassificationScore(current_sol, train_data, train_labels, idx):
	new_sol = np.array(current_sol)
	new_sol[idx] = True
	return knn.getKNNClasiffierTrainingScore(train_data[:, new_sol], train_labels)

#vectorized version of previous function
vgetFeatureClassificationScore = np.vectorize(getFeatureClassificationScore, excluded = ['current_sol','train_data','train_labels'], otypes=[np.ndarray])

#Function that implements the random SFS algorithm
def randSFS(data, labels):
	n_features = len(data[0])
	finish = False
	sol = np.repeat(False, n_features)
	sol_score = 0.0
	CL = range(0, n_features) #candidate list
	alpha = 0.3

	#while we get profit and we can add new features
	while (not finish and len(CL) != 0):
		#compute individual feature scores in relation to the current set of selected features
		scores = vgetFeatureClassificationScore(current_sol = sol, train_data = data, train_labels = labels, idx = CL)
		max_score = np.max(scores)
		min_score = np.min(scores)

		mu = max_score - alpha*(max_score - min_score)
		RCL = CL[scores >= mu] #restricted candidate list
		random_candidate = np.random.choice(range(0,len(RLC)))
		candidate_score 

		if candidate_score >= sol_score:
			sol[random_candidate] = True
			sol_score = candidate_score
			del CL[candidate_idx]
		else:
			finish = True

	return sol, sol_score
