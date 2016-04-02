import numpy as np
import knn

#Function that computes the training classification score of a solution
def getFeatureClassificationScore(current_sol, train_data, train_labels, idx):
	new_sol = np.array(current_sol)
	new_sol[idx] = True
	return knn.getKNNClasiffierTrainingScore(train_data[:, new_sol], train_labels)

#vectorized version of previous function
vgetFeatureClassificationScore = np.vectorize(getFeatureClassificationScore, excluded = ['current_sol','train_data','train_labels'], otypes=[np.ndarray])

#Function that implements the SFS algorithm
def SFS(data, labels):
	n_features = len(data[0])
	finish = False
	sol = np.repeat(False, n_features)
	sol_score = 0
	c_idx = range(0, n_features)

	#while we get profit and we can add new features
	while (not finish and len(c_idx) != 0):
		#compute individual feature scores in relation to the current set of selected features
		scores = vgetFeatureClassificationScore(current_sol = sol, train_data = data, train_labels = labels, idx = c_idx)
		max_idx = np.argmax(scores)
		max_score = scores[max_idx]

		if max_score > sol_score:
			sol[c_idx[max_idx]] = True
			sol_score = max_score
			del c_idx[max_idx]
		else:
			finish = True

	return sol, sol_score
