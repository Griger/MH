import numpy as np
import knn

#annealing's scheme
def nextT(T, beta):
	return T/(1+beta*T)

def flip(s,i):
	new_s = np.array(s)
	new_s[i] = not new_s[i]
	return new_s

def ES(data, labels):

	#Common parameter values
	mu = 0.3
	fi = 0.3
	Tf = 10**(-3)
	max_evaluations = 5000
	#Calculation of algorithm's parameter
	n = len(data[0])

	max_neighbours = 10*n
	max_successes = 0.1*max_neighbours
	M = max_evaluations/max_neighbours

	s = np.random.choice([True, False], n) #initial solution
	best_s = np.array(s)
	s_score = knn.getKNNClasiffierTrainingScore(data[:, s], labels)
	best_score = s_score

	T0 = -mu*s_score/np.log(fi) #initial temperature

	if (T0 <= Tf):
		Tf = T0/10.0

	T = T0 #current temperature
	beta = (T0-Tf)/(M*T0*Tf)
	n_evaluations = 0 #number of generated solutions

	no_success = False

	while (not no_success and (n_evaluations < max_evaluations)):
		n_successes = 0

		for i in range(0, max_neighbours):
			#choose a random neighbour
			idx = np.random.random_integers(0,n-1)
			neighbour = flip(s, idx)
			neighbour_score = knn.getKNNClasiffierTrainingScore(data[:, neighbour], labels)

			n_evaluations += 1

			delta = s_score - neighbour_score

			if delta != 0 and ((delta < 0) or np.random.uniform() <= np.exp(-delta/T)):
				s, s_score = neighbour, neighbour_score
				n_successes += 1

				if (s_score > best_score):
					best_s, best_score = s, s_score

			if n_successes == max_successes or n_evaluations == max_evaluations:
				break

		T = nextT(T, beta)
		no_success = (n_successes == 0)

	return best_s, best_score
