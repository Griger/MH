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
	max_evaluations = 15000

	#Calculation of algorithm's parameter
	n = len(data[0])

	s = np.random.choice([True, False], n) #initial solution
	best_s = np.array(s)
	s_score = 100*knn.getKNNClasiffierTrainingScore(data[:, s], labels)
	best_score = s_score

	max_neighbours = 10*n
	max_successes = 0.1*max_neighbours
	M = max_evaluations/max_neighbours
	T0 = -mu*s_score/np.log(fi) #initial temperature

	if (T0 <= Tf):
		Tf = T0/10.0

	print "n: ", n
	print "Tf: ", Tf

	T = T0 #current temperature
	beta = (T0-Tf)/(M*T0*Tf)
	n_evaluations = 0 #number of generated solutions

	no_success = False

	while (not no_success and (n_evaluations < max_evaluations)):
		n_successes = 0
		print "Temperatura enfriamiento actual: ", T

		for i in range(0, max_neighbours):
			#choose a random neighbour
			idx = np.random.random_integers(0,n-1)
			neighbour = flip(s, idx)
			neighbour_score = 100*knn.getKNNClasiffierTrainingScore(data[:, neighbour], labels)

			n_evaluations = n_evaluations + 1

			delta = s_score - neighbour_score

			if ((delta < 0) or np.random.uniform() <= np.exp(-delta/T)) and delta != 0:
				s = neighbour
				s_score = neighbour_score
				n_successes = n_successes + 1

			if (neighbour_score > best_score):
				best_s = neighbour
				best_score = neighbour_score

			if n_successes == max_successes or n_evaluations == max_evaluations:
				break

		T = nextT(T, beta)
		print "exitos en iteracion: ", n_successes
		print "evaluaciones totales: ", n_evaluations
		no_success = (n_successes == 0)

	return best_s, best_score
