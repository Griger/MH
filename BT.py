import numpy as np
import random
import knn

def flip(s,i):
	new_s = np.array(s)
	new_s[i] = not new_s[i]
	return new_s

def BT(data, labels):

	n = len(data[0]) #problem's size
	max_evaluations = 5000
	n_evaluations = 0
	tl_tam = n/3 #tabu list's tam
	tabu_idx = 0 #start of tabu list
	TL = np.repeat(-1, tl_tam) #tabu list
	neighbourhood_tam = 30

	s = np.random.choice([True, False], n) #initial solution
	best_s = s
	s_score = knn.getKNNClasiffierTrainingScore(data[:, s], labels)
	best_score = s_score

	while n_evaluations < max_evaluations:
		idx = random.sample(range(0,n), neighbourhood_tam)
		best_neighbour_score = 0.0
		idx_best_neighbour = -1

		for i in idx:
			neighbour = flip(s, i)
			neighbour_score = knn.getKNNClasiffierTrainingScore(data[:, neighbour], labels)
			n_evaluations = n_evaluations + 1

			#print neighbour
			#print neighbour_score

			if (i in TL):
				if (neighbour_score > best_score and neighbour_score > best_neighbour_score):#aspiration criteria
					#print "Acepto por aspiracion"
					idx_best_neighbour, best_neighbour, best_neighbour_score = i, neighbour, neighbour_score
			else:
				if (neighbour_score > best_neighbour_score):
					#print "Acepto vecino por ser mejor que el anterior"
					idx_best_neighbour, best_neighbour, best_neighbour_score = i, neighbour, neighbour_score

			if n_evaluations == max_evaluations:
				break

		s = best_neighbour

		if best_neighbour_score > best_score:
			best_s, best_score = s, best_neighbour_score

		TL[tabu_idx] = idx_best_neighbour
		tabu_idx = (tabu_idx + 1)%tl_tam #cyclic list

		#print "Total evaluations: ", n_evaluations
		#print "Best neighbout idx: ", idx_best_neighbour
		#print "TL: ", TL

	return best_s, best_score
