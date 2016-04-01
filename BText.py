import numpy as np
import time
import random
import knn

def flip(s,i):
	new_s = np.array(s)
	new_s[i] = not new_s[i]
	return new_s

def BT(data, labels):
	n = len(data[0]) #problem's size'
	max_evaluations = 15000
	n_evaluations = 0
	tl_tam = n/3 #tabu list's tam
	tabu_idx = 0 #start of tabu list
	TL = np.repeat(-1, tl_tam) #tabu list
	neighbourhood_tam = 30

	s = np.random.choice([True, False], n) #initial solution
	best_s = s
	s_score = 100*knn.getKNNClasiffierTrainingScore(data[:, s], labels)
	best_score = s_score

	it_without_improving = 0

	while n_evaluations < max_evaluations:
		idx = random.sample(range(0,n), neighbourhood_tam)
		score_best_neighbour = 0.0
		idx_best_neighbour = -1
		#print "New idx: ", idx

		it_without_improving += 1

		for i in idx:
			neighbour = flip(s, i)
			neighbour_score = 100*knn.getKNNClasiffierTrainingScore(data[:, neighbour], labels)
			n_evaluations = n_evaluations + 1

			if (i in TL):
				if (neighbour_score > best_score):#aspiration criteria
					print "Aspiration criteria: "
					idx_best_neighbour = i
					best_neighbour = neighbour
					score_best_neighbour = neighbour_score

					best_s = neighbour
					best_score = neighbour_score

					it_without_improving = 0
			else:
				if (neighbour_score > score_best_neighbour):
					idx_best_neighbour = i
					best_neighbour = neighbour
					score_best_neighbour = neighbour_score

				if (neighbour_score > best_score):
					best_s = neighbour
					best_score = neighbour_score

					it_without_improving = 0

			if n_evaluations == max_evaluations:
				break

		s = best_neighbour

		#TODO esto es para que acabe antes
		if (s_score == score_best_neighbour):
			break

		s_score = score_best_neighbour

		if (it_without_improving == 10):
			it_without_improving = 0
			s = reinicio #TODO
		
		if not idx_best_neighbour in TL: #TODO estamos evitando meter dos veces el mismo movimiento en la lista
			TL[tabu_idx] = idx_best_neighbour
			tabu_idx = (tabu_idx + 1)%tl_tam #cyclic list

		print "Total evaluations: ", n_evaluations
		print "Best neighbout idx: ", idx_best_neighbour
		print "TL: ", TL
	return best_s, best_score
