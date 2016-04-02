import numpy as np
import time
import random
import knn
from math import *

def flip(s,i):
	new_s = np.array(s)
	new_s[i] = not new_s[i]
	return new_s

def BText(data, labels):
	n = len(data[0]) #problem's size
	max_evaluations = 5000
	n_evaluations = 0
	tl_tam = n/3 #tabu list's tam
	tabu_idx = 0 #start of tabu list
	TL = np.repeat(-1, tl_tam) #tabu list
	neighbourhood_tam = 30

	s = np.random.choice([True, False], n) #initial solution
	best_s = s
	s_score = 100*knn.getKNNClasiffierTrainingScore(data[:, s], labels)
	best_score = s_score

	#new version adds
	it_without_new_best_sol = 0
	long_term_mem = np.repeat(0, n) #long term memory
	num_accepted_sols = 0
	while n_evaluations < max_evaluations:
		idx = random.sample(range(0,n), neighbourhood_tam)
		best_neighbour_score = 0.0
		idx_best_neighbour = -1

		for i in idx:
			neighbour = flip(s, i)
			neighbour_score = 100*knn.getKNNClasiffierTrainingScore(data[:, neighbour], labels)
			n_evaluations = n_evaluations + 1

			if (i in TL):
				if (neighbour_score > best_score and neighbour_score > best_neighbour_score):#aspiration criteria
					idx_best_neighbour, best_neighbour, best_neighbour_score = i, neighbour, neighbour_score
			else:
				if (neighbour_score > best_neighbour_score):
					idx_best_neighbour, best_neighbour, best_neighbour_score = i, neighbour, neighbour_score

			if n_evaluations == max_evaluations:
				break

		s = best_neighbour
		num_accepted_sols += 1
		long_term_mem[s] += 1 #update long term memory

		if best_neighbour_score > best_score:
			best_s, best_score = s, best_neighbour_score
			it_without_new_best_sol = 0
		else:
			it_without_new_best_sol += 1


		if it_without_new_best_sol == 10:
			it_without_new_best_sol = 0

			restart_method = np.random.choice([1,2,3], p = [0.25, 0.25, 0.5])

			if restart_method == 1: #random sol
				s = np.random.choice([True, False], n)
			elif restart_method == 2: #best sol
				s = best_s
			else: #random sol with long term memory
				u = np.random.uniform()
				s = u < (1 - long_term_mem/num_accepted_sols)

			#update tabu list tam
			reduce_tam = np.random.choice([True, False])

			if reduce_tam:
				print "Reducir tam:"
				print "TL antes: ", TL, "Tamanio antes: ", tl_tam
				new_tam = ceil(tl_tam/2.0)
				if (tabu_idx < new_tam):
					TL = TL[:int(new_tam)]
				else:
					TL = TL[int(new_tam):]
					tabu_idx -= int(new_tam)
				tl_tam = len(TL)
				print "TL despues: ", TL, "Tamanio despues: ", tl_tam
			else:
				print "Aumentar tam"
				print "TL antes: ", TL, "Tamanio antes: ", tl_tam
				TL = np.concatenate( (TL[tabu_idx:], TL[:tabu_idx], np.repeat(-1, ceil(tl_tam/2.0))) )
				tl_tam = len(TL)
				tabu_idx = 0
				print "TL despues: ", TL, "Tamanio despues: ", tl_tam

		TL[tabu_idx] = idx_best_neighbour
		tabu_idx = (tabu_idx + 1)%tl_tam #cyclic list
		print TL

	return best_s, best_score
