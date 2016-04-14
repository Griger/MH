from BLCUDA import *
import numpy as np

def flip(s,i):
	new_s = np.array(s)
	new_s[i] = not new_s[i]
	return new_s

def BMBCUDA(train_data, train_labels):
	print("Ejecutando BMB")
	n = len(train_data[0])
	best_s = None
	best_score = 0.0

	for _ in range(0,25):
		#print "Iteracion ",_
		random_sol = np.random.choice([True, False], n)
		s, s_score = BLCUDA(train_data, train_labels, random_sol)

		if best_score < s_score:
			best_s, best_score = s, s_score

	return best_s, best_score
