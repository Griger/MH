import numpy as np
import BMBCUDA
import time
import knn
import sys

def splitFeaturesAndLabels(data):
	n_features = len(data[0]) - 1
	#return data[:, 0:n_features], data[:, n_features]
	return data[:, 0:n_features], np.array(data[:, n_features], dtype = np.int32)

def getResult(heuristic_name, heuristic, train_data, train_labels, test_data, test_labels):
	np.random.seed(12345678)
	start = time.time()
	sol, sol_training_score = heuristic(train_data, train_labels)
	end = time.time()

	print("% reduction: ", 1.0*(len(sol) - len(sol[sol == True]))/len(sol)*100, "%")
	print("Final solution's training score: ", sol_training_score)
	print("Final solution's test score: ", knn.getKNNClasiffierScore(train_data[:, sol], train_labels, test_data[:, sol], test_labels))
	print(heuristic_name + "' execution time in seconds: ", end-start)


heuristics = {'BMB': BMBCUDA.BMBCUDA}
heuristic_names = ['BMB']

if len(sys.argv) < 2 or not sys.argv[1] in heuristic_names:
	print("Algoritmo no especificado o no valido. Elija entre BMB")
else:
	heuristic_name = sys.argv[1]
	heuristic = heuristics[heuristic_name]
	'''
	wdbc_test_data = []
	wdbc_test_labels = []
	wdbc_train_data = []
	wdbc_train_labels = []


	for i in range(0,5):
		test = np.load("partitions32b/wdbc" + str(i) + "test.npy")
		training = np.load("partitions32b/wdbc" + str(i) + "training.npy")
		D, L = splitFeaturesAndLabels(test)
		wdbc_test_data.append(D)
		wdbc_test_labels.append(L)
		D, L = splitFeaturesAndLabels(training)
		wdbc_train_data.append(D)
		wdbc_train_labels.append(L)

	for i in range(0,5):
		print("Results for", heuristic_name, "in wdbc")
		print("Partition ", i+1, "-", 1)
		getResult(heuristic_name, heuristic, wdbc_train_data[i], wdbc_train_labels[i], wdbc_test_data[i], wdbc_test_labels[i])
		print("Partition ", i+1, "-", 2)
		getResult(heuristic_name, heuristic, wdbc_test_data[i], wdbc_test_labels[i], wdbc_train_data[i], wdbc_train_labels[i])


	libras_test_data = []
	libras_test_labels = []
	libras_train_data = []
	libras_train_labels = []

	for i in range(0,5):
		test = np.load("partitions32b/libras" + str(i) + "test.npy")
		training = np.load("partitions32b/libras" + str(i) + "training.npy")
		D, L = splitFeaturesAndLabels(test)
		libras_test_data.append(D)
		libras_test_labels.append(L)
		D, L = splitFeaturesAndLabels(training)
		libras_train_data.append(D)
		libras_train_labels.append(L)

	for i in range(0,5):
		print("Results for", heuristic_name, "in libras")
		print("Partition ", i+1, "-", 1)
		getResult(heuristic_name, heuristic, libras_train_data[i], libras_train_labels[i], libras_test_data[i], libras_test_labels[i])
		print("Partition ", i+1, "-", 2)
		getResult(heuristic_name, heuristic, libras_test_data[i], libras_test_labels[i], libras_train_data[i], libras_train_labels[i])

	'''
	arr_test_data = []
	arr_test_labels = []
	arr_train_data = []
	arr_train_labels = []

	for i in range(0,5):
		test = np.load("partitions32b/arr" + str(i) + "test.npy")
		training = np.load("partitions32b/arr" + str(i) + "training.npy")
		D, L = splitFeaturesAndLabels(test)
		arr_test_data.append(D)
		arr_test_labels.append(L)
		D, L = splitFeaturesAndLabels(training)
		arr_train_data.append(D)
		arr_train_labels.append(L)

	for i in range(0,5):
		print("Results for", heuristic_name, "in arr")
		print("Partition ", i+1, "-", 1)
		getResult(heuristic_name, heuristic, arr_train_data[i], arr_train_labels[i], arr_test_data[i], arr_test_labels[i])
		print("Partition ", i+1, "-", 2)
		getResult(heuristic_name, heuristic, arr_test_data[i], arr_test_labels[i], arr_train_data[i], arr_train_labels[i])
