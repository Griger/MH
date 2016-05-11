#Partition creation's file

import numpy as np

np.random.seed(12345678)

#wdbc partitions
data_wdbc = np.load("data_wdbc.npy")
data = data_wdbc[:,0:30]
labels = data_wdbc[:,30]

n_data = len(labels)

zeros = data_wdbc[labels == 0.0]
ones = data_wdbc[labels == 1.0]

n_zeros = len(zeros)
n_ones = len(ones)

for i in range(0,5):
	perm_zeros = np.random.permutation(n_zeros)
	perm_ones  = np.random.permutation(n_ones)

	training_part = np.concatenate((zeros[perm_zeros[n_zeros/2:]], ones[perm_ones[n_ones/2:]]))
	test_part = np.concatenate((zeros[perm_zeros[:n_zeros/2]], ones[perm_ones[:n_ones/2]]))

	np.save("wdbc"+str(i)+"training.npy", training_part)
	np.save("wdbc"+str(i)+"test.npy", test_part)


#movement_libras partitions
data_libras = np.load("data_libras.npy")
data = data_libras[:, 0:90]
labels = data_libras[:,90]

categories = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0]

for i in range(0,5):
	elements = data_libras[labels == 1.0]
	n_elem = len(elements)
	perm = np.random.permutation(n_elem)

	training_part = elements[perm[n_elem/2:]]
	test_part = elements[perm[:n_elem/2]]

	for c in categories[1:]:
		elements = data_libras[labels == c]
		n_elem = len(elements)
		perm = np.random.permutation(n_elem)

		training_part = np.concatenate((training_part, elements[perm[n_elem/2:]]))
		test_part = np.concatenate((test_part, elements[perm[:n_elem/2]]))

	np.save("libras"+str(i)+"training.npy", training_part)
	np.save("libras"+str(i)+"test.npy", test_part)


#arrhythmia partitions
data_arr = np.load("data_arrhythmia.npy")
data = data_arr[:, 0:278]
labels = data_arr[:,278]

categories = [1.0,2.0,6.0,10.0,16.0]

for i in range(0,5):
	elements = data_arr[labels == 1.0]
	n_elem = len(elements)
	perm = np.random.permutation(n_elem)

	training_part = elements[perm[n_elem/2:]]
	test_part = elements[perm[:n_elem/2]]

	for c in categories[1:]:
		elements = data_arr[labels == c]
		n_elem = len(elements)
		perm = np.random.permutation(n_elem)

		training_part = np.concatenate((training_part, elements[perm[n_elem/2:]]))
		test_part = np.concatenate((test_part, elements[perm[:n_elem/2]]))

	np.save("arr"+str(i)+"training.npy", training_part)
	np.save("arr"+str(i)+"test.npy", test_part)
