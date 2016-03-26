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
print n_zeros
print n_ones
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


'''
n_data = len(labels)

one_set = data_libras[labels == 1.0]
two_set = data_libras[labels == 2.0]
three_set = data_libras[labels == 3.0]
four_set = data_libras[labels == 4.0]
five_set = data_libras[labels == 5.0]
six_set = data_libras[labels == 6.0]
seven_set = data_libras[labels == 7.0]
eight_set = data_libras[labels == 8.0]
nine_set = data_libras[labels == 9.0]
ten_set = data_libras[labels == 10.0]
eleven_set = data_libras[labels == 11.0]
twelve_set = data_libras[labels == 12.0]
thirteen_set = data_libras[labels == 13.0]
fourteen_set = data_libras[labels == 14.0]
fifteen_set = data_libras[labels == 15.0]

n1 = len(one_set)
n2 = len(two_set)
n3 = len(three_set)
n4 = len(four_set)
n5 = len(five_set)
n6 = len(six_set)
n7 = len(seven_set)
n8 = len(eight_set)
n9 = len(nine_set)
n10 = len(ten_set)
n11 = len(eleven_set)
'''
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
