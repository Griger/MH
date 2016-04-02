import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation

classifier = KNeighborsClassifier(n_neighbors = 3)

def getKNNClasiffierScore (training_data, training_labels, test_data, test_labels):
	classifier.fit(training_data, training_labels)
	return 100*classifier.score(test_data, test_labels)

def getKNNClasiffierTrainingScore (training_data, training_labels):
	n = len(training_labels)
	score = 0.0
	loo = cross_validation.LeaveOneOut(n)

	if len(training_data[0]) == 0:
		return 0.0

	for train_index, test_index in loo:
		X_train, X_test = training_data[train_index], training_data[test_index]
		y_train, y_test = training_labels[train_index], training_labels[test_index]
		classifier.fit(X_train, y_train)
		score = score + classifier.score(X_test, y_test)

	return 100*score/float(n)
