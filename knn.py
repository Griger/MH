import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation

classifier = KNeighborsClassifier(n_neighbors = 3)

def getKNNClasiffierScore (training_data, training_labels, test_data, test_labels):
	classifier.fit(training_data, training_labels)
	return classifier.score(test_data, test_labels)

'''
def getKNNClasiffierTrainingScoreOneExample (example_idx, training_data, training_labels):
	distractors_data = np.delete(training_data, [example_idx], axis = 0)
	distractors_labels = np.delete(training_labels, [example_idx])
	#print example_idx
	#print "Distractors_data: ", distractors_data
	#print "Distractors_labels: ", distractors_labels
	classifier.fit(distractors_data, distractors_labels)
	return classifier.score([training_data[example_idx]], [training_labels[example_idx]])
	#return classifier.score(np.array([training_data[example_idx]]), np.array([training_labels[example_idx]]))

vgetKNNClasiffierTrainingScoreOneExample = np.vectorize(getKNNClasiffierTrainingScoreOneExample, excluded = ['training_data', 'training_labels'], otypes = [np.ndarray])


def getKNNClasiffierTrainingScore (training_data, training_labels):
	idx = range(0, len(training_data))
	#print "Un resultado: ", vgetKNNClasiffierTrainingScoreOneExample(idx, training_data = training_data, training_labels = training_labels)
	return np.mean(vgetKNNClasiffierTrainingScoreOneExample(idx, training_data = training_data, training_labels = training_labels))
'''

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
		score = score + classifier.score(X_test, y_test)/float(n)

	return score
