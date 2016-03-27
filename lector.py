from scipy.io import arff
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time

data_libras = None

def loadData():
	global data_libras
	data, metadata = arff.loadarff('movement_libras.arff')

	l = []
	e = []
	for d in data:
		l.append(list(d)[0:90])
		e.append(d[90])

	D = np.array(l)
	L = np.array(e, dtype = int)
	data_libras = np.array(zip(D,L))


start = time.time()
data, metadata = arff.loadarff('movement_libras.arff')

l = []
e = []
for d in data:
	l.append(list(d)[0:90])
	e.append(d[90])

D = np.array(l)
L = np.array(e, dtype = np.float64)
data_libras = np.array(zip(D,L))
end = time.time()
print "Cargando desde arff y formateando he tardado: ", end - start

start = time.time()
data_libras = np.load("data_libras.npy")
end = time.time()

print "cargando el array desde un archivo npy he tardado: ", end - start
#print data_libras

X = [[0], [1], [2], [3]]
y = [2, 2, 2, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))
