from scipy.io import arff
import numpy as np
from sklearn.preprocessing import MinMaxScaler

m = MinMaxScaler()

#Movement Libras load and formating
data, metadata = arff.loadarff('movement_libras.arff')

l = []
for d in data:
	l.append(list(d))

D = np.array([l[0]], dtype = np.float64)

for i in range(1,len(l)):
	D = np.append(D, np.array([l[i]], dtype = np.float64), axis = 0)

D[:, 0:90] = m.fit_transform(D[:, 0:90])
np.save("data_libras.npy", D)

#Wdbc load and formating
samples, metadata = arff.loadarff('wdbc.arff')

l = []
for d in samples:
	l.append(list(d))

D = np.array([l[0][1:31]], dtype = np.float64)

for i in range(1,len(l)):
	D = np.append(D, np.array([l[i][1:31]], dtype = np.float64), axis = 0)

D = m.fit_transform(D)

dictionary = dict(zip(['B','M'], [0,1]))
new_labels = [dictionary[label] for label in samples['class']]

D = np.insert(D, 30, new_labels, axis = 1)

np.save("data_wdbc.npy", D)

#Arrhythmia load and formating
samples, metadata = arff.loadarff('arrhythmia.arff')

l = []
for d in samples:
	l.append(list(d))

D = np.array([l[0]], dtype = np.float64)

for i in range(1,len(l)):
	D = np.append(D, np.array([l[i]], dtype = np.float64), axis = 0)

D[:, 0:278] = m.fit_transform(D[:, 0:278])
np.save("data_arrhythmia.npy", D)
