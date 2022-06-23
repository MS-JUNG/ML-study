from sklearn.cluster import KMeans
import numpy as np 
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')
apple = fruits [0:100].reshape(-1,100*100)
pineapple = fruits[100:200].reshape(-1,100*100)
banana = fruits[200:300].reshape(-1,100*100)


fruits_2d = fruits.reshape(-1,100*100)

km = KMeans(n_clusters =3, random_state = 42)


inertia = []
for k in range(2,7):
    km = KMeans(n_clusters =k, random_state = 42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2,7),inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()