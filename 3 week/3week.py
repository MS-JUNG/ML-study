import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

fish = pd.read_csv('https://bit.ly/fish_csv_data')
np.set_printoptions(precision=6, suppress=True)
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(fish_input,fish_target, random_state= 11)
pre = StandardScaler()
pre.fit(train_input)
train_scaled = pre.transform(train_input)
test_scaled = pre.transform(test_input)
kn = KNeighborsClassifier(n_neighbors = 3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
print(kn.classes_)

print(kn.predict_proba(test_scaled))
print(test_target[1])
l = np.array([-1.062186, -1.368436, -1.430107, -1.491334, -1.516303])
m = np.array([-0.96697 , -1.331267, -1.318376, -0.538183, -1.35469 ])
l = l.reshape(1,-1)
m = m.reshape(1,-1)
distances, indexes = kn.kneighbors(test_scaled[0:40])
# print(kn.predict(l))
print(distances)
print(kn.predict(l))
print(kn.predict(m))
print(train_target[indexes])
l = 0
for i in range(40):
    print(kn.predict(test_scaled[i].reshape(1,-1)))
    l += 1 

print(l)
breakpoint()