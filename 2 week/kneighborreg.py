import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

data = pd.read_csv('Fish.csv')
df = pd.DataFrame(data)


perch_length = np.array(list(data[df.Species == 'Perch']['Length2']))
perch_weight = np.array(list(data[df.Species == 'Perch']['Weight']))


perch_length = perch_length.reshape(perch_length.shape[0],1)
perch_weight = perch_weight.reshape(-1,1)




train_input, test_input, train_target, test_target = train_test_split(perch_length,perch_weight, random_state= 11)

knr = KNeighborsRegressor(n_neighbors= 5)
knr = knr.fit(train_input,train_target )
print(knr.score(test_input, test_target))
test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)

distances, indexes = knr.kneighbors([[50]])
print(distances)
print(knr.predict([[50]]))
plt.scatter(train_input, train_target)
plt.xlabel('length')
plt.ylabel('weight')

plt.scatter(train_input[indexes], train_target[indexes], marker = 'D')
plt.scatter(50, 1033, marker = 'h')
plt.show()