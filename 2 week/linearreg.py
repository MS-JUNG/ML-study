import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv('Fish.csv')
df = pd.DataFrame(data)


perch_length = np.array(list(data[df.Species == 'Perch']['Length2']))
perch_weight = np.array(list(data[df.Species == 'Perch']['Weight']))


perch_length = perch_length.reshape(perch_length.shape[0],1)
perch_weight = perch_weight.reshape(-1,1)



lr = LinearRegression()

train_input, test_input, train_target, test_target = train_test_split(perch_length,perch_weight, random_state= 11)

# lr.fit(train_input, train_target)

# print(lr.predict([[50]]))
#polynomial linear regression

train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))


lr = LinearRegression()


lr.fit(train_poly, train_target)
print(lr.coef_, lr.intercept_)

point = np.arange(15,50)
plt.scatter(train_input, train_target )
plt.plot(point, 1.03*(point**2) -23.59*point + 148.9 )
plt.show()
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))