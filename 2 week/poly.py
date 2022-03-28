import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge


data = pd.read_csv('Fish.csv')
df = pd.DataFrame(data)
da = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = da.to_numpy()
perch_length = np.array(list(data[df.Species == 'Perch']['Length2']))
perch_weight = np.array(list(data[df.Species == 'Perch']['Weight']))

perch_length = perch_length.reshape(perch_length.shape[0],1)
perchh_weight = perch_weight.reshape(-1,1)

train_input, test_input, train_target, test_target = train_test_split(perch_full,perch_weight, random_state= 11)


# perch_length = perch_length.reshape(perch_length.shape[0],1)
# perch_weight = perch_weight.reshape(-1,1)

poly = PolynomialFeatures(degree = 5,include_bias = False)
poly.fit(train_input)

train_poly = poly.transform(train_input)

test_poly = poly.transform(test_input)
print(train_poly.shape)
print(test_poly.shape)

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(test_poly, test_target))
# poly.fit([[2,4,3]])
# print(poly.transform([[2,4,3]]))

ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled  =ss.transform(test_poly)


train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1,10,100]
for alpha in alpha_list:
    ridge = Ridge()
    ridge.fit(train_scaled, train_target)    
    train_score.append(ridge.score(train_scaled ))
    test_score.append(ridge.score(test_scaled, test_target))
