
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

data = load_boston()
datas = data.data
target = data.target
train_input, test_input, train_target, test_target = train_test_split(datas, target,test_size =0.2, random_state= 42)
breakpoint()
model =LinearRegression()
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

model.fit(train_input,train_target)
print(model.score(train_input, train_target))
print(model.score(test_input, test_target))
model_2 = Ridge()
model_2.fit(train_input,train_target)
model_3 = Lasso()
model_3.fit(train_input,train_target)
model_3.score(test_input,test_target)
print(model_2.score(test_input,test_target))
print(model_3.score(test_input, test_target))