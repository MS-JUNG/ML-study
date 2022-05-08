from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

data = load_boston()
data_df = pd.DataFrame(data.data, columns = data.feature_names)


datas = data.data
target = data.target

train_input, test_input, train_target, test_target = train_test_split(datas, target,test_size =0.2, random_state= 42)

model = ElasticNet()
for i in range(13):
    
    globals()[data.feature_names[i]] = np.delete(datas,i,axis=1)
    print(data.feature_names[i])

train_score = []
score =[]
for i in range(13):
    train_input, test_input, train_target, test_target = train_test_split(globals()[data.feature_names[i]], target,test_size =0.2, random_state = 42)
    model.fit(train_input,train_target)

    score.append(model.score(test_input, test_target))
    train_score.append(model.score(train_input,train_target))


x = data.feature_names
y = score
y_2 = train_score
plt.figure(figsize=(20,10))
plt.ylim(0,0.7)
plt.subplot(1,2,1)

plt.plot(x,y)
plt.plot(x,y_2)
plt.title('Elasticnet')
model = LinearRegression()
train_score = []
score =[]
for i in range(13):
    train_input, test_input, train_target, test_target = train_test_split(globals()[data.feature_names[i]], target,test_size =0.2, random_state = 42)
    model.fit(train_input,train_target)

    score.append(model.score(test_input, test_target))
    train_score.append(model.score(train_input,train_target))

# 1,7,8,10,12
x = data.feature_names
y = score
y_2 = train_score

plt.subplot(1,2,2)

plt.plot(x,y)
plt.plot(x,y_2)
plt.title('Logisitic')

plt.show()
new_data = np.delete(datas,(0,2,3,4,5,6,9,11),axis =1)
train_input, test_input, train_target, test_target = train_test_split(new_data, target,test_size =0.2, random_state = 42)
model.fit(train_input,train_target)
print(model.score(train_input,train_target))
print(model.score(test_input, test_target))
