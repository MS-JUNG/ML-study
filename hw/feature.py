from sklearn.model_selection import cross_validate
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

plt.title('Linearregression')

plt.show()
new_data = np.delete(datas,(0,2,3,4,5,6,9,11),axis =1)
train_input, test_input, train_target, test_target = train_test_split(new_data, target,test_size =0.2, random_state = 42)
model.fit(train_input,train_target)
print(model.score(train_input,train_target))
print(model.score(test_input, test_target))


# data preprocessing
data_d = data_df[['INDUS','RM','ZN','CHAS','AGE','RAD','DIS']]
dict = data_d.to_dict('records')
data_d = data_d.values.tolist()

score_trainsample = []
score_testsample = []
cross_vali = []


for i in range(50):
    train_input, test_input, train_target, test_target = train_test_split(data_d, target,test_size =0.2, random_state = i,)
    model.fit(train_input,train_target)
    train_sample = model.score(train_input, train_target)
    test_sample = model.score(test_input, test_target)
    score_trainsample.append(train_sample)
    score_testsample.append(test_sample)

    scores = cross_validate(model, train_input, train_target, return_train_score = True)

    test_score = np.mean(scores['test_score'])

    cross_vali.append(test_score)



    
x = range(50)
y = score_trainsample
x_2 = range(50)
y_2 = score_testsample
x_3 = range(50)
y_3 = cross_vali
plt.plot(x,y,'green')
plt.plot(x_2, y_2)
plt.plot(x_3, y_3,'red')


plt.show()

print(scores)