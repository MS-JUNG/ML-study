import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv('Fish.csv')
print(data)
print(data.shape[0])
bream_length  = []
bream_weight = []
smelt_length = []
smelt_weight = []

bream_length

for i in range(35):
    bream_length.append(data['Length2'][i])
for i in range(35):
    bream_weight.append(data['Weight'][i])
for i in range(145,159):
    smelt_length.append(data['Length2'][i])   
for i in range(145,159):
    smelt_weight.append(data['Weight'][i])



length = bream_length + smelt_length 
weight = bream_weight + smelt_weight




X_train = np.array([[l,w] for l,w in zip(length,weight)])

# print(fish_data)

y_train = np.array([1] *35 + [0] * 14)

print(X_train)
np.random.seed(45)
index = np.arange(48)
np.random.shuffle(index)

train_input = X_train[index[:35]]
test_input = X_train[index[35:]]
train_target = y_train[index[:35]]
test_target = y_train[index[35:]]

kn = KNeighborsClassifier()

# 2-2 Data preprocessing 

# stack = np.column_stack(([1,2,3],[4,5,6]))

fish_data = np.column_stack((length, weight))
print(fish_data)


fish_target = np.concatenate((np.ones(35), np.zeros(14)))

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state= 11)

print(test_target)














