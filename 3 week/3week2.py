import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot as plt
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(fish_input,fish_target, random_state= 11)
bream_smelt_indexes = (train_target == 'Bream') | (train_target =='Smelt')
pre = StandardScaler()
pre.fit(train_input)
train_scaled = pre.transform(train_input)
test_scaled = pre.transform(test_input)
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
ts = []
tr = []
for i in range(10,100):

    SG = SGDClassifier(loss ='log', max_iter = i, random_state = 42)

    SG.fit(train_scaled, train_target)
    train = SG.score(train_scaled, train_target)
    test =SG.score(test_scaled, test_target)
    ts.append(test)
    tr.append(train)


plt.xlabel("max_iter")
plt.ylabel("score")
plt.plot(range(10,100),ts,color = 'green')
plt.plot(range(10,100),tr)
plt.show()


SG = SGDClassifier(loss ='log', random_state = 42)
train_score = []
test_score = []

classes = np.unique(train_target)
for _ in range(0,1500):
    SG.partial_fit(train_scaled, train_target, classes = classes)
    train_score.append(SG.score(train_scaled, train_target))
    breakpoint()    
    test_score.append(SG.score(test_scaled, test_target))




plt.plot(train_score, color = 'green')
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('score')
plt.show()
    
