import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

## chapter 1
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



plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)

plt.xlabel('length')
plt.ylabel('weight')
plt.show()


length = bream_length + smelt_length 
weight = bream_weight + smelt_weight


X_train = [[l,w] for l,w in zip(length,weight)]

print(X_train)
y_train = [1] *35 + [0] * 14
print(y_train)



breamtest_length = []
breamtest_weight = []



for i in range(10,59):
    breamtest_length.append(data['Length1'][i])
for i in range(10,59):
    breamtest_weight.append(data['Weight'][i])
# for i in range(145,159):
#     smelttest_length.append(data['Length1'][i])   
# for i in range(145,159):
#     smelttest_weight.append(data['Weight'][i])

lengthtest = breamtest_length 
weighttest = breamtest_weight 
X_test = [[l,w] for l,w in zip(lengthtest,weighttest)]
y_test = [1] *24 + [0] *25

# kn.fit(fish_data, fish_target)
# score = kn.score(fish_data, fish_target)
error_rate  = []
for i in range(1,50):
   kn = KNeighborsClassifier(n_neighbors= i)

   kn.fit(X_train, y_train)
   predict_i = kn.predict(X_test)
   error_rate.append(np.mean(predict_i != y_test))

print(error_rate)

   

plt.figure(figsize=(30,12))
plt.plot(range(49),error_rate,marker="o",markerfacecolor="green",
         linestyle="dashed",color="red",markersize=15)
plt.title("Error rate vs k value",fontsize=20)
plt.xlabel("k- values",fontsize=20)
plt.ylabel("error rate",fontsize=20)
plt.xticks(range(1,49))
plt.show()
# print(score)
# breamtest_length = []
# breamtest_weight = []

# fish_target_score = [1] *24 + [0] *25

# for i in range(10,59):
#     breamtest_length.append(data['Length1'][i])
# for i in range(10,59):
#     breamtest_weight.append(data['Weight'][i])
# # for i in range(145,159):
# #     smelttest_length.append(data['Length1'][i])   
# # for i in range(145,159):
# #     smelttest_weight.append(data['Weight'][i])

# lengthtest = breamtest_length 
# weighttest = breamtest_weight 
# fish_testdata = [[l,w] for l,w in zip(lengthtest,weighttest)]

# score = kn.score(fish_testdata, fish_target_score)
# # print(score)
print(kn.predict([[20, 200]]))


## knn 알고리즘에서 최적의 k  값 찾기 일반적으론 총 자료 개수의 제곱근
## chapter 2 

