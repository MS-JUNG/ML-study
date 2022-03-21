from cv2 import normalize
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, plot_confusion_matrix

# X_train = [[l,w] for l,w in zip(length,weight)]
## K value에 따른 오차 확인 및 개선 방안 및 분류 오차 줄이는 방안에 대해서 고민
data = pd.read_csv('Fish.csv')
df = pd.DataFrame(data)


## 6가지 물고기 종류별 weight, height 추출
Bream_weight = list(data[df.Species == 'Bream']['Weight'])
Bream_Height = list(data[df.Species == 'Bream']['Height'])
Roach_weight = list(data[df.Species == 'Roach']['Weight'])
Roach_Height= list(data[df.Species == 'Roach']['Height'])
Whitefish_weight = list(data[df.Species == 'Whitefish']['Weight'])
Whitefish_Height = list(data[df.Species == 'Whitefish']['Height'])
Parkki_weight = list(data[df.Species == 'Parkki']['Weight'])
Parkki_Height = list(data[df.Species == 'Parkki']['Height'])
Perch_weight = list(data[df.Species == 'Perch']['Weight'])
Perch_Height = list(data[df.Species == 'Perch']['Height'])
Pike_weight = list(data[df.Species == 'Pike']['Weight'])
Pike_Height = list(data[df.Species == 'Pike']['Height'])
Smelt_weight = list(data[df.Species == 'Smelt']['Weight'])
Smelt_Height = list(data[df.Species == 'Smelt']['Height'])

target_value = [0]*35 + [1] * 20 + [2] * 6 + [3] * 11 + [4] * 56 + [5] * 17 + [6] * 14
weight = Bream_weight + Roach_weight + Whitefish_weight + Parkki_weight + Perch_weight + Pike_weight + Smelt_weight
height = Bream_Height + Roach_Height + Whitefish_Height + Parkki_Height + Perch_Height + Pike_Height + Smelt_Height


## 6가지 물고기 별 weight와 height 분포
plt.scatter(Bream_Height, Bream_weight)

plt.scatter(Roach_Height, Roach_weight)
plt.scatter(Whitefish_Height, Whitefish_weight)
plt.scatter(Parkki_Height, Parkki_weight)
plt.scatter(Perch_Height, Perch_weight)
plt.scatter(Pike_Height, Pike_weight)
plt.scatter(Smelt_Height, Smelt_weight)

plt.xlabel('length')
plt.ylabel('Height')
plt.show()

target = [0]*35 + [1] * 20 + [2] * 6 + [3] * 11 + [4] * 56 + [5] * 17 + [6] * 14
fish_data = np.column_stack((weight, height))

fish_target = np.array(target)

error_rate = []
# dataset = [[h,w] for h,w in zip(weight, height)]
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, train_size = 0.7, random_state= 11)

## Kvalue의 변화에 따른 오차 확인
for i in range(1,51):

    kn = KNeighborsClassifier(n_neighbors=i) 
    kn.fit(train_input, train_target)
    predict_i = kn.predict(test_input)
    error_rate.append(np.mean(predict_i != test_target))


plt.figure(figsize=(30,8))
plt.plot(range(50),error_rate,marker="o",markerfacecolor="green",
         linestyle="dashed",color="red",markersize=15)
plt.title("Error rate vs k value",fontsize=20)
plt.xlabel("k- values",fontsize=40)
plt.ylabel("error rate",fontsize=40)
plt.xticks(range(1,50))
plt.show()

kn = KNeighborsClassifier(n_neighbors=12) 
kn.fit(train_input, train_target)
predict_i = kn.predict(test_input)
error_rate.append(np.mean(predict_i != test_target))

predict_i = predict_i.reshape(-1,1)
test_target = test_target.reshape(-1,1)
# test_target = test_target.tolist()
# predict_i = predict_i.tolist()
# breakpoint()
cm = confusion_matrix(test_target, predict_i, labels = [0,1,2,3,4,5,6])
disp  = ConfusionMatrixDisplay(confusion_matrix= cm , display_labels =  [0,1,2,3,4,5,6])

disp.plot()
plt.show()
## 오차가 너무 높게 나와서 줄일 수 있는 방안에 대해 고민!
