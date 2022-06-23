from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data = load_iris()



datas = data.data
target = data.target 
train_input, test_input, train_target, test_target = train_test_split(datas, target, test_size = 0.4, random_state = 42 )

model = DecisionTreeClassifier()
model.fit(train_input, train_target)
print(model.score(test_input, test_target ))

breakpoint()