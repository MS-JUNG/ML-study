import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree




wine = pd.read_csv('wine.txt')

print(wine.info())
print(wine.describe())

print(wine.tail(100))
data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2 ,random_state = 42)

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled =  ss.transform(test_input)
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(train_scaled, train_target)

plt.figure()
plot_tree(dt, max_depth = 1, filled = True, feature_names = ['alcohol','sugar', 'pH'])

plt.show()


