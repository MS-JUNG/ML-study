from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
wine = pd.read_csv('wine.txt')



data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2 ,random_state = 42)


# t_input, val_input, t_target, val_target = train_test_split(train_input, train_target, test_size = 0.2, random_state = 42)
dt = DecisionTreeClassifier()


scores = cross_validate(dt,train_input,train_target, cv = StratifiedKFold())

params = {'min_impourtiy_'}
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb,train_input, train_target, return_train_score = True, n_jobs = -1)

breakpoint()

