DATASET = 'Churn_Modelling.csv'

import numpy as np
import pandas as pd

dataset = pd.read_csv(DATASET)
X = dataset.iloc[:, 3:-1].values # Assume relevant features are in columns 3+
y = dataset.iloc[:, -1].values

# Assume column 2 is binary categorical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Assume column 1 is multi categorical
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train).astype(np.float32)
X_test = sc.transform(X_test).astype(np.float32)

y_train = np.array([[item] for item in y_train]).astype(np.float32)
