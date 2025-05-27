import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/dataset.csv')
X = data[['TV']]  # Only 1 feature for simplicity
y = data['Sales']

model = LinearRegression()
model.fit(X, y)

with open('app/model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
