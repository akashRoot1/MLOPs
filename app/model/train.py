import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv('data/dataset.csv', sep='\s+')

# Use only TV as feature
X = data[['TV']]
y = data['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model/model.pkl")
print(data.columns)

