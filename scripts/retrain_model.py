import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def load_data(filepath="processed/processed_data.csv"):
    return pd.read_csv(filepath)

def train_model(df):
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]
    
    model = LinearRegression()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    print("✅ Model saved to model/model.pkl")
