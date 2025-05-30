import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def ingest_data(filepath="data/dataset.csv"):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df = df.dropna()
    
    # Features to scale
    numeric_cols = ["TV", "Radio", "Newspaper"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

if __name__ == "__main__":
    df = ingest_data()
    df = preprocess_data(df)
    os.makedirs("processed", exist_ok=True)
    df.to_csv("processed/processed_data.csv", index=False)
    print("✅ Preprocessed data saved to processed/processed_data.csv")
