import os
import pandas as pd
from scripts import retrain_model
import joblib

def test_load_data():
    df = retrain_model.load_data(filepath="processed/processed_data.csv")
    assert not df.empty, "Processed data should not be empty"
    assert set(["TV", "Radio", "Newspaper", "Sales"]).issubset(df.columns), "Columns missing"

def test_train_model():
    df = retrain_model.load_data(filepath="processed/processed_data.csv")
    model = retrain_model.train_model(df)
    # Check if model coefficients exist
    assert hasattr(model, "coef_"), "Model should have coefficients"
    # Check prediction on first row
    X = df[["TV", "Radio", "Newspaper"]].iloc[[0]]
    pred = model.predict(X)
    assert isinstance(pred[0], float), "Prediction should be a float"
