import os
import pandas as pd
from scripts import ingest_preprocess

def test_ingest_data():
    # Use the sample dataset path
    df = ingest_preprocess.ingest_data(filepath="data/dataset.csv")
    assert not df.empty, "DataFrame should not be empty"
    assert set(["TV", "Radio", "Newspaper", "Sales"]).issubset(df.columns), "Columns missing"
    
def test_preprocess_data():
    df = ingest_preprocess.ingest_data(filepath="data/dataset.csv")
    df_processed = ingest_preprocess.preprocess_data(df)
    assert not df_processed.isnull().values.any(), "No missing values after preprocessing"
    # Check if columns are scaled (mean approx 0, std approx 1)
    for col in ["TV", "Radio", "Newspaper"]:
        assert abs(df_processed[col].mean()) < 1e-6, f"{col} mean should be ~0 after scaling"
        assert abs(df_processed[col].std() - 1) < 1e-6, f"{col} std should be ~1 after scaling"
