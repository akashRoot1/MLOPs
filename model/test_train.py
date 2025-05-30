import pandas as pd
import pickle

def test_prediction_above_zero():
    model = pickle.load(open('model/model.pkl', 'rb'))
    test_data = pd.DataFrame({'TV': [100, 200]})
    predictions = model.predict(test_data)
    assert all(p > 0 for p in predictions), "Predictions not greater than 0"

if __name__ == "__main__":
    test_prediction_above_zero()
    print("Test passed!")
